//! Port of packages/coding-agent/src/modes/interactive/components/tree-selector.ts

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use indexmap::IndexMap;

use pi_tui::components::input::Input;
use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::components::truncated_text::TruncatedText;
use pi_tui::keybindings::get_keybindings;
use pi_tui::tui::{Component, Container, Focusable};
use pi_tui::utils::truncate_to_width;

use super::super::theme::theme::theme;
use super::dynamic_border::{ColorFn, DynamicBorder};
use super::keybinding_hints::{key_hint, key_text, KeyTextOptions};
use crate::modes::agent_connection::types::{
    AgentConnectionSessionEntry, AgentConnectionSessionTreeNode,
};

/// Gutter info: position (displayIndent where connector was) and whether to show │
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GutterInfo {
    /// displayIndent level where the connector was shown
    pub position: usize,
    /// true = show │, false = show spaces
    pub show: bool,
}

/// Flattened tree node for navigation
#[derive(Debug, Clone)]
pub struct FlatNode {
    pub node: AgentConnectionSessionTreeNode,
    /// Indentation level (each level = 3 chars)
    pub indent: usize,
    /// Whether to show connector (├─ or └─) - true if parent has multiple children
    pub show_connector: bool,
    /// If showConnector, true = last sibling (└─), false = not last (├─)
    pub is_last: bool,
    /// Gutter info for each ancestor branch point
    pub gutters: Vec<GutterInfo>,
    /// True if this node is a root under a virtual branching root (multiple roots)
    pub is_virtual_root_child: bool,
}

/// Filter mode for tree display
pub type FilterMode = &'static str;
pub const FILTER_MODES: [FilterMode; 5] =
    ["default", "no-tools", "user-only", "labeled-only", "all"];

/// Tool call info for lookup
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolCallInfo {
    pub name: String,
    pub arguments: serde_json::Map<String, serde_json::Value>,
}

/// `containsActive` as a map keyed by entry id.
///
/// The TypeScript keys the map by object identity; entry ids are unique per
/// tree, so the port keys by id and keeps the same result.
fn contains_active_map(
    roots: &[AgentConnectionSessionTreeNode],
    leaf_id: Option<&str>,
) -> HashMap<String, bool> {
    let mut contains_active: HashMap<String, bool> = HashMap::new();

    // Build list in pre-order, then process in reverse for post-order effect
    let mut all_nodes: Vec<&AgentConnectionSessionTreeNode> = Vec::new();
    let mut pre_order_stack: Vec<&AgentConnectionSessionTreeNode> = roots.iter().collect();
    while let Some(node) = pre_order_stack.pop() {
        all_nodes.push(node);
        // Push children in reverse so they're processed left-to-right
        for i in (0..node.children.len()).rev() {
            pre_order_stack.push(&node.children[i]);
        }
    }
    // Process in reverse (post-order): children before parents
    for node in all_nodes.iter().rev() {
        let mut has = leaf_id.map(|leaf| node.entry.id() == leaf).unwrap_or(false);
        for child in &node.children {
            if contains_active
                .get(child.entry.id())
                .copied()
                .unwrap_or(false)
            {
                has = true;
            }
        }
        contains_active.insert(node.entry.id().to_string(), has);
    }
    contains_active
}

/// Tree list component with selection and ASCII art visualization.
pub struct TreeList {
    flat_nodes: Vec<FlatNode>,
    filtered_nodes: Vec<FlatNode>,
    selected_index: usize,
    current_leaf_id: Option<String>,
    max_visible_lines: usize,
    filter_mode: FilterMode,
    search_query: String,
    tool_call_map: HashMap<String, ToolCallInfo>,
    multiple_roots: bool,
    show_label_timestamps: bool,
    active_path_ids: HashSet<String>,
    visible_parent_map: IndexMap<String, Option<String>>,
    visible_children_map: IndexMap<Option<String>, Vec<String>>,
    last_selected_id: Option<String>,
    folded_nodes: HashSet<String>,

    pub on_select: Option<Box<dyn FnMut(&str)>>,
    pub on_cancel: Option<Box<dyn FnMut()>>,
    pub on_label_edit: Option<Box<dyn FnMut(String, Option<String>)>>,
}

impl TreeList {
    pub fn new(
        tree: &[AgentConnectionSessionTreeNode],
        current_leaf_id: Option<String>,
        max_visible_lines: usize,
        initial_selected_id: Option<String>,
        initial_filter_mode: Option<FilterMode>,
    ) -> Self {
        let mut list = Self {
            flat_nodes: Vec::new(),
            filtered_nodes: Vec::new(),
            selected_index: 0,
            current_leaf_id: current_leaf_id.clone(),
            max_visible_lines,
            filter_mode: initial_filter_mode.unwrap_or("default"),
            search_query: String::new(),
            tool_call_map: HashMap::new(),
            multiple_roots: tree.len() > 1,
            show_label_timestamps: false,
            active_path_ids: HashSet::new(),
            visible_parent_map: IndexMap::new(),
            visible_children_map: IndexMap::new(),
            last_selected_id: None,
            folded_nodes: HashSet::new(),
            on_select: None,
            on_cancel: None,
            on_label_edit: None,
        };
        list.multiple_roots = tree.len() > 1;
        list.flat_nodes = list.flatten_tree(tree);
        list.build_active_path();
        list.apply_filter();

        // Start with initialSelectedId if provided, otherwise current leaf
        let target_id = initial_selected_id.or(current_leaf_id);
        list.selected_index = list.find_nearest_visible_index(target_id.as_deref());
        list.last_selected_id = list
            .filtered_nodes
            .get(list.selected_index)
            .map(|flat_node| flat_node.node.entry.id().to_string());
        list
    }

    /// Find the index of the nearest visible entry, walking up the parent chain if needed.
    /// Returns the index in filteredNodes, or the last index as fallback.
    fn find_nearest_visible_index(&self, entry_id: Option<&str>) -> usize {
        if self.filtered_nodes.is_empty() {
            return 0;
        }

        // Build a map for parent lookup
        let mut entry_map: HashMap<&str, &FlatNode> = HashMap::new();
        for flat_node in &self.flat_nodes {
            entry_map.insert(flat_node.node.entry.id(), flat_node);
        }

        // Build a map of visible entry IDs to their indices in filteredNodes
        let mut visible_id_to_index: HashMap<&str, usize> = HashMap::new();
        for (index, node) in self.filtered_nodes.iter().enumerate() {
            visible_id_to_index.insert(node.node.entry.id(), index);
        }

        // Walk from entryId up to root, looking for a visible entry
        let mut current_id = entry_id.map(|id| id.to_string());
        while let Some(id) = current_id.clone() {
            if let Some(index) = visible_id_to_index.get(id.as_str()) {
                return *index;
            }
            let node = entry_map.get(id.as_str());
            match node {
                Some(node) => {
                    current_id = node.node.entry.parent_id().map(|parent| parent.to_string())
                }
                None => break,
            }
        }

        // Fallback: last visible entry
        self.filtered_nodes.len() - 1
    }

    /// Build the set of entry IDs on the path from root to current leaf
    fn build_active_path(&mut self) {
        self.active_path_ids.clear();
        let Some(leaf_id) = self.current_leaf_id.clone() else {
            return;
        };

        // Build a map of id -> entry for parent lookup
        let mut entry_map: HashMap<String, Option<String>> = HashMap::new();
        for flat_node in &self.flat_nodes {
            entry_map.insert(
                flat_node.node.entry.id().to_string(),
                flat_node.node.entry.parent_id().map(|id| id.to_string()),
            );
        }

        // Walk from leaf to root
        let mut current_id: Option<String> = Some(leaf_id);
        while let Some(id) = current_id.clone() {
            self.active_path_ids.insert(id.clone());
            match entry_map.get(&id) {
                Some(Some(parent)) => current_id = Some(parent.clone()),
                Some(None) => break,
                None => break,
            }
        }
    }

    /// Port of `flattenTree(roots)`.
    fn flatten_tree(&mut self, roots: &[AgentConnectionSessionTreeNode]) -> Vec<FlatNode> {
        let mut result: Vec<FlatNode> = Vec::new();
        self.tool_call_map.clear();

        // Indentation rules:
        // - At indent 0: stay at 0 unless parent has >1 children (then +1)
        // - At indent 1: children always go to indent 2 (visual grouping of subtree)
        // - At indent 2+: stay flat for single-child chains, +1 only if parent branches

        // Stack items: [node, indent, justBranched, showConnector, isLast, gutters, isVirtualRootChild]
        struct StackItem<'a> {
            node: &'a AgentConnectionSessionTreeNode,
            indent: usize,
            just_branched: bool,
            show_connector: bool,
            is_last: bool,
            gutters: Vec<GutterInfo>,
            is_virtual_root_child: bool,
        }
        let mut stack: Vec<StackItem> = Vec::new();

        // Determine which subtrees contain the active leaf (to sort current branch first)
        // Use iterative post-order traversal to avoid stack overflow
        let contains_active = contains_active_map(roots, self.current_leaf_id.as_deref());

        // Add roots in reverse order, prioritizing the one containing the active leaf
        // If multiple roots, treat them as children of a virtual root that branches
        let multiple_roots = roots.len() > 1;
        let mut ordered_roots: Vec<&AgentConnectionSessionTreeNode> = roots.iter().collect();
        // `[...roots].sort((a, b) => Number(containsActive.get(b)) - Number(containsActive.get(a)))`.
        // Rust's `sort_by_key` is stable, so the input order inside each group is
        // the same order the TypeScript sort preserves.
        ordered_roots.sort_by_key(|node| {
            if contains_active
                .get(node.entry.id())
                .copied()
                .unwrap_or(false)
            {
                0
            } else {
                1
            }
        });
        for i in (0..ordered_roots.len()).rev() {
            let is_last = i == ordered_roots.len() - 1;
            stack.push(StackItem {
                node: ordered_roots[i],
                indent: if multiple_roots { 1 } else { 0 },
                just_branched: multiple_roots,
                show_connector: multiple_roots,
                is_last,
                gutters: Vec::new(),
                is_virtual_root_child: multiple_roots,
            });
        }

        while let Some(item) = stack.pop() {
            let StackItem {
                node,
                indent,
                just_branched,
                show_connector,
                is_last,
                gutters,
                is_virtual_root_child,
            } = item;

            // Extract tool calls from assistant messages for later lookup
            // (`entry.message.content` is duck-typed in the TypeScript: blocks with
            // `type === "toolCall"`. The port reads the same blocks from the
            // serialized message, which keeps the field names identical.)
            if let AgentConnectionSessionEntry::Message { message, .. } = &node.entry {
                if message.role() == "assistant" {
                    let message_value =
                        serde_json::to_value(message).unwrap_or(serde_json::Value::Null);
                    if let Some(blocks) = message_value
                        .get("content")
                        .and_then(|value| value.as_array())
                    {
                        for block in blocks {
                            if block.get("type").and_then(|kind| kind.as_str()) != Some("toolCall")
                            {
                                continue;
                            }
                            let id = block.get("id").and_then(|id| id.as_str()).unwrap_or("");
                            let name = block
                                .get("name")
                                .and_then(|name| name.as_str())
                                .unwrap_or("");
                            let arguments = block
                                .get("arguments")
                                .and_then(|arguments| arguments.as_object())
                                .cloned()
                                .unwrap_or_default();
                            self.tool_call_map.insert(
                                id.to_string(),
                                ToolCallInfo {
                                    name: name.to_string(),
                                    arguments,
                                },
                            );
                        }
                    }
                }
            }

            result.push(FlatNode {
                // Flat rows only need this entry's metadata. Retaining every
                // descendant here makes a chain's copies quadratic in its depth.
                node: AgentConnectionSessionTreeNode {
                    entry: node.entry.clone(),
                    label: node.label.clone(),
                    label_timestamp: node.label_timestamp.clone(),
                    children: Vec::new(),
                },
                indent,
                show_connector,
                is_last,
                gutters: gutters.clone(),
                is_virtual_root_child,
            });

            let multiple_children = node.children.len() > 1;

            // Order children so the branch containing the active leaf comes first
            let mut ordered_children: Vec<&AgentConnectionSessionTreeNode> = node.children.iter().collect();
            ordered_children.sort_by_key(|child| {
                if contains_active
                    .get(child.entry.id())
                    .copied()
                    .unwrap_or(false)
                {
                    0
                } else {
                    1
                }
            });

            // Calculate child indent
            let child_indent = if multiple_children {
                // Parent branches: children get +1
                indent + 1
            } else if just_branched && indent > 0 {
                // First generation after a branch: +1 for visual grouping
                indent + 1
            } else {
                // Single-child chain: stay flat
                indent
            };

            // Build gutters for children
            // If this node showed a connector, add a gutter entry for descendants
            // Only add gutter if connector is actually displayed (not suppressed for virtual root children)
            let connector_displayed = show_connector && !is_virtual_root_child;
            // When connector is displayed, add a gutter entry at the connector's position
            // Connector is at position (displayIndent - 1), so gutter should be there too
            let current_display_indent = if self.multiple_roots {
                indent.saturating_sub(1)
            } else {
                indent
            };
            let connector_position = current_display_indent.saturating_sub(1);
            let child_gutters: Vec<GutterInfo> = if connector_displayed {
                let mut child_gutters = gutters.clone();
                child_gutters.push(GutterInfo {
                    position: connector_position,
                    show: !is_last,
                });
                child_gutters
            } else {
                gutters.clone()
            };

            // Add children in reverse order
            for i in (0..ordered_children.len()).rev() {
                let child_is_last = i == ordered_children.len() - 1;
                stack.push(StackItem {
                    node: ordered_children[i],
                    indent: child_indent,
                    just_branched: multiple_children,
                    show_connector: multiple_children,
                    is_last: child_is_last,
                    gutters: child_gutters.clone(),
                    is_virtual_root_child: false,
                });
            }
        }

        result
    }

    /// Port of `applyFilter()`.
    fn apply_filter(&mut self) {
        // Update lastSelectedId only when we have a valid selection (non-empty list)
        // This preserves the selection when switching through empty filter results
        if !self.filtered_nodes.is_empty() {
            if let Some(node) = self.filtered_nodes.get(self.selected_index) {
                self.last_selected_id = Some(node.node.entry.id().to_string());
            }
        }

        let search_tokens: Vec<String> = self
            .search_query
            .to_lowercase()
            .split_whitespace()
            .filter(|token| !token.is_empty())
            .map(|token| token.to_string())
            .collect();

        let current_leaf_id = self.current_leaf_id.clone();
        let filter_mode = self.filter_mode;
        let search_query = self.search_query.clone();
        let mut filtered: Vec<FlatNode> = self.flat_nodes
            .iter()
            .filter(|flat_node| {
                let entry = &flat_node.node.entry;
                let is_current_leaf = Some(entry.id()) == current_leaf_id.as_deref();

                // Skip assistant messages with only tool calls (no text) unless error/aborted
                // Always show current leaf so active position is visible
                if let AgentConnectionSessionEntry::Message { message, .. } = &entry {
                    if message.role() == "assistant" && !is_current_leaf {
                        let message_value =
                            serde_json::to_value(message).unwrap_or(serde_json::Value::Null);
                        let has_text = has_text_content(message_value.get("content"));
                        let stop_reason = message_value
                            .get("stopReason")
                            .and_then(|value| value.as_str())
                            .unwrap_or("");
                        let is_error_or_aborted = !stop_reason.is_empty()
                            && stop_reason != pi_ai::types::STOP_REASON_STOP
                            && stop_reason != pi_ai::types::STOP_REASON_TOOL_USE;
                        // Only hide if no text AND not an error/aborted message
                        if !has_text && !is_error_or_aborted {
                            return false;
                        }
                    }
                }

                // Apply filter mode
                let passes_filter;
                // Entry types hidden in default view (settings/bookkeeping)
                let is_settings_entry = is_settings_entry(&entry);

                match filter_mode {
                    "user-only" => {
                        // Just user messages
                        passes_filter = matches!(
                            &entry,
                            AgentConnectionSessionEntry::Message { message, .. }
                                if message.role() == "user"
                        );
                    }
                    "no-tools" => {
                        // Default minus tool results
                        passes_filter = !is_settings_entry
                            && !matches!(
                                &entry,
                                AgentConnectionSessionEntry::Message { message, .. }
                                    if message.role() == "toolResult"
                            );
                    }
                    "labeled-only" => {
                        // Just labeled entries
                        passes_filter = flat_node.node.label.is_some();
                    }
                    "all" => {
                        // Show everything
                        passes_filter = true;
                    }
                    _ => {
                        // Default mode: hide settings/bookkeeping entries
                        passes_filter = !is_settings_entry;
                    }
                }

                if !passes_filter {
                    return false;
                }

                // Apply search filter
                if !search_tokens.is_empty() {
                    let node_text = get_searchable_text(&flat_node.node).to_lowercase();
                    return search_tokens.iter().all(|token| node_text.contains(token));
                }

                true
            })
            .cloned()
            .collect();

        let _ = search_query;

        // Filter out descendants of folded nodes.
        if !self.folded_nodes.is_empty() {
            let mut skip_set: HashSet<String> = HashSet::new();
            for flat_node in &self.flat_nodes {
                let id = flat_node.node.entry.id().to_string();
                let parent_id = flat_node.node.entry.parent_id().map(|id| id.to_string());
                if let Some(parent_id) = parent_id {
                    if self.folded_nodes.contains(&parent_id) || skip_set.contains(&parent_id) {
                        skip_set.insert(id);
                    }
                }
            }
            filtered.retain(|flat_node| !skip_set.contains(flat_node.node.entry.id()));
        }

        self.filtered_nodes = filtered;

        // Recalculate visual structure (indent, connectors, gutters) based on visible tree
        self.recalculate_visual_structure();

        // Try to preserve cursor on the same node, or find nearest visible ancestor
        if self.last_selected_id.is_some() {
            let last = self.last_selected_id.clone();
            self.selected_index = self.find_nearest_visible_index(last.as_deref());
        } else if self.selected_index >= self.filtered_nodes.len() {
            // Clamp index if out of bounds
            self.selected_index = self.filtered_nodes.len().saturating_sub(1);
        }

        // Update lastSelectedId to the actual selection (may have changed due to parent walk)
        if !self.filtered_nodes.is_empty() {
            if let Some(node) = self.filtered_nodes.get(self.selected_index) {
                self.last_selected_id = Some(node.node.entry.id().to_string());
            }
        }
    }

    /**
     * Recompute indentation/connectors for the filtered view
     *
     * Filtering can hide intermediate entries; descendants attach to the nearest visible ancestor.
     * Keep indentation semantics aligned with flattenTree() so single-child chains don't drift right.
     */
    fn recalculate_visual_structure(&mut self) {
        if self.filtered_nodes.is_empty() {
            return;
        }

        let visible_ids: HashSet<String> = self
            .filtered_nodes
            .iter()
            .map(|n| n.node.entry.id().to_string())
            .collect();

        // Build entry map for efficient parent lookup (using full tree)
        let mut entry_map: HashMap<String, String> = HashMap::new();
        for flat_node in &self.flat_nodes {
            entry_map.insert(
                flat_node.node.entry.id().to_string(),
                flat_node
                    .node
                    .entry
                    .parent_id()
                    .map(|id| id.to_string())
                    .unwrap_or_default(),
            );
        }

        // Find nearest visible ancestor for a node
        let find_visible_ancestor =
            |entry_map: &HashMap<String, String>, node_id: &str| -> Option<String> {
                let mut current_id = entry_map.get(node_id).cloned().filter(|id| !id.is_empty());
                while let Some(id) = current_id.clone() {
                    if visible_ids.contains(&id) {
                        return Some(id);
                    }
                    current_id = entry_map.get(&id).cloned().filter(|id| !id.is_empty());
                }
                None
            };

        // Build visible tree structure:
        // - visibleParent: nodeId → nearest visible ancestor (or null for roots)
        // - visibleChildren: parentId → list of visible children (in filteredNodes order)
        let mut visible_parent: IndexMap<String, Option<String>> = IndexMap::new();
        let mut visible_children: IndexMap<Option<String>, Vec<String>> = IndexMap::new();
        visible_children.insert(None, Vec::new()); // root-level nodes

        for flat_node in &self.filtered_nodes {
            let node_id = flat_node.node.entry.id().to_string();
            let ancestor_id = find_visible_ancestor(&entry_map, &node_id);
            visible_parent.insert(node_id.clone(), ancestor_id.clone());

            visible_children
                .entry(ancestor_id)
                .or_default()
                .push(node_id);
        }

        // Update multipleRoots based on visible roots
        let visible_root_ids = visible_children.get(&None).cloned().unwrap_or_default();
        self.multiple_roots = visible_root_ids.len() > 1;

        // Build a map for quick lookup: nodeId → FlatNode
        let mut filtered_node_map: HashMap<String, FlatNode> = HashMap::new();
        for flat_node in &self.filtered_nodes {
            filtered_node_map.insert(flat_node.node.entry.id().to_string(), flat_node.clone());
        }

        // DFS over the visible tree using flattenTree() indentation semantics
        struct StackItem {
            node_id: String,
            indent: usize,
            just_branched: bool,
            show_connector: bool,
            is_last: bool,
            gutters: Vec<GutterInfo>,
            is_virtual_root_child: bool,
        }
        let mut stack: Vec<StackItem> = Vec::new();

        // Add visible roots in reverse order (to process in forward order via stack)
        for i in (0..visible_root_ids.len()).rev() {
            let is_last = i == visible_root_ids.len() - 1;
            stack.push(StackItem {
                node_id: visible_root_ids[i].clone(),
                indent: if self.multiple_roots { 1 } else { 0 },
                just_branched: self.multiple_roots,
                show_connector: self.multiple_roots,
                is_last,
                gutters: Vec::new(),
                is_virtual_root_child: self.multiple_roots,
            });
        }

        while let Some(item) = stack.pop() {
            let StackItem {
                node_id,
                indent,
                just_branched,
                show_connector,
                is_last,
                gutters,
                is_virtual_root_child,
            } = item;

            let Some(mut flat_node) = filtered_node_map.get(&node_id).cloned() else {
                continue;
            };

            // Update this node's visual properties
            flat_node.indent = indent;
            flat_node.show_connector = show_connector;
            flat_node.is_last = is_last;
            flat_node.gutters = gutters.clone();
            flat_node.is_virtual_root_child = is_virtual_root_child;
            filtered_node_map.insert(node_id.clone(), flat_node.clone());

            // Get visible children of this node
            let children = visible_children
                .get(&Some(node_id.clone()))
                .cloned()
                .unwrap_or_default();
            let multiple_children = children.len() > 1;

            // Child indent follows flattenTree(): branch points (and first generation after a branch) shift +1
            let child_indent = if multiple_children {
                indent + 1
            } else if just_branched && indent > 0 {
                indent + 1
            } else {
                indent
            };

            // Child gutters follow flattenTree() connector/gutter rules
            let connector_displayed = show_connector && !is_virtual_root_child;
            let current_display_indent = if self.multiple_roots {
                indent.saturating_sub(1)
            } else {
                indent
            };
            let connector_position = current_display_indent.saturating_sub(1);
            let child_gutters: Vec<GutterInfo> = if connector_displayed {
                let mut child_gutters = gutters.clone();
                child_gutters.push(GutterInfo {
                    position: connector_position,
                    show: !is_last,
                });
                child_gutters
            } else {
                gutters.clone()
            };

            // Add children in reverse order (to process in forward order via stack)
            for i in (0..children.len()).rev() {
                let child_is_last = i == children.len() - 1;
                stack.push(StackItem {
                    node_id: children[i].clone(),
                    indent: child_indent,
                    just_branched: multiple_children,
                    show_connector: multiple_children,
                    is_last: child_is_last,
                    gutters: child_gutters.clone(),
                    is_virtual_root_child: false,
                });
            }
        }

        // Write the recalculated structure back onto the filtered nodes.
        for node in self.filtered_nodes.iter_mut() {
            if let Some(updated) = filtered_node_map.get(node.node.entry.id()) {
                node.indent = updated.indent;
                node.show_connector = updated.show_connector;
                node.is_last = updated.is_last;
                node.gutters = updated.gutters.clone();
                node.is_virtual_root_child = updated.is_virtual_root_child;
            }
        }

        // Store visible tree maps for ancestor/descendant lookups in navigation
        self.visible_parent_map = visible_parent;
        self.visible_children_map = visible_children;
    }

    /// `invalidate()`.
    pub fn invalidate_list(&mut self) {}

    pub fn get_search_query(&self) -> &str {
        &self.search_query
    }

    pub fn get_selected_node(&self) -> Option<&AgentConnectionSessionTreeNode> {
        self.filtered_nodes
            .get(self.selected_index)
            .map(|node| &node.node)
    }

    pub fn selected_index(&self) -> usize {
        self.selected_index
    }

    pub fn update_node_label(
        &mut self,
        entry_id: &str,
        label: Option<String>,
        label_timestamp: Option<String>,
    ) {
        let mut updated = false;
        for flat_node in self.flat_nodes.iter_mut() {
            if flat_node.node.entry.id() == entry_id {
                flat_node.node.label = label.clone();
                flat_node.node.label_timestamp = match label {
                    Some(_) => Some(label_timestamp.unwrap_or_else(|| now_iso8601())),
                    None => None,
                };
                updated = true;
                break;
            }
        }
        if updated {
            // label changes can alter both search matches and labeled-only
            // membership, so the visible projection is stale until rebuilt.
            self.apply_filter();
        }
    }

    fn get_status_labels(&self) -> String {
        let mut labels = String::new();
        match self.filter_mode {
            "no-tools" => labels.push_str(" [no-tools]"),
            "user-only" => labels.push_str(" [user]"),
            "labeled-only" => labels.push_str(" [labeled]"),
            "all" => labels.push_str(" [all]"),
            _ => {}
        }
        if self.show_label_timestamps {
            labels.push_str(" [+label time]");
        }
        labels
    }

    /// Port of `render(width)`.
    pub fn render(&mut self, width: f64) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();

        if self.filtered_nodes.is_empty() {
            lines.push(truncate_to_width(
                &theme().fg("muted", "  No entries found"),
                width,
                "",
                false,
            ));
            lines.push(truncate_to_width(
                &theme().fg("muted", &format!("  (0/0){}", self.get_status_labels())),
                width,
                "",
                false,
            ));
            return lines;
        }

        let start_index = self
            .selected_index
            .saturating_sub(self.max_visible_lines / 2)
            .min(
                self.filtered_nodes
                    .len()
                    .saturating_sub(self.max_visible_lines),
            );
        let end_index = (start_index + self.max_visible_lines).min(self.filtered_nodes.len());

        for i in start_index..end_index {
            let Some(flat_node) = self.filtered_nodes.get(i).cloned() else {
                continue;
            };
            let entry = flat_node.node.entry.clone();
            let is_selected = i == self.selected_index;

            // Build line: cursor + prefix + path marker + label + content
            let cursor = if is_selected {
                theme().fg("accent", "› ")
            } else {
                "  ".to_string()
            };

            // If multiple roots, shift display (roots at 0, not 1)
            let display_indent = if self.multiple_roots {
                flat_node.indent.saturating_sub(1)
            } else {
                flat_node.indent
            };

            // Build prefix with gutters at their correct positions
            // Each gutter has a position (displayIndent where its connector was shown)
            let connector = if flat_node.show_connector && !flat_node.is_virtual_root_child {
                if flat_node.is_last {
                    "└─ "
                } else {
                    "├─ "
                }
            } else {
                ""
            };
            let connector_position: Option<usize> = if !connector.is_empty() {
                Some(display_indent.saturating_sub(1))
            } else {
                None
            };

            // Build prefix char by char, placing gutters and connector at their positions
            let total_chars = display_indent * 3;
            let mut prefix_chars: Vec<&str> = Vec::new();
            let is_folded = self.folded_nodes.contains(entry.id());
            for i in 0..total_chars {
                let level = i / 3;
                let pos_in_level = i % 3;

                // Check if there's a gutter at this level
                let gutter = flat_node.gutters.iter().find(|g| g.position == level);
                if let Some(gutter) = gutter {
                    if pos_in_level == 0 {
                        prefix_chars.push(if gutter.show { "│" } else { " " });
                    } else {
                        prefix_chars.push(" ");
                    }
                } else if !connector.is_empty() && Some(level) == connector_position {
                    // Connector at this level, with fold indicator
                    if pos_in_level == 0 {
                        prefix_chars.push(if flat_node.is_last { "└" } else { "├" });
                    } else if pos_in_level == 1 {
                        let foldable = self.is_foldable(entry.id());
                        prefix_chars.push(if is_folded {
                            "⊞"
                        } else if foldable {
                            "⊟"
                        } else {
                            "─"
                        });
                    } else {
                        prefix_chars.push(" ");
                    }
                } else {
                    prefix_chars.push(" ");
                }
            }
            let prefix = prefix_chars.join("");

            // Fold marker for nodes without connectors (roots)
            let shows_fold_in_connector =
                flat_node.show_connector && !flat_node.is_virtual_root_child;
            let fold_marker = if is_folded && !shows_fold_in_connector {
                theme().fg("accent", "⊞ ")
            } else {
                String::new()
            };

            // Active path marker - shown right before the entry text
            let is_on_active_path = self.active_path_ids.contains(entry.id());
            let path_marker = if is_on_active_path {
                theme().fg("accent", "• ")
            } else {
                String::new()
            };

            let label = match &flat_node.node.label {
                Some(label) => theme().fg("warning", &format!("[{label}] ")),
                None => String::new(),
            };
            let label_timestamp = if self.show_label_timestamps
                && flat_node.node.label.is_some()
                && flat_node.node.label_timestamp.is_some()
            {
                theme().fg(
                    "muted",
                    &format!(
                        "{} ",
                        format_label_timestamp(flat_node.node.label_timestamp.as_deref().unwrap())
                    ),
                )
            } else {
                String::new()
            };
            let content = self.get_entry_display_text(&flat_node.node, is_selected);

            let mut line = format!(
                "{cursor}{}{fold_marker}{path_marker}{label}{label_timestamp}{content}",
                theme().fg("dim", &prefix)
            );
            if is_selected {
                line = theme().get_selection_background_color()(&line);
            }
            lines.push(truncate_to_width(&line, width, "", false));
        }

        lines.push(truncate_to_width(
            &theme().fg(
                "muted",
                &format!(
                    "  ({}/{}){}",
                    self.selected_index + 1,
                    self.filtered_nodes.len(),
                    self.get_status_labels()
                ),
            ),
            width,
            "",
            false,
        ));

        lines
    }

    /// Port of `getEntryDisplayText(node, isSelected)`.
    fn get_entry_display_text(
        &self,
        node: &AgentConnectionSessionTreeNode,
        is_selected: bool,
    ) -> String {
        let entry = &node.entry;
        let result: String;

        match entry {
            AgentConnectionSessionEntry::Message { message, .. } => {
                let role = message.role();
                match role {
                    "user" => {
                        let message_value =
                            serde_json::to_value(message).unwrap_or(serde_json::Value::Null);
                        let content = normalize_text(&extract_content(&message_value));
                        result = format!("{}{content}", theme().fg("accent", "user: "));
                    }
                    "assistant" => {
                        let message_value =
                            serde_json::to_value(message).unwrap_or(serde_json::Value::Null);
                        let text_content = normalize_text(&extract_content(&message_value));
                        let stop_reason = message_value
                            .get("stopReason")
                            .and_then(|value| value.as_str())
                            .unwrap_or("")
                            .to_string();
                        let error_message = message_value
                            .get("errorMessage")
                            .and_then(|value| value.as_str())
                            .map(|value| value.to_string());
                        if !text_content.is_empty() {
                            result =
                                format!("{}{text_content}", theme().fg("success", "assistant: "));
                        } else if stop_reason == pi_ai::types::STOP_REASON_ABORTED {
                            result = format!(
                                "{}{}",
                                theme().fg("success", "assistant: "),
                                theme().fg("muted", "(aborted)")
                            );
                        } else if let Some(error_message) = error_message {
                            let error_message: String =
                                normalize_text(&error_message).chars().take(80).collect();
                            result = format!(
                                "{}{}",
                                theme().fg("success", "assistant: "),
                                theme().fg("error", &error_message)
                            );
                        } else {
                            result = format!(
                                "{}{}",
                                theme().fg("success", "assistant: "),
                                theme().fg("muted", "(no content)")
                            );
                        }
                    }
                    "toolResult" => {
                        let message_value =
                            serde_json::to_value(message).unwrap_or(serde_json::Value::Null);
                        let tool_call_id = message_value
                            .get("toolCallId")
                            .and_then(|value| value.as_str())
                            .unwrap_or("");
                        let tool_name = message_value
                            .get("toolName")
                            .and_then(|value| value.as_str())
                            .unwrap_or("tool");
                        let tool_call = self.tool_call_map.get(tool_call_id);
                        match tool_call {
                            Some(tool_call) => {
                                result = theme().fg(
                                    "muted",
                                    &format_tool_call(&tool_call.name, &tool_call.arguments),
                                );
                            }
                            None => {
                                result = theme().fg("muted", &format!("[{tool_name}]"));
                            }
                        }
                    }
                    _ => {
                        // `role === "bashExecution"` and the remaining custom roles.
                        let message_value =
                            serde_json::to_value(message).unwrap_or(serde_json::Value::Null);
                        if role == "bashExecution" {
                            let command = message_value
                                .get("command")
                                .and_then(|value| value.as_str())
                                .unwrap_or("");
                            result =
                                theme().fg("dim", &format!("[bash]: {}", normalize_text(command)));
                        } else {
                            result = theme().fg("dim", &format!("[{role}]"));
                        }
                    }
                }
            }
            AgentConnectionSessionEntry::CustomMessage {
                custom_type,
                content,
                ..
            } => {
                let content = match content {
                    serde_json::Value::String(text) => text.clone(),
                    other => extract_content_value(other),
                };
                result = format!(
                    "{}{}",
                    theme().fg("customMessageLabel", &format!("[{custom_type}]: ")),
                    normalize_text(&content)
                );
            }
            AgentConnectionSessionEntry::Compaction { tokens_before, .. } => {
                let tokens = (tokens_before / 1000.0).round();
                result = theme().fg(
                    "borderAccent",
                    &format!("[compaction: {tokens:.0}k tokens]"),
                );
            }
            AgentConnectionSessionEntry::BranchSummary { summary, .. } => {
                result = format!(
                    "{}{}",
                    theme().fg("warning", "[branch summary]: "),
                    normalize_text(summary)
                );
            }
            AgentConnectionSessionEntry::ModelChange { model_id, .. } => {
                result = theme().fg("dim", &format!("[model: {model_id}]"));
            }
            AgentConnectionSessionEntry::ThinkingLevelChange { thinking_level, .. } => {
                result = theme().fg("dim", &format!("[thinking: {thinking_level}]"));
            }
            AgentConnectionSessionEntry::ServiceTierChange { service_tier, .. } => {
                let service_tier_display = service_tier
                    .as_ref()
                    .and_then(|tier| tier.as_deref())
                    .unwrap_or("default");
                result = theme().fg("dim", &format!("[service tier: {service_tier_display}]"));
            }
            AgentConnectionSessionEntry::Custom { custom_type, .. } => {
                result = theme().fg("dim", &format!("[custom: {custom_type}]"));
            }
            AgentConnectionSessionEntry::ChildUsageAttribution { child_usage, .. } => {
                let input = json_number(child_usage, "input")
                    + json_number(child_usage, "cacheRead")
                    + json_number(child_usage, "cacheWrite");
                let output = json_number(child_usage, "output");
                result = theme().fg(
                    "dim",
                    &format!("[child usage: {input} input, {output} output]"),
                );
            }
            AgentConnectionSessionEntry::Label { label, .. } => {
                let label_display = match label {
                    Some(label) => label.clone(),
                    None => "(cleared)".to_string(),
                };
                result = theme().fg("dim", &format!("[label: {label_display}]"));
            }
            AgentConnectionSessionEntry::SessionInfo { name, .. } => {
                result = match name {
                    Some(name) => format!(
                        "{}{}{}",
                        theme().fg("dim", "[title: "),
                        theme().fg("dim", name),
                        theme().fg("dim", "]")
                    ),
                    None => format!(
                        "{}{}{}",
                        theme().fg("dim", "[title: "),
                        theme().italic(&theme().fg("dim", "empty")),
                        theme().fg("dim", "]")
                    ),
                };
            }
            _ => {
                result = String::new();
            }
        }

        if is_selected {
            theme().bold(&result)
        } else {
            result
        }
    }

    /// Port of `handleInput(keyData)`.
    pub fn handle_input(&mut self, key_data: &str) {
        let kb = get_keybindings();
        if self.filtered_nodes.is_empty()
            && (kb.matches(key_data, "tui.select.up")
                || kb.matches(key_data, "tui.select.down"))
        {
            return;
        }
        if kb.matches(key_data, "tui.select.up") {
            self.selected_index = if self.selected_index == 0 {
                self.filtered_nodes.len() - 1
            } else {
                self.selected_index - 1
            };
        } else if kb.matches(key_data, "tui.select.down") {
            self.selected_index = if self.selected_index == self.filtered_nodes.len() - 1 {
                0
            } else {
                self.selected_index + 1
            };
        } else if kb.matches(key_data, "app.tree.foldOrUp") {
            let current_id = self
                .filtered_nodes
                .get(self.selected_index)
                .map(|node| node.node.entry.id().to_string());
            if let Some(current_id) = current_id {
                if self.is_foldable(&current_id) && !self.folded_nodes.contains(&current_id) {
                    self.folded_nodes.insert(current_id);
                    self.apply_filter();
                } else {
                    self.selected_index = self.find_branch_segment_start("up");
                }
            }
        } else if kb.matches(key_data, "app.tree.unfoldOrDown") {
            let current_id = self
                .filtered_nodes
                .get(self.selected_index)
                .map(|node| node.node.entry.id().to_string());
            if let Some(current_id) = current_id {
                if self.folded_nodes.contains(&current_id) {
                    self.folded_nodes.remove(&current_id);
                    self.apply_filter();
                } else {
                    self.selected_index = self.find_branch_segment_start("down");
                }
            }
        } else if kb.matches(key_data, "tui.editor.cursorLeft")
            || kb.matches(key_data, "tui.select.pageUp")
        {
            // Page up
            self.selected_index = self.selected_index.saturating_sub(self.max_visible_lines);
        } else if kb.matches(key_data, "tui.editor.cursorRight")
            || kb.matches(key_data, "tui.select.pageDown")
        {
            // Page down
            self.selected_index = (self.selected_index + self.max_visible_lines)
                .min(self.filtered_nodes.len().saturating_sub(1));
        } else if kb.matches(key_data, "tui.select.confirm") {
            let selected = self
                .filtered_nodes
                .get(self.selected_index)
                .map(|node| node.node.entry.id().to_string());
            if let (Some(selected), Some(on_select)) = (selected, self.on_select.as_mut()) {
                on_select(&selected);
            }
        } else if kb.matches(key_data, "tui.select.cancel") {
            if !self.search_query.is_empty() {
                self.search_query = String::new();
                self.folded_nodes.clear();
                self.apply_filter();
            } else if let Some(on_cancel) = self.on_cancel.as_mut() {
                on_cancel();
            }
        } else if kb.matches(key_data, "app.tree.filter.default") {
            // Direct filter: default
            self.filter_mode = "default";
            self.folded_nodes.clear();
            self.apply_filter();
        } else if kb.matches(key_data, "app.tree.filter.noTools") {
            // Toggle filter: no-tools ↔ default
            self.filter_mode = if self.filter_mode == "no-tools" {
                "default"
            } else {
                "no-tools"
            };
            self.folded_nodes.clear();
            self.apply_filter();
        } else if kb.matches(key_data, "app.tree.filter.userOnly") {
            // Toggle filter: user-only ↔ default
            self.filter_mode = if self.filter_mode == "user-only" {
                "default"
            } else {
                "user-only"
            };
            self.folded_nodes.clear();
            self.apply_filter();
        } else if kb.matches(key_data, "app.tree.filter.labeledOnly") {
            // Toggle filter: labeled-only ↔ default
            self.filter_mode = if self.filter_mode == "labeled-only" {
                "default"
            } else {
                "labeled-only"
            };
            self.folded_nodes.clear();
            self.apply_filter();
        } else if kb.matches(key_data, "app.tree.filter.all") {
            // Toggle filter: all ↔ default
            self.filter_mode = if self.filter_mode == "all" {
                "default"
            } else {
                "all"
            };
            self.folded_nodes.clear();
            self.apply_filter();
        } else if kb.matches(key_data, "app.tree.filter.cycleBackward") {
            // Cycle filter backwards
            let current_index = FILTER_MODES
                .iter()
                .position(|mode| *mode == self.filter_mode)
                .unwrap_or(0);
            let next = (current_index + FILTER_MODES.len() - 1) % FILTER_MODES.len();
            self.filter_mode = FILTER_MODES[next];
            self.folded_nodes.clear();
            self.apply_filter();
        } else if kb.matches(key_data, "app.tree.filter.cycleForward") {
            // Cycle filter forwards: default → no-tools → user-only → labeled-only → all → default
            let current_index = FILTER_MODES
                .iter()
                .position(|mode| *mode == self.filter_mode)
                .unwrap_or(0);
            let next = (current_index + 1) % FILTER_MODES.len();
            self.filter_mode = FILTER_MODES[next];
            self.folded_nodes.clear();
            self.apply_filter();
        } else if kb.matches(key_data, "tui.editor.deleteCharBackward") {
            if !self.search_query.is_empty() {
                let mut chars: Vec<char> = self.search_query.chars().collect();
                chars.pop();
                self.search_query = chars.into_iter().collect();
                self.folded_nodes.clear();
                self.apply_filter();
            }
        } else if kb.matches(key_data, "app.tree.editLabel") {
            let selected = self
                .filtered_nodes
                .get(self.selected_index)
                .map(|node| (node.node.entry.id().to_string(), node.node.label.clone()));
            if let (Some((id, label)), Some(on_label_edit)) =
                (selected, self.on_label_edit.as_mut())
            {
                on_label_edit(id, label);
            }
        } else if kb.matches(key_data, "app.tree.toggleLabelTimestamp") {
            self.show_label_timestamps = !self.show_label_timestamps;
        } else {
            let has_control_chars = key_data.chars().any(|ch| {
                let code = ch as u32;
                code < 32 || code == 0x7f || (0x80..=0x9f).contains(&code)
            });
            if !has_control_chars && !key_data.is_empty() {
                self.search_query.push_str(key_data);
                self.folded_nodes.clear();
                self.apply_filter();
            }
        }
    }

    /**
     * Whether a node can be folded. A node is foldable if it has visible children
     * and is either a root (no visible parent) or a segment start (visible parent
     * has multiple visible children).
     */
    fn is_foldable(&self, entry_id: &str) -> bool {
        let children = match self.visible_children_map.get(&Some(entry_id.to_string())) {
            Some(children) => children,
            None => return false,
        };
        if children.is_empty() {
            return false;
        }
        let parent_id = self.visible_parent_map.get(entry_id);
        match parent_id {
            None | Some(None) => true,
            Some(Some(parent_id)) => {
                match self.visible_children_map.get(&Some(parent_id.clone())) {
                    Some(siblings) => siblings.len() > 1,
                    None => false,
                }
            }
        }
    }

    /**
     * Find the index of the next branch segment start in the given direction.
     * A segment start is the first child of a branch point.
     *
     * "up" walks the visible parent chain; "down" walks visible children
     * (always following the first child).
     */
    fn find_branch_segment_start(&self, direction: &str) -> usize {
        let selected_id = self
            .filtered_nodes
            .get(self.selected_index)
            .map(|node| node.node.entry.id().to_string());
        let Some(selected_id) = selected_id else {
            return self.selected_index;
        };

        let mut index_by_entry_id: HashMap<String, usize> = HashMap::new();
        for (i, node) in self.filtered_nodes.iter().enumerate() {
            index_by_entry_id.insert(node.node.entry.id().to_string(), i);
        }

        let mut current_id = selected_id;
        if direction == "down" {
            loop {
                let children = self
                    .visible_children_map
                    .get(&Some(current_id.clone()))
                    .cloned()
                    .unwrap_or_default();
                if children.is_empty() {
                    return index_by_entry_id.get(&current_id).copied().unwrap_or(0);
                }
                if children.len() > 1 {
                    return index_by_entry_id.get(&children[0]).copied().unwrap_or(0);
                }
                current_id = children[0].clone();
            }
        }

        // direction === "up"
        loop {
            let parent_id = self.visible_parent_map.get(&current_id).cloned().flatten();
            let Some(parent_id) = parent_id else {
                return index_by_entry_id.get(&current_id).copied().unwrap_or(0);
            };
            let children = self
                .visible_children_map
                .get(&Some(parent_id.clone()))
                .cloned()
                .unwrap_or_default();
            if children.len() > 1 {
                let segment_start = index_by_entry_id.get(&current_id).copied().unwrap_or(0);
                if segment_start < self.selected_index {
                    return segment_start;
                }
            }
            current_id = parent_id;
        }
    }
}

impl Component for TreeList {
    fn render(&mut self, width: f64) -> Vec<String> {
        TreeList::render(self, width)
    }

    fn handle_input(&mut self, data: &str) {
        TreeList::handle_input(self, data);
    }

    fn invalidate(&mut self) {
        TreeList::invalidate_list(self);
    }
}

/// Component that displays the current search query
pub struct SearchLine {
    query_source: Rc<RefCell<TreeList>>,
}

impl SearchLine {
    pub fn new(tree_list: Rc<RefCell<TreeList>>) -> Self {
        Self {
            query_source: tree_list,
        }
    }
}

impl Component for SearchLine {
    fn invalidate(&mut self) {}

    fn render(&mut self, width: f64) -> Vec<String> {
        let query = self.query_source.borrow().get_search_query().to_string();
        if !query.is_empty() {
            return vec![truncate_to_width(
                &format!(
                    "  {} {}",
                    theme().fg("muted", "Type to search:"),
                    theme().fg("accent", &query)
                ),
                width,
                "",
                false,
            )];
        }
        vec![truncate_to_width(
            &format!("  {}", theme().fg("muted", "Type to search:")),
            width,
            "",
            false,
        )]
    }

    fn handle_input(&mut self, _key_data: &str) {}
}

/// Label input component shown when editing a label
pub struct LabelInput {
    input: Input,
    entry_id: String,
    focused: bool,
    pub on_submit: Option<Box<dyn FnMut(String, Option<String>)>>,
    pub on_cancel: Option<Box<dyn FnMut()>>,
}

impl LabelInput {
    pub fn new(entry_id: &str, current_label: Option<&str>) -> Self {
        let mut input = Input::new();
        if let Some(current_label) = current_label {
            input.set_value(current_label.to_string());
        }
        Self {
            input,
            entry_id: entry_id.to_string(),
            focused: false,
            on_submit: None,
            on_cancel: None,
        }
    }

    /// Port of `handleInput(keyData)`.
    pub fn handle_input(&mut self, key_data: &str) {
        let kb = get_keybindings();
        if kb.matches(key_data, "tui.select.confirm") {
            let value = self.get_value().trim().to_string();
            if let Some(on_submit) = self.on_submit.as_mut() {
                on_submit(
                    self.entry_id.clone(),
                    if value.is_empty() { None } else { Some(value) },
                );
            }
        } else if kb.matches(key_data, "tui.select.cancel") {
            if let Some(on_cancel) = self.on_cancel.as_mut() {
                on_cancel();
            }
        } else {
            Component::handle_input(&mut self.input, key_data);
        }
    }

    pub fn get_value(&self) -> String {
        self.input.get_value().to_string()
    }
}

impl Focusable for LabelInput {
    fn focused(&self) -> bool {
        self.focused
    }

    fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
        Focusable::set_focused(&mut self.input, focused);
    }
}

impl Component for LabelInput {
    fn invalidate(&mut self) {}

    fn render(&mut self, width: f64) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        let indent = "  ";
        let available_width = width - indent.len() as f64;
        lines.push(truncate_to_width(
            &format!(
                "{indent}{}",
                theme().fg("muted", "Label (empty to remove):")
            ),
            width,
            "",
            false,
        ));
        for line in Component::render(&mut self.input, available_width) {
            lines.push(truncate_to_width(
                &format!("{indent}{line}"),
                width,
                "",
                false,
            ));
        }
        lines.push(truncate_to_width(
            &format!(
                "{indent}{}  {}",
                key_hint("tui.select.confirm", "save", &KeyTextOptions::default()),
                key_hint("tui.select.cancel", "cancel", &KeyTextOptions::default())
            ),
            width,
            "",
            false,
        ));
        lines
    }

    fn handle_input(&mut self, key_data: &str) {
        LabelInput::handle_input(self, key_data);
    }

    fn as_focusable(&mut self) -> Option<&mut dyn Focusable> {
        Some(self)
    }
}

/// Component that renders a session tree selector for navigation
pub struct TreeSelectorComponent {
    tree_list: Rc<RefCell<TreeList>>,
    label_input: Option<Rc<RefCell<LabelInput>>>,
    label_input_container: Rc<RefCell<Container>>,
    tree_container: Rc<RefCell<Container>>,
    /// `onLabelChange?: (entryId, label) => void`.
    on_label_change_callback: Rc<RefCell<Option<Box<dyn FnMut(String, Option<String>)>>>>,
    container: Container,
    focused: bool,
    /// `treeList.onLabelEdit = (entryId, currentLabel) => this.showLabelInput(...)`.
    ///
    /// The tree list and the selector need each other, so the port shares the
    /// pending edit request through this slot: the tree list records the request
    /// and `handle_input`/`render` forward it to `show_label_input`.
    pending_label_edit: Rc<RefCell<Option<(String, Option<String>)>>>,
}

impl TreeSelectorComponent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tree: &[AgentConnectionSessionTreeNode],
        current_leaf_id: Option<String>,
        terminal_height: f64,
        on_select: Box<dyn FnMut(&str)>,
        on_cancel: Box<dyn FnMut()>,
        on_label_change: Option<Box<dyn FnMut(String, Option<String>)>>,
        initial_selected_id: Option<String>,
        initial_filter_mode: Option<FilterMode>,
    ) -> Self {
        let max_visible_lines = 5usize.max((terminal_height / 2.0).floor() as usize);

        let mut tree_list = TreeList::new(
            tree,
            current_leaf_id,
            max_visible_lines,
            initial_selected_id,
            initial_filter_mode,
        );
        tree_list.on_select = Some(on_select);
        tree_list.on_cancel = Some(on_cancel);
        let pending_label_edit: Rc<RefCell<Option<(String, Option<String>)>>> =
            Rc::new(RefCell::new(None));
        {
            let pending = Rc::clone(&pending_label_edit);
            tree_list.on_label_edit = Some(Box::new(
                move |entry_id: String, current_label: Option<String>| {
                    *pending.borrow_mut() = Some((entry_id, current_label));
                },
            ));
        }
        let tree_list = Rc::new(RefCell::new(tree_list));

        let tree_container = Rc::new(RefCell::new(Container::new()));
        tree_container
            .borrow_mut()
            .add_child(Rc::clone(&tree_list) as Rc<RefCell<dyn Component>>);

        let label_input_container = Rc::new(RefCell::new(Container::new()));

        let mut container = Container::new();
        container.add_child(Rc::new(RefCell::new(Spacer::new(1))));
        container.add_child(Rc::new(RefCell::new(DynamicBorder::new(border_color()))));
        container.add_child(Rc::new(RefCell::new(Text::new(
            theme().bold("  Session Tree"),
            1,
            0,
            None,
        ))));
        let filter_keys = [
            key_text("app.tree.filter.default", &KeyTextOptions::default()),
            key_text("app.tree.filter.noTools", &KeyTextOptions::default()),
            key_text("app.tree.filter.userOnly", &KeyTextOptions::default()),
            key_text("app.tree.filter.labeledOnly", &KeyTextOptions::default()),
            key_text("app.tree.filter.all", &KeyTextOptions::default()),
        ]
        .join("/");
        let cycle_keys = format!(
            "{}/{}",
            key_text("app.tree.filter.cycleForward", &KeyTextOptions::default()),
            key_text("app.tree.filter.cycleBackward", &KeyTextOptions::default())
        );
        container.add_child(Rc::new(RefCell::new(TruncatedText::new(
            theme().fg(
                "muted",
                &format!(
                    "  ↑/↓: move. ←/→: page. ^←/^→ or Alt+←/Alt+→: fold/branch. {}: label. {filter_keys}: filters ({cycle_keys} cycle). {}: label time",
                    key_text("app.tree.editLabel", &KeyTextOptions::default()),
                    key_text("app.tree.toggleLabelTimestamp", &KeyTextOptions::default())
                ),
            ),
            0,
            0,
        ))));
        container.add_child(Rc::new(RefCell::new(SearchLine::new(Rc::clone(
            &tree_list,
        )))));
        container.add_child(Rc::new(RefCell::new(DynamicBorder::new(border_color()))));
        container.add_child(Rc::new(RefCell::new(Spacer::new(1))));
        container.add_child(Rc::clone(&tree_container) as Rc<RefCell<dyn Component>>);
        container.add_child(Rc::clone(&label_input_container) as Rc<RefCell<dyn Component>>);
        container.add_child(Rc::new(RefCell::new(Spacer::new(1))));
        container.add_child(Rc::new(RefCell::new(DynamicBorder::new(border_color()))));

        Self {
            tree_list,
            label_input: None,
            label_input_container,
            tree_container,
            on_label_change_callback: Rc::new(RefCell::new(on_label_change)),
            container,
            focused: false,
            pending_label_edit,
        }
    }

    /// Port of `showLabelInput(entryId, currentLabel)`.
    pub fn show_label_input(&mut self, entry_id: &str, current_label: Option<String>) {
        let mut label_input = LabelInput::new(entry_id, current_label.as_deref());
        let tree_list = Rc::clone(&self.tree_list);
        let label_change = Rc::clone(&self.on_label_change_callback);
        let pending = Rc::clone(&self.pending_label_edit);
        let submit_pending = Rc::clone(&pending);
        label_input.on_submit = Some(Box::new(move |id: String, label: Option<String>| {
            tree_list
                .borrow_mut()
                .update_node_label(&id, label.clone(), None);
            if let Some(callback) = label_change.borrow_mut().as_mut() {
                callback(id, label);
            }
            // `this.hideLabelInput()`.
            *submit_pending.borrow_mut() = Some((String::new(), None));
        }));
        // `this.labelInput.onCancel = () => this.hideLabelInput()`.
        let cancel_pending = Rc::clone(&pending);
        label_input.on_cancel = Some(Box::new(move || {
            *cancel_pending.borrow_mut() = Some((String::new(), None));
        }));

        // Propagate current focused state to the new labelInput
        let mut label_input = label_input;
        Focusable::set_focused(&mut label_input, self.focused);

        self.label_input = Some(Rc::new(RefCell::new(label_input)));

        // `this.treeContainer.clear(); this.labelInputContainer.clear();
        //  this.labelInputContainer.addChild(this.labelInput);`
        self.tree_container.borrow_mut().clear();
        self.label_input_container.borrow_mut().clear();
        self.label_input_container
            .borrow_mut()
            .add_child(Rc::clone(self.label_input.as_ref().unwrap()) as Rc<RefCell<dyn Component>>);
    }

    /// Port of `hideLabelInput()`.
    pub fn hide_label_input(&mut self) {
        self.label_input = None;
        self.label_input_container.borrow_mut().clear();
        self.tree_container.borrow_mut().clear();
        self.tree_container
            .borrow_mut()
            .add_child(Rc::clone(&self.tree_list) as Rc<RefCell<dyn Component>>);
    }

    /// Port of `handleInput(keyData)`.
    pub fn handle_input(&mut self, key_data: &str) {
        // The tree list records `onLabelEdit` into `pending_label_edit`.
        if let Some(label_input) = self.label_input.as_ref() {
            let mut input = label_input.borrow_mut();
            LabelInput::handle_input(&mut input, key_data);
        } else {
            self.tree_list.borrow_mut().handle_input(key_data);
        }
        let pending_label_edit = self.pending_label_edit.borrow_mut().take();
        if let Some((entry_id, label)) = pending_label_edit {
            if entry_id.is_empty() {
                self.hide_label_input();
            } else {
                self.show_label_input(&entry_id, label);
            }
        }
    }

    pub fn get_tree_list(&self) -> Rc<RefCell<TreeList>> {
        Rc::clone(&self.tree_list)
    }
}

impl Focusable for TreeSelectorComponent {
    fn focused(&self) -> bool {
        self.focused
    }

    fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
        // Propagate to labelInput when it's active
        if let Some(label_input) = self.label_input.as_ref() {
            Focusable::set_focused(&mut *label_input.borrow_mut(), focused);
        }
    }
}

impl Component for TreeSelectorComponent {
    fn render(&mut self, width: f64) -> Vec<String> {
        Component::render(&mut self.container, width)
    }

    fn handle_input(&mut self, data: &str) {
        TreeSelectorComponent::handle_input(self, data);
    }

    fn invalidate(&mut self) {
        Component::invalidate(&mut self.container);
    }

    fn as_focusable(&mut self) -> Option<&mut dyn Focusable> {
        Some(self)
    }
}

/// `new DynamicBorder()`'s default color: `theme.fg("border", str)`.
fn border_color() -> ColorFn {
    Box::new(|text: &str| theme().fg("border", text))
}

/// `s.replace(/[\n\t]/g, " ").trim()`.
fn normalize_text(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        match ch {
            '\n' | '\t' => out.push(' '),
            other => out.push(other),
        }
    }
    out.trim().to_string()
}

/// `extractContent(content)` for an `AgentMessage`.
///
/// The TypeScript reads `(msg as { content?: unknown }).content` for every role
/// and accepts a string or an array of `{ type: "text", text }` blocks; the port
/// reads the serialized message so both shapes hit the same code.
fn extract_content(message: &serde_json::Value) -> String {
    match message.get("content") {
        Some(content) => extract_content_value(content),
        None => String::new(),
    }
}

/// `extractContent(content)` for a raw JSON value (custom messages).
fn extract_content_value(content: &serde_json::Value) -> String {
    const MAX_LEN: usize = 200;
    match content {
        serde_json::Value::String(text) => text.chars().take(MAX_LEN).collect(),
        serde_json::Value::Array(items) => {
            let mut result = String::new();
            for item in items {
                if let Some(text) = item.get("text").and_then(|text| text.as_str()) {
                    if item.get("type").and_then(|kind| kind.as_str()) == Some("text") {
                        result.push_str(text);
                        if result.len() >= MAX_LEN {
                            return result.chars().take(MAX_LEN).collect();
                        }
                    }
                }
            }
            result
        }
        _ => String::new(),
    }
}

/// `hasTextContent(content)`.
fn has_text_content(content: Option<&serde_json::Value>) -> bool {
    match content {
        Some(serde_json::Value::String(text)) => !text.trim().is_empty(),
        Some(serde_json::Value::Array(items)) => items.iter().any(|item| {
            if item.get("type").and_then(|kind| kind.as_str()) != Some("text") {
                return false;
            }
            item.get("text")
                .and_then(|text| text.as_str())
                .map(|text| !text.trim().is_empty())
                .unwrap_or(false)
        }),
        _ => false,
    }
}

/// `isSettingsEntry` - entry types hidden in the default view.
fn is_settings_entry(entry: &AgentConnectionSessionEntry) -> bool {
    matches!(
        entry,
        AgentConnectionSessionEntry::Label { .. }
            | AgentConnectionSessionEntry::Custom { .. }
            | AgentConnectionSessionEntry::ModelChange { .. }
            | AgentConnectionSessionEntry::ThinkingLevelChange { .. }
            | AgentConnectionSessionEntry::ServiceTierChange { .. }
            | AgentConnectionSessionEntry::SessionInfo { .. }
            | AgentConnectionSessionEntry::ChildUsageAttribution { .. }
    )
}

/// `formatToolCall(name, args)`.
fn format_tool_call(name: &str, args: &serde_json::Map<String, serde_json::Value>) -> String {
    let shorten_path = |path: &str| -> String {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_default();
        if !home.is_empty() && path.starts_with(&home) {
            format!("~{}", &path[home.len()..])
        } else {
            path.to_string()
        }
    };

    match name {
        "edit" => {
            let path = json_string(args, "path")
                .or_else(|| json_string(args, "file_path"))
                .unwrap_or_default();
            format!("[edit: {}]", shorten_path(&path))
        }
        "bash" => {
            let raw_cmd = json_string(args, "command").unwrap_or_default();
            let cmd: String = normalize_text(&raw_cmd).chars().take(50).collect();
            format!(
                "[bash: {cmd}{}]",
                if raw_cmd.len() > 50 { "..." } else { "" }
            )
        }
        "ipython" => {
            let raw_code = json_string(args, "code").unwrap_or_default();
            let code: String = normalize_text(&raw_code).chars().take(50).collect();
            format!(
                "[ipython: {code}{}]",
                if raw_code.len() > 50 { "..." } else { "" }
            )
        }
        _ => {
            // Custom tool - show name and truncated JSON args
            let args_value = serde_json::Value::Object(args.clone());
            let args_str = args_value.to_string();
            let truncated: String = args_str.chars().take(40).collect();
            format!(
                "[{name}: {truncated}{}]",
                if args_str.len() > 40 { "..." } else { "" }
            )
        }
    }
}

fn json_string(args: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<String> {
    match args.get(key) {
        Some(serde_json::Value::String(text)) => Some(text.clone()),
        Some(serde_json::Value::Number(number)) => Some(number.to_string()),
        Some(serde_json::Value::Bool(value)) => Some(value.to_string()),
        _ => None,
    }
}

/// `entry.childUsage.input` and friends - numbers read out of a JSON object.
fn json_number(value: &serde_json::Value, key: &str) -> f64 {
    value
        .get(key)
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0)
}

/// `formatLabelTimestamp(timestamp)`.
fn format_label_timestamp(timestamp: &str) -> String {
    let Some(millis) = parse_iso_millis(timestamp) else {
        return timestamp.to_string();
    };
    let (year, month, day, hours, minutes) = local_parts(millis);
    let now_millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(millis);
    let (now_year, now_month, now_day, _, _) = local_parts(now_millis);
    let time = format!("{hours:02}:{minutes:02}");

    if year == now_year && month == now_month && day == now_day {
        return time;
    }
    if year == now_year {
        return format!("{month}/{day} {time}");
    }

    let short_year = year % 100;
    format!("{short_year}/{month}/{day} {time}")
}

/// Local-time components of an epoch-millisecond instant.
///
/// JavaScript `Date` getters are local time; the port reads the process UTC
/// offset once so the same formatting rules apply.
fn local_parts(millis: i64) -> (i64, i64, i64, i64, i64) {
    let offset_seconds = local_utc_offset_seconds();
    let seconds_total = millis.div_euclid(1000) + offset_seconds;
    let days = seconds_total.div_euclid(86400);
    let seconds_of_day = seconds_total.rem_euclid(86400);
    let (year, month, day) = civil_from_days(days);
    (
        year,
        month,
        day,
        seconds_of_day / 3600,
        (seconds_of_day % 3600) / 60,
    )
}

/// The process UTC offset in seconds for the current instant.
fn local_utc_offset_seconds() -> i64 {
    // `chrono::Local` is already a dependency of this crate and exposes the
    // platform offset, matching the JavaScript local-time getters.
    use chrono::{Local, Offset};
    let now = Local::now();
    let offset = now.offset().fix();
    offset.local_minus_utc() as i64
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `new Date(value)` -> epoch milliseconds, or `undefined` when not finite.
fn parse_iso_millis(value: &str) -> Option<i64> {
    let trimmed = value.trim();
    let (date_part, rest) = trimmed.split_once('T')?;
    let mut date_fields = date_part.split('-');
    let year: i64 = date_fields.next()?.parse().ok()?;
    let month: i64 = date_fields.next()?.parse().ok()?;
    let day: i64 = date_fields.next()?.parse().ok()?;

    let (time_part, offset_seconds) = split_timezone(rest)?;
    let mut time_fields = time_part.split(':');
    let hour: i64 = time_fields.next()?.parse().ok()?;
    let minute: i64 = time_fields.next()?.parse().ok()?;
    let seconds: f64 = time_fields.next().unwrap_or("0").parse().ok()?;

    let days = days_from_civil(year, month, day);
    let seconds_total = days * 86400 + hour * 3600 + minute * 60 + seconds.floor() as i64;
    Some((seconds_total - offset_seconds) * 1000)
}

fn split_timezone(rest: &str) -> Option<(&str, i64)> {
    if let Some(time) = rest.strip_suffix('Z') {
        return Some((time, 0));
    }
    for (index, ch) in rest.char_indices() {
        if ch == '+' || ch == '-' {
            let time = &rest[..index];
            let offset = &rest[index..];
            let sign = if ch == '-' { -1 } else { 1 };
            let mut parts = offset[1..].split(':');
            let hours: i64 = parts.next()?.parse().ok()?;
            let minutes: i64 = parts.next().unwrap_or("0").parse().ok()?;
            return Some((time, sign * (hours * 3600 + minutes * 60)));
        }
    }
    Some((rest, 0))
}

fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// `new Date().toISOString()`.
fn now_iso8601() -> String {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0);
    let seconds_total = millis.div_euclid(1000);
    let days = seconds_total.div_euclid(86400);
    let seconds_of_day = seconds_total.rem_euclid(86400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        seconds_of_day / 3600,
        (seconds_of_day % 3600) / 60,
        seconds_of_day % 60,
        millis.rem_euclid(1000)
    )
}

/// `getSearchableText(node)`.
fn get_searchable_text(node: &AgentConnectionSessionTreeNode) -> String {
    let entry = &node.entry;
    let mut parts: Vec<String> = Vec::new();

    if let Some(label) = &node.label {
        parts.push(label.clone());
    }

    match entry {
        AgentConnectionSessionEntry::Message { message, .. } => {
            parts.push(message.role().to_string());
            let message_value = serde_json::to_value(message).unwrap_or(serde_json::Value::Null);
            let content = extract_content(&message_value);
            if !content.is_empty() {
                parts.push(content);
            }
            if message.role() == "bashExecution" {
                if let Some(command) = message_value
                    .get("command")
                    .and_then(|value| value.as_str())
                {
                    parts.push(command.to_string());
                }
            }
        }
        AgentConnectionSessionEntry::CustomMessage {
            custom_type,
            content,
            ..
        } => {
            parts.push(custom_type.clone());
            match content {
                serde_json::Value::String(text) => parts.push(text.clone()),
                other => parts.push(extract_content_value(other)),
            }
        }
        AgentConnectionSessionEntry::Compaction { .. } => parts.push("compaction".to_string()),
        AgentConnectionSessionEntry::BranchSummary { summary, .. } => {
            parts.push("branch summary".to_string());
            parts.push(summary.clone());
        }
        AgentConnectionSessionEntry::SessionInfo { name, .. } => {
            parts.push("title".to_string());
            if let Some(name) = name {
                parts.push(name.clone());
            }
        }
        AgentConnectionSessionEntry::ModelChange { model_id, .. } => {
            parts.push("model".to_string());
            parts.push(model_id.clone());
        }
        AgentConnectionSessionEntry::ThinkingLevelChange { thinking_level, .. } => {
            parts.push("thinking".to_string());
            parts.push(thinking_level.clone());
        }
        AgentConnectionSessionEntry::ServiceTierChange { service_tier, .. } => {
            parts.push("service tier".to_string());
            parts.push(
                service_tier
                    .as_ref()
                    .and_then(|tier| tier.as_deref())
                    .unwrap_or("default")
                    .to_string(),
            );
            if service_tier.as_ref().and_then(|tier| tier.as_deref()) == Some("priority") {
                parts.push("fast on".to_string());
            }
        }
        AgentConnectionSessionEntry::Custom { custom_type, .. } => {
            parts.push("custom".to_string());
            parts.push(custom_type.clone());
        }
        AgentConnectionSessionEntry::ChildUsageAttribution { target_id, .. } => {
            parts.push("child usage".to_string());
            parts.push(target_id.clone());
        }
        AgentConnectionSessionEntry::Label { label, .. } => {
            parts.push("label".to_string());
            parts.push(label.clone().unwrap_or_default());
        }
        _ => {}
    }

    parts.join(" ")
}

#[cfg(test)]
mod linear_storage_tests {
    use super::*;

    fn custom_node(id: &str) -> AgentConnectionSessionTreeNode {
        AgentConnectionSessionTreeNode {
            entry: AgentConnectionSessionEntry::CustomMessage {
                id: id.to_string(),
                parent_id: None,
                timestamp: "2026-09-18T00:00:00Z".into(),
                custom_type: "test".into(),
                content: serde_json::Value::String("payload".into()),
                details: None,
                display: true,
            },
            label: None,
            label_timestamp: None,
            children: Vec::new(),
        }
    }

    #[test]
    fn empty_tree_up_and_down_keep_zero_selection_and_render_no_entries() {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        let mut list = TreeList::new(&[], None, 5, None, None);
        for key in ["\u{1b}[A", "\u{1b}[B"] {
            list.handle_input(key);
            assert_eq!(list.selected_index(), 0);
            assert!(list.render(40.0).iter().any(|line| line.contains("No entries found")));
        }
    }

    #[test]
    fn search_empty_tree_up_and_down_keep_zero_selection_and_render_no_entries() {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        let mut list = TreeList::new(&[custom_node("node")], None, 5, None, None);
        list.handle_input("no-match");
        assert!(list.filtered_nodes.is_empty());
        for key in ["\u{1b}[A", "\u{1b}[B"] {
            list.handle_input(key);
            assert_eq!(list.selected_index(), 0);
            assert!(list.render(40.0).iter().any(|line| line.contains("No entries found")));
        }
    }

    #[test]
    fn label_save_rebuilds_labeled_projection_selection_and_activation() {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        let nodes = vec![custom_node("node-0"), custom_node("node-1")];
        let mut list = TreeList::new(&nodes, None, 10, None, Some("labeled-only"));
        assert!(list.filtered_nodes.is_empty());
        let activated = Rc::new(RefCell::new(Vec::<String>::new()));
        let sink = Rc::clone(&activated);
        list.on_select = Some(Box::new(move |id| sink.borrow_mut().push(id.to_string())));

        list.update_node_label("node-1", Some("fresh label".into()), None);
        assert_eq!(list.filtered_nodes.len(), 1);
        assert_eq!(list.get_selected_node().unwrap().entry.id(), "node-1");
        assert!(list.render(100.0).join("\n").contains("fresh label"));

        list.update_node_label("node-1", Some("renamed now".into()), None);
        let rendered = list.render(100.0).join("\n");
        assert!(rendered.contains("renamed now"));
        assert!(!rendered.contains("fresh label"));

        list.update_node_label("node-0", Some("fallback".into()), None);
        assert_eq!(list.filtered_nodes.len(), 2);
        assert_eq!(list.get_selected_node().unwrap().entry.id(), "node-1");
        list.update_node_label("node-1", None, None);
        assert_eq!(list.filtered_nodes.len(), 1);
        assert_eq!(list.get_selected_node().unwrap().entry.id(), "node-0");
        list.handle_input("\r");
        assert_eq!(activated.borrow().as_slice(), ["node-0"]);

        list.update_node_label("node-0", None, None);
        assert!(list.filtered_nodes.is_empty());
        list.handle_input("\r");
        assert_eq!(
            activated.borrow().as_slice(),
            ["node-0"],
            "an empty projection cannot activate a stale row"
        );
    }

    #[test]
    fn long_chain_flat_rows_remain_shallow_through_label_fold_and_search() {
        let mut children = Vec::new();
        for index in (0..200).rev() {
            children = vec![AgentConnectionSessionTreeNode {
                entry: AgentConnectionSessionEntry::CustomMessage {
                    id: format!("node-{index}"),
                    parent_id: (index > 0).then(|| format!("node-{}", index - 1)),
                    timestamp: "2026-09-18T00:00:00Z".into(),
                    custom_type: "test".into(),
                    content: serde_json::Value::String("payload".repeat(128)),
                    details: None, display: true,
                },
                label: None, label_timestamp: None, children,
            }];
        }
        let mut list = TreeList::new(&children, Some("node-199".into()), 15, Some("node-0".into()), Some("all"));
        assert_eq!(list.flat_nodes.len(), 200);
        assert_eq!(list.filtered_nodes.len(), 200);
        assert!(list.flat_nodes.iter().all(|row| row.node.children.is_empty()));
        assert!(list.filtered_nodes.iter().all(|row| row.node.children.is_empty()));
        assert_eq!(list.get_selected_node().unwrap().entry, children[0].entry);
        assert!(list.get_selected_node().unwrap().children.is_empty());
        list.update_node_label("node-0", Some("root label".into()), Some("2026-09-18T01:00:00Z".into()));
        list.apply_filter();
        assert_eq!(list.get_selected_node().unwrap().label.as_deref(), Some("root label"));
        assert!(list.get_selected_node().unwrap().children.is_empty());
        list.folded_nodes.insert("node-0".into());
        list.apply_filter();
        assert_eq!(list.filtered_nodes.len(), 1);
        list.folded_nodes.clear();
        list.apply_filter();
        assert_eq!(list.filtered_nodes.len(), 200);
        list.search_query = "root label".into();
        list.apply_filter();
        assert_eq!(list.filtered_nodes.len(), 1);
        assert_eq!(list.get_selected_node().unwrap().entry.id(), "node-0");
    }
}
