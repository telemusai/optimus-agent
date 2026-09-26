//! Port of packages/coding-agent/src/modes/interactive/components/model-selector.ts
//!
//! Uses the shared menu components for the same search, rows and responsive
//! layout as the TypeScript model picker.

use pi_ai::types::Model;
use pi_tui::components::{spacer::Spacer, text::Text};
use pi_tui::fuzzy::fuzzy_match;
use pi_tui::keybindings::{get_keybindings, KeyMatchOptions};
use pi_tui::tui::{Component, Focusable};
use serde_json::Value;
use std::cell::RefCell;
use std::rc::Rc;

use crate::core::model_registry::ModelRegistry;
use crate::modes::interactive::theme::theme::theme;

use super::keybinding_hints::{key_hint, KeyTextOptions};
use super::menu_panel::{
    get_menu_list_layout, MenuList, MenuListLayout, MenuListLayoutOptions, MenuPanel,
    MenuPanelOptions, MenuRow, MenuRowOptions, MenuSearchInput, MenuViewportProvider,
};
use super::modal_back::{should_treat_as_back, BackGuardInput};

/// Adapter that exposes the model filter's cursor column to `modalBack`'s
/// `BackGuardInput` guard (components/modal-back.ts).
struct SearchInputCursor<'a>(&'a MenuSearchInput);

impl BackGuardInput for SearchInputCursor<'_> {
    fn get_cursor(&self) -> usize {
        self.0.get_cursor()
    }
}

// ---------------------------------------------------------------------------
// ModelSelector
// ---------------------------------------------------------------------------

/// `ModelItem`
#[derive(Debug, Clone)]
pub struct ModelItemModel {
    pub provider: String,
    pub id: String,
    pub name: String,
    pub featured: bool,
    pub raw: Value,
}

/// `ModelItem`
#[derive(Debug, Clone)]
pub struct ModelItem {
    pub provider: String,
    pub id: String,
    pub model: ModelItemModel,
}

/// `ScopedModelItem`
#[derive(Debug, Clone)]
pub struct ScopedModelItem {
    pub model: ModelItemModel,
    pub thinking_level: Option<String>,
}

/// `enum ModelSearchMatchQuality`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ModelSearchMatchQuality {
    ExactShortId,
    ExactFullId,
    PrefixOrToken,
    Fuzzy,
}

/// `interface ModelSearchMatch`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelSearchMatch {
    pub quality: ModelSearchMatchQuality,
    pub score: f64,
}

fn normalize_model_search_text(value: &str) -> String {
    let mut result = String::new();
    for ch in value.to_lowercase().chars() {
        if ch.is_whitespace() || matches!(ch, '-' | '_' | '.' | ':' | '/') {
            continue;
        }
        result.push(ch);
    }
    result
}

/// `getModelSearchFields`
#[derive(Debug, Clone)]
pub struct ModelSearchFields {
    pub short_id: String,
    pub full_ids: Vec<String>,
    pub all: Vec<String>,
}

pub fn get_model_search_fields(item: &ModelItem) -> ModelSearchFields {
    let short_id = match item.id.rfind('/') {
        Some(index) => item.id[index + 1..].to_string(),
        None => item.id.clone(),
    };
    let full_ids = vec![item.id.clone(), format!("{}/{}", item.provider, item.id)];
    let mut all = vec![short_id.clone()];
    all.extend(full_ids.iter().cloned());
    all.push(item.model.name.clone());
    all.push(item.provider.clone());
    ModelSearchFields {
        short_id,
        full_ids,
        all,
    }
}

fn get_best_fuzzy_score(query_tokens: &[String], fields: &[String]) -> Option<f64> {
    let mut total = 0.0;
    for token in query_tokens {
        let mut best = f64::INFINITY;
        for field in fields {
            let matched = fuzzy_match(token, field);
            if matched.matches {
                best = best.min(matched.score as f64);
            }
        }
        if !best.is_finite() {
            return None;
        }
        total += best;
    }
    Some(total)
}

/// Port of `scoreModelSearch`.
pub fn score_model_search(item: &ModelItem, query: &str) -> Option<ModelSearchMatch> {
    let query_tokens: Vec<String> = query.split_whitespace().map(str::to_string).collect();
    let normalized_query = normalize_model_search_text(query);
    let normalized_tokens: Vec<String> = query_tokens
        .iter()
        .map(|token| normalize_model_search_text(token))
        .filter(|token| !token.is_empty())
        .collect();
    if normalized_query.is_empty() || normalized_tokens.is_empty() {
        return None;
    }

    let fields = get_model_search_fields(item);
    if normalize_model_search_text(&fields.short_id) == normalized_query {
        return Some(ModelSearchMatch {
            quality: ModelSearchMatchQuality::ExactShortId,
            score: 0.0,
        });
    }
    if fields
        .full_ids
        .iter()
        .any(|field| normalize_model_search_text(field) == normalized_query)
    {
        return Some(ModelSearchMatch {
            quality: ModelSearchMatchQuality::ExactFullId,
            score: 0.0,
        });
    }

    let normalized_fields: Vec<String> = fields
        .all
        .iter()
        .map(|field| normalize_model_search_text(field))
        .collect();
    let field_tokens: Vec<String> = fields
        .all
        .iter()
        .flat_map(|field| {
            field
                .split(|ch: char| ch.is_whitespace() || ch == '/' || ch == '_' || ch == '-')
                .map(str::to_string)
                .collect::<Vec<String>>()
        })
        .map(|token| normalize_model_search_text(&token))
        .filter(|token| !token.is_empty())
        .collect();
    let fuzzy_score = get_best_fuzzy_score(&normalized_tokens, &normalized_fields);
    let is_prefix_or_token = normalized_tokens.iter().all(|token| {
        normalized_fields
            .iter()
            .any(|field| field.starts_with(token))
            || field_tokens.iter().any(|field| field.starts_with(token))
    });
    if is_prefix_or_token && fuzzy_score.is_some() {
        return Some(ModelSearchMatch {
            quality: ModelSearchMatchQuality::PrefixOrToken,
            score: fuzzy_score.unwrap_or(0.0),
        });
    }
    fuzzy_score.map(|score| ModelSearchMatch {
        quality: ModelSearchMatchQuality::Fuzzy,
        score,
    })
}

/// `ModelSelectorOptions`
#[derive(Default)]
pub struct ModelSelectorOptions {
    pub available_models: Option<Vec<ModelItemModel>>,
    pub configured_providers: Option<Vec<String>>,
    pub header_rows: Option<f64>,
    pub header: Option<Rc<RefCell<dyn Component>>>,
    pub get_header_rows: Option<Rc<dyn Fn() -> f64>>,
    pub subtitle: Option<String>,
    pub get_rows: Option<Rc<dyn Fn() -> f64>>,
    pub recent_models: Option<Vec<String>>,
    pub initial_search_input: Option<String>,
}

/// `type ModelScope = "all" | "scoped"`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelScope {
    All,
    Scoped,
}

/// `PREFERRED_VISIBLE_MODELS`
pub const PREFERRED_VISIBLE_MODELS: usize = 10;
/// `MODEL_LIST_RESERVED_ROWS`
pub const MODEL_LIST_RESERVED_ROWS_BASE: usize = 7;
/// `MODEL_LIST_RESERVED_ROWS.detail`
pub const MODEL_LIST_RESERVED_ROWS_DETAIL: usize = 2;
/// `MODEL_SCROLL_INDICATOR_ROWS`
pub const MODEL_SCROLL_INDICATOR_ROWS: usize = 1;
/// `MODEL_HELP_MIN_ROWS`
pub const MODEL_HELP_MIN_ROWS: f64 = 12.0;
/// `MODEL_DETAIL_MIN_ROWS`
pub const MODEL_DETAIL_MIN_ROWS: f64 = 14.0;

/// Port of `modelsAreEqual` (pi-ai `types.ts`): same provider and same id.
pub fn models_are_equal(a: Option<&ModelItemModel>, b: Option<&ModelItemModel>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.provider == b.provider && a.id == b.id,
        (None, None) => true,
        _ => false,
    }
}

/// The registry surface `ModelSelectorComponent` uses.
pub trait ModelRegistryLike {
    fn refresh(&mut self);
    fn get_error(&self) -> Option<String>;
    fn get_available(&self) -> Vec<ModelItemModel>;
    fn find(&self, provider: &str, model_id: &str) -> Option<ModelItemModel>;
    fn has_configured_auth(&self, model: &ModelItemModel) -> bool;
}

impl ModelRegistryLike for ModelRegistry {
    fn refresh(&mut self) {
        ModelRegistry::refresh(self);
    }

    fn get_error(&self) -> Option<String> {
        ModelRegistry::get_error(self).map(str::to_string)
    }

    fn get_available(&self) -> Vec<ModelItemModel> {
        ModelRegistry::get_available(self)
            .into_iter()
            .map(model_item_model_from_ai)
            .collect()
    }

    fn find(&self, provider: &str, model_id: &str) -> Option<ModelItemModel> {
        ModelRegistry::find(self, provider, model_id).map(model_item_model_from_ai)
    }

    fn has_configured_auth(&self, model: &ModelItemModel) -> bool {
        let Ok(ai_model) = serde_json::from_value::<Model>(model.raw.clone()) else {
            return false;
        };
        ModelRegistry::has_configured_auth(self, &ai_model)
    }
}

fn model_item_model_from_ai(model: Model) -> ModelItemModel {
    let raw = serde_json::to_value(&model).unwrap_or(Value::Null);
    ModelItemModel {
        provider: model.provider.clone(),
        id: model.id.clone(),
        name: model.name.clone(),
        featured: model.featured.unwrap_or(false),
        raw,
    }
}

/// State of `ModelSelectorComponent`. The class methods are ported below; the
/// chrome (panel/list/rows/search input) lives behind these fields.
pub struct ModelSelectorComponent {
    pub search_query: String,
    pub search_value: String,
    pub cursor: usize,
    pub all_models: Vec<ModelItem>,
    pub scoped_model_items: Vec<ModelItem>,
    pub active_models: Vec<ModelItem>,
    pub filtered_models: Vec<ModelItem>,
    pub selected_index: usize,
    pub current_model: Option<ModelItemModel>,
    pub available_models: Option<Vec<ModelItemModel>>,
    pub configured_providers: Option<Vec<String>>,
    pub recent_rank: std::collections::HashMap<String, usize>,
    pub error_message: Option<String>,
    pub scoped_models: Vec<ScopedModelItem>,
    pub scope: ModelScope,
    pub list_layout: MenuListLayout,
    pub responsive_layout_key: String,
    pub viewport: MenuViewportProvider,
    pub header_rows: Option<f64>,
    pub header: Option<Rc<RefCell<dyn Component>>>,
    pub get_header_rows: Option<Rc<dyn Fn() -> f64>>,
    pub has_header: bool,
    pub subtitle: Option<String>,
    pub rows_requested: bool,
    /// `MenuSearchInput` state (its `Input` component and the last visible slice).
    pub search_input: Rc<RefCell<MenuSearchInput>>,
    pub visible_range: (usize, usize),
    /// Set when the user cancelled (`onCancelCallback`).
    pub cancelled: bool,
    /// Set when the user confirmed a model (`onSelectCallback`).
    pub selected_model: Option<ModelItemModel>,
}

impl ModelSelectorComponent {
    /// `constructor(...)`
    pub fn new(
        current_model: Option<ModelItemModel>,
        scoped_models: Vec<ScopedModelItem>,
        options: ModelSelectorOptions,
    ) -> Self {
        let scope = if scoped_models.is_empty() {
            ModelScope::All
        } else {
            ModelScope::Scoped
        };
        let recent_rank = options
            .recent_models
            .clone()
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(index, key)| (key, index))
            .collect();
        let initial_search = options.initial_search_input.clone().unwrap_or_default();
        let header_rows = options.header_rows;
        let has_header = header_rows.is_some();
        let mut component = Self {
            search_query: String::new(),
            search_value: String::new(),
            cursor: 0,
            all_models: Vec::new(),
            scoped_model_items: Vec::new(),
            active_models: Vec::new(),
            filtered_models: Vec::new(),
            selected_index: 0,
            current_model,
            available_models: options.available_models,
            configured_providers: options.configured_providers,
            recent_rank,
            error_message: None,
            scoped_models,
            scope,
            list_layout: MenuListLayout {
                compact: false,
                visible_items: 0,
            },
            responsive_layout_key: String::new(),
            viewport: MenuViewportProvider {
                get_rows: options.get_rows.clone(),
            },
            header_rows: if has_header {
                Some(header_rows.unwrap_or(2.0))
            } else {
                None
            },
            has_header,
            header: options.header,
            get_header_rows: options.get_header_rows,
            subtitle: options.subtitle,
            rows_requested: false,
            search_input: Rc::new(RefCell::new(MenuSearchInput::new("Search models".into()))),
            visible_range: (0, 0),
            cancelled: false,
            selected_model: None,
        };
        component.list_layout = get_menu_list_layout(MenuListLayoutOptions {
            preferred_visible_items: PREFERRED_VISIBLE_MODELS,
            reserved_rows: MODEL_LIST_RESERVED_ROWS_BASE,
            comfortable_item_rows: 3,
            compact_item_rows: Some(2),
            ..Default::default()
        });
        component.load_models();
        component.set_search_value(initial_search);
        component
    }

    /// Port of `updateAvailableModels`.
    pub fn update_available_models(&mut self, available_models: Vec<ModelItemModel>) {
        self.update_state(
            self.current_model.clone(),
            Some(available_models),
            self.configured_providers.clone(),
        );
    }

    /// Port of `updateState`.
    pub fn update_state(
        &mut self,
        current_model: Option<ModelItemModel>,
        available_models: Option<Vec<ModelItemModel>>,
        configured_providers: Option<Vec<String>>,
    ) {
        self.current_model = current_model;
        self.available_models = available_models;
        self.configured_providers = configured_providers;
        let query = self.search_input.borrow().get_value();
        let selected_key = self.get_selected_model_key();

        self.load_models();
        self.filter_models(&query);

        if let Some(selected_key) = selected_key {
            if let Some(index) = self
                .filtered_models
                .iter()
                .position(|item| self.get_model_key(item) == selected_key)
            {
                self.selected_index = index;
                self.update_list();
            }
        }

        self.rows_requested = true;
    }

    fn load_models(&mut self) {
        self.error_message = None;

        // Load available models (built-in models still work even if models.json failed)
        let available_models: Vec<ModelItemModel> = match &self.available_models {
            Some(models) => models.clone(),
            None => Vec::new(),
        };
        let models: Vec<ModelItem> = available_models
            .iter()
            .map(|model| ModelItem {
                provider: model.provider.clone(),
                id: model.id.clone(),
                model: model.clone(),
            })
            .collect();

        self.all_models = self.sort_models(models);
        let available_by_id: std::collections::HashMap<String, ModelItemModel> = available_models
            .iter()
            .map(|model| (format!("{}/{}", model.provider, model.id), model.clone()))
            .collect();
        self.scoped_models = std::mem::take(&mut self.scoped_models)
            .into_iter()
            .map(|scoped| {
                let scoped_model_id = format!("{}/{}", scoped.model.provider, scoped.model.id);
                match available_by_id.get(&scoped_model_id) {
                    Some(refreshed) => ScopedModelItem {
                        model: refreshed.clone(),
                        thinking_level: scoped.thinking_level,
                    },
                    None => scoped,
                }
            })
            .collect();
        self.scoped_model_items = self
            .scoped_models
            .iter()
            .map(|scoped| ModelItem {
                provider: scoped.model.provider.clone(),
                id: scoped.model.id.clone(),
                model: scoped.model.clone(),
            })
            .collect();
        self.active_models = if self.scope == ModelScope::Scoped {
            self.scoped_model_items.clone()
        } else {
            self.all_models.clone()
        };
        self.filtered_models = self.active_models.clone();
        let current_index = self
            .filtered_models
            .iter()
            .position(|item| models_are_equal(self.current_model.as_ref(), Some(&item.model)));
        self.selected_index = match current_index {
            Some(index) => index,
            None => self
                .selected_index
                .min(self.get_selectable_count().saturating_sub(1)),
        };
    }

    /// Port of `loadModels` when `ModelRegistry` supplies the catalog.
    pub fn load_models_from_registry(&mut self, registry: &mut dyn ModelRegistryLike) {
        self.error_message = None;
        if self.available_models.is_none() {
            registry.refresh();
            if let Some(error) = registry.get_error() {
                self.error_message = Some(error);
            }
        }
        let available_models = match &self.available_models {
            Some(models) => models.clone(),
            None => registry.get_available(),
        };
        self.available_models = Some(available_models);
        self.load_models();
    }

    fn get_model_key(&self, item: &ModelItem) -> String {
        format!("{}/{}", item.provider, item.id)
    }

    fn get_selected_model_key(&self) -> Option<String> {
        self.filtered_models
            .get(self.selected_index)
            .map(|item| self.get_model_key(item))
    }

    fn recent_rank_of(&self, item: &ModelItem) -> usize {
        // Unseen models sort after every recorded recent model.
        self.recent_rank
            .get(&format!("{}/{}", item.provider, item.id))
            .copied()
            .unwrap_or(usize::MAX)
    }

    fn is_provider_configured(&self, item: &ModelItem) -> bool {
        self.configured_providers
            .as_ref()
            .map(|providers| providers.iter().any(|provider| provider == &item.provider))
            .unwrap_or(false)
    }

    fn sort_models(&self, models: Vec<ModelItem>) -> Vec<ModelItem> {
        let mut sorted = models;
        sorted.sort_by(|a, b| {
            let configured_diff =
                (self.is_provider_configured(b) as i32) - (self.is_provider_configured(a) as i32);
            if configured_diff != 0 {
                return configured_diff.cmp(&0);
            }
            let a_is_current = models_are_equal(self.current_model.as_ref(), Some(&a.model));
            let b_is_current = models_are_equal(self.current_model.as_ref(), Some(&b.model));
            if a_is_current != b_is_current {
                return if a_is_current {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                };
            }
            let rank_order = self.recent_rank_of(a).cmp(&self.recent_rank_of(b));
            if rank_order != std::cmp::Ordering::Equal {
                return rank_order;
            }
            let provider_diff = a.provider.cmp(&b.provider);
            if provider_diff != std::cmp::Ordering::Equal {
                return provider_diff;
            }
            if a.model.featured != b.model.featured {
                return if a.model.featured {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                };
            }
            numeric_locale_compare(&a.id, &b.id)
        });
        sorted
    }

    fn get_scope_text(&self) -> String {
        let all_text = if self.scope == ModelScope::All {
            theme().fg("accent", "all")
        } else {
            theme().fg("muted", "all")
        };
        let scoped_text = if self.scope == ModelScope::Scoped {
            theme().fg("accent", "scoped")
        } else {
            theme().fg("muted", "scoped")
        };
        format!(
            "{}{}{}{}",
            theme().fg("muted", "Scope: "),
            all_text,
            theme().fg("muted", " | "),
            scoped_text
        )
    }

    fn get_scope_hint_text(&self) -> String {
        format!(
            "{}{}",
            key_hint("app.model.toggleScope", "scope", &KeyTextOptions::default()),
            theme().fg("muted", " (all/scoped)")
        )
    }

    /// Port of `setScope`.
    pub fn set_scope(&mut self, scope: ModelScope) {
        if self.scope == scope {
            return;
        }
        self.scope = scope;
        self.active_models = if self.scope == ModelScope::Scoped {
            self.scoped_model_items.clone()
        } else {
            self.all_models.clone()
        };
        let current_index = self
            .active_models
            .iter()
            .position(|item| models_are_equal(self.current_model.as_ref(), Some(&item.model)));
        self.selected_index = current_index.unwrap_or(0);
        let query = self.search_input.borrow().get_value();
        self.filter_models(&query);
    }

    /// Port of `filterModels`.
    pub fn filter_models(&mut self, query: &str) {
        let query_changed = query != self.search_query;
        self.search_query = query.to_string();
        if !query.trim().is_empty() {
            let mut matches: Vec<(ModelItem, ModelSearchMatch)> = self
                .active_models
                .iter()
                .filter_map(|item| {
                    score_model_search(item, query).map(|matched| (item.clone(), matched))
                })
                .collect();
            matches.sort_by(|(a_item, a), (b_item, b)| {
                (a.quality as i32)
                    .cmp(&(b.quality as i32))
                    .then_with(|| {
                        a.score
                            .partial_cmp(&b.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| {
                        (self.is_provider_configured(b_item) as i32)
                            .cmp(&(self.is_provider_configured(a_item) as i32))
                    })
                    .then_with(|| {
                        let a_current =
                            models_are_equal(self.current_model.as_ref(), Some(&a_item.model))
                                as i32;
                        let b_current =
                            models_are_equal(self.current_model.as_ref(), Some(&b_item.model))
                                as i32;
                        b_current.cmp(&a_current)
                    })
                    .then_with(|| {
                        self.recent_rank_of(a_item)
                            .cmp(&self.recent_rank_of(b_item))
                    })
                    .then_with(|| {
                        numeric_locale_compare(
                            &self.get_model_key(a_item),
                            &self.get_model_key(b_item),
                        )
                    })
            });
            self.filtered_models = matches.into_iter().map(|(item, _)| item).collect();
        } else {
            self.filtered_models = self.active_models.clone();
        }
        self.selected_index = if query_changed {
            0
        } else {
            self.selected_index
                .min(self.get_selectable_count().saturating_sub(1))
        };
        self.update_list();
    }

    /// Port of `updateList`.
    pub fn update_list(&mut self) {
        self.update_responsive_layout();
        let max_visible = self.list_layout.visible_items;
        let selected_model_index = self
            .selected_index
            .min(self.filtered_models.len().saturating_sub(1));
        let start_index = (selected_model_index as i64 - (max_visible / 2) as i64)
            .min(self.filtered_models.len() as i64 - max_visible as i64)
            .max(0) as usize;
        let end_index = (start_index + max_visible).min(self.filtered_models.len());
        self.visible_range = (start_index, end_index);
    }

    /// Port of `handleInput`.
    pub fn handle_input(&mut self, key_data: &str) {
        let kb = get_keybindings();
        if kb.matches_with_options(key_data, "app.model.toggleScope", KeyMatchOptions { option_composed: true }) {
            if !self.scoped_model_items.is_empty() {
                let next_scope = if self.scope == ModelScope::All {
                    ModelScope::Scoped
                } else {
                    ModelScope::All
                };
                self.set_scope(next_scope);
            }
            return;
        }
        // Up arrow - wrap to bottom when at top
        if kb.matches(key_data, "tui.select.up") {
            let selectable_count = self.get_selectable_count();
            if selectable_count == 0 {
                return;
            }
            self.selected_index = if self.selected_index == 0 {
                selectable_count - 1
            } else {
                self.selected_index - 1
            };
            self.update_list();
        }
        // Down arrow - wrap to top when at bottom
        else if kb.matches(key_data, "tui.select.down") {
            let selectable_count = self.get_selectable_count();
            if selectable_count == 0 {
                return;
            }
            self.selected_index = if self.selected_index == selectable_count - 1 {
                0
            } else {
                self.selected_index + 1
            };
            self.update_list();
        }
        // Enter
        else if kb.matches(key_data, "tui.select.confirm") {
            self.handle_confirm();
        }
        // Escape / Ctrl+C, or left arrow when the search field is at its start
        else if kb.matches(key_data, "tui.select.cancel")
            || should_treat_as_back(
                key_data,
                Some(&SearchInputCursor(&self.search_input.borrow())),
            )
        {
            self.cancelled = true;
        }
        // Pass everything else to search input
        else {
            self.search_input.borrow_mut().handle_input(key_data);
            let value = self.search_input.borrow().get_value();
            self.filter_models(&value);
        }
    }

    fn handle_select(&mut self, model: &ModelItemModel) {
        self.selected_model = Some(model.clone());
    }

    fn handle_confirm(&mut self) {
        if let Some(selected) = self.filtered_models.get(self.selected_index).cloned() {
            self.handle_select(&selected.model);
        }
    }

    fn get_selectable_count(&self) -> usize {
        self.filtered_models.len()
    }

    /// Port of `updateResponsiveLayout`.
    fn update_responsive_layout(&mut self) {
        let show_header_help = self.should_show_header_help();
        let mut header_help_rows = 0.0;
        if show_header_help {
            if !self.scoped_model_items.is_empty() {
                header_help_rows += 2.0;
            } else {
                header_help_rows += 1.0;
            }
            header_help_rows += 1.0;
        }

        let header_rows = if let Some(rows) = &self.get_header_rows {
            rows()
        } else if self.header_rows.is_some() {
            self.header_rows.unwrap_or(2.0)
        } else {
            0.0
        };
        let reserved_rows = MODEL_LIST_RESERVED_ROWS_BASE as f64
            + header_rows
            + header_help_rows
            + if self.should_show_selected_details() {
                MODEL_LIST_RESERVED_ROWS_DETAIL as f64
            } else {
                0.0
            };
        self.list_layout = get_menu_list_layout(MenuListLayoutOptions {
            get_rows: self.viewport.get_rows.clone(),
            preferred_visible_items: PREFERRED_VISIBLE_MODELS,
            total_items: Some(self.filtered_models.len()),
            reserved_rows: reserved_rows as usize,
            comfortable_item_rows: 3,
            compact_item_rows: Some(2),
            scroll_indicator_rows: Some(MODEL_SCROLL_INDICATOR_ROWS),
            ..Default::default()
        });
        self.responsive_layout_key = [
            format!("{header_rows}"),
            if show_header_help {
                "help".to_string()
            } else {
                "no-help".to_string()
            },
            format!("{header_help_rows}"),
            if self.should_show_selected_details() {
                "detail".to_string()
            } else {
                "no-detail".to_string()
            },
            if self.list_layout.compact {
                "compact".to_string()
            } else {
                "comfortable".to_string()
            },
            format!("{}", self.list_layout.visible_items),
        ]
        .join(":");
    }

    fn should_show_header_help(&self) -> bool {
        self.has_rows(MODEL_HELP_MIN_ROWS)
    }

    fn should_show_selected_details(&self) -> bool {
        self.has_rows(MODEL_DETAIL_MIN_ROWS)
    }

    fn has_rows(&self, min_rows: f64) -> bool {
        let rows = self.viewport.get_rows.as_ref().map(|get_rows| get_rows());
        match rows {
            None => true,
            Some(rows) => !rows.is_finite() || rows >= min_rows,
        }
    }

    /// Port of `getSearchInput()`.
    pub fn search_input(&self) -> Rc<RefCell<MenuSearchInput>> {
        self.search_input.clone()
    }

    pub fn get_search_value(&self) -> String {
        self.search_input.borrow().get_value()
    }

    pub fn set_search_value(&mut self, value: impl Into<String>) {
        let value = value.into();
        self.search_input.borrow_mut().set_value(&value);
        self.filter_models(&value);
    }

    pub fn get_cursor(&self) -> usize {
        self.search_input.borrow().get_cursor()
    }
}

impl Component for ModelSelectorComponent {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.update_list();
        let mut panel = MenuPanel::new(MenuPanelOptions {
            title: "Models".into(),
            subtitle: Some(
                self.subtitle
                    .clone()
                    .unwrap_or_else(|| "All models across supported providers.".into()),
            ),
        });
        if let Some(header) = &self.header {
            panel.add_child(header.clone());
            panel.add_child(Rc::new(RefCell::new(Spacer::new(1))));
        }
        if self.should_show_header_help() {
            let help = if self.scoped_model_items.is_empty() {
                theme().fg(
                    "muted",
                    "Signed-in providers first. Other models prompt sign-in.",
                )
            } else {
                format!("{}\n{}", self.get_scope_text(), self.get_scope_hint_text())
            };
            panel.add_child(Rc::new(RefCell::new(Text::new(help, 0, 0, None))));
            panel.add_child(Rc::new(RefCell::new(Spacer::new(1))));
        }
        panel.add_full_width_child(self.search_input.clone());
        panel.add_child(Rc::new(RefCell::new(Spacer::new(1))));
        let compact = self.list_layout.compact;
        let mut list = MenuList::new(Some(Box::new(move || compact)));
        let (start, end) = self.visible_range;
        for (index, item) in self
            .filtered_models
            .iter()
            .enumerate()
            .take(end)
            .skip(start)
        {
            let current = models_are_equal(self.current_model.as_ref(), Some(&item.model));
            let meta = if self.is_provider_configured(item) {
                current.then(|| theme().fg("success", "current"))
            } else {
                Some(theme().fg(
                    "warning",
                    if current {
                        "current · sign in"
                    } else {
                        "sign in"
                    },
                ))
            };
            list.add_row(Rc::new(RefCell::new(MenuRow::new(MenuRowOptions {
                primary: item.id.clone(),
                secondary: Some(item.provider.clone()),
                meta,
                selected: index == self.selected_index,
            }))));
        }
        if start > 0 || end < self.filtered_models.len() {
            list.add_child(
                Rc::new(RefCell::new(Text::new(
                    theme().fg(
                        "muted",
                        &format!(
                            "  ({}/{})",
                            self.selected_index + 1,
                            self.filtered_models.len()
                        ),
                    ),
                    0,
                    0,
                    None,
                ))),
                false,
            );
        }
        if let Some(error) = &self.error_message {
            for line in error.lines() {
                list.add_child(
                    Rc::new(RefCell::new(Text::new(
                        theme().fg("error", line),
                        0,
                        0,
                        None,
                    ))),
                    false,
                );
            }
        } else if self.filtered_models.is_empty() {
            list.add_child(
                Rc::new(RefCell::new(Text::new(
                    theme().fg("muted", "No matching models"),
                    0,
                    0,
                    None,
                ))),
                false,
            );
        } else if self.should_show_selected_details() {
            if let Some(selected) = self.filtered_models.get(self.selected_index) {
                list.add_child(Rc::new(RefCell::new(Spacer::new(1))), false);
                list.add_child(
                    Rc::new(RefCell::new(Text::new(
                        theme().fg("muted", &selected.model.name),
                        0,
                        0,
                        None,
                    ))),
                    false,
                );
            }
        }
        panel.add_full_width_child(Rc::new(RefCell::new(list)));
        panel.render(width)
    }

    fn handle_input(&mut self, data: &str) {
        ModelSelectorComponent::handle_input(self, data);
    }

    fn invalidate(&mut self) {
        self.search_input.borrow_mut().invalidate();
    }

    fn as_focusable(&mut self) -> Option<&mut dyn Focusable> {
        Some(self)
    }
}

impl Focusable for ModelSelectorComponent {
    fn focused(&self) -> bool {
        self.search_input.borrow().focused()
    }
    fn set_focused(&mut self, focused: bool) {
        self.search_input.borrow_mut().set_focused(focused);
    }
}

/// `localeCompare(a, b, { numeric: true })` for model ids.
pub fn numeric_locale_compare(a: &str, b: &str) -> std::cmp::Ordering {
    let mut a_chars = a.chars().peekable();
    let mut b_chars = b.chars().peekable();
    loop {
        match (a_chars.peek().copied(), b_chars.peek().copied()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some(a_ch), Some(b_ch)) => {
                if a_ch.is_ascii_digit() && b_ch.is_ascii_digit() {
                    let mut a_num = String::new();
                    while let Some(ch) = a_chars.peek().copied() {
                        if !ch.is_ascii_digit() {
                            break;
                        }
                        a_num.push(ch);
                        a_chars.next();
                    }
                    let mut b_num = String::new();
                    while let Some(ch) = b_chars.peek().copied() {
                        if !ch.is_ascii_digit() {
                            break;
                        }
                        b_num.push(ch);
                        b_chars.next();
                    }
                    let a_value: u128 = a_num.parse().unwrap_or(0);
                    let b_value: u128 = b_num.parse().unwrap_or(0);
                    if a_value != b_value {
                        return a_value.cmp(&b_value);
                    }
                } else {
                    if a_ch != b_ch {
                        return a_ch.cmp(&b_ch);
                    }
                    a_chars.next();
                    b_chars.next();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(provider: &str, id: &str, name: &str) -> ModelItem {
        ModelItem {
            provider: provider.to_string(),
            id: id.to_string(),
            model: ModelItemModel {
                provider: provider.to_string(),
                id: id.to_string(),
                name: name.to_string(),
                featured: false,
                raw: Value::Null,
            },
        }
    }

    #[test]
    fn search_fields_split_the_short_id() {
        let fields =
            get_model_search_fields(&item("anthropic", "claude/sonnet-4", "Claude Sonnet 4"));
        assert_eq!(fields.short_id, "sonnet-4");
        assert_eq!(
            fields.full_ids,
            vec!["claude/sonnet-4", "anthropic/claude/sonnet-4"]
        );
    }

    #[test]
    fn exact_short_id_and_full_id_rank_above_fuzzy() {
        let target = item("anthropic", "claude/sonnet-4", "Claude Sonnet 4");
        assert_eq!(
            score_model_search(&target, "sonnet4").unwrap().quality,
            ModelSearchMatchQuality::ExactShortId
        );
        assert_eq!(
            score_model_search(&target, "anthropic/claude/sonnet-4")
                .unwrap()
                .quality,
            ModelSearchMatchQuality::ExactFullId
        );
        assert_eq!(
            score_model_search(&target, "son").unwrap().quality,
            ModelSearchMatchQuality::PrefixOrToken
        );
    }

    #[test]
    fn empty_query_has_no_match() {
        assert!(score_model_search(&item("a", "b", "c"), "   ").is_none());
    }

    #[test]
    fn search_match_quality_ordering_matches_the_enum_order() {
        assert!(ModelSearchMatchQuality::ExactShortId < ModelSearchMatchQuality::Fuzzy);
        assert!(ModelSearchMatchQuality::PrefixOrToken < ModelSearchMatchQuality::Fuzzy);
    }

    #[test]
    fn numeric_compare_orders_version_numbers() {
        assert_eq!(
            numeric_locale_compare("model2", "model10"),
            std::cmp::Ordering::Less
        );
        assert_eq!(numeric_locale_compare("a", "a"), std::cmp::Ordering::Equal);
        assert_eq!(
            numeric_locale_compare("b", "a"),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn layout_prefers_compact_only_when_it_shows_more_or_the_comfortable_one_does_not_fit() {
        let layout = get_menu_list_layout(MenuListLayoutOptions {
            get_rows: Some(Rc::new(|| 20.0)),
            preferred_visible_items: 10,
            reserved_rows: 7,
            comfortable_item_rows: 3,
            compact_item_rows: Some(2),
            total_items: Some(30),
            scroll_indicator_rows: Some(1),
            ..Default::default()
        });
        assert!(layout.visible_items <= 10);
    }

    #[test]
    fn selector_starts_in_scoped_scope_when_scoped_models_exist() {
        let scoped = ScopedModelItem {
            model: ModelItemModel {
                provider: "p".to_string(),
                id: "m".to_string(),
                name: "M".to_string(),
                featured: false,
                raw: Value::Null,
            },
            thinking_level: None,
        };
        let selector =
            ModelSelectorComponent::new(None, vec![scoped], ModelSelectorOptions::default());
        assert_eq!(selector.scope, ModelScope::Scoped);
    }

    #[test]
    fn selector_defaults_to_all_scope_without_scoped_models() {
        let selector =
            ModelSelectorComponent::new(None, Vec::new(), ModelSelectorOptions::default());
        assert_eq!(selector.scope, ModelScope::All);
    }

    #[test]
    fn arrow_keys_wrap_the_selection() {
        let mut selector =
            ModelSelectorComponent::new(None, Vec::new(), ModelSelectorOptions::default());
        selector.available_models = Some(vec![
            ModelItemModel {
                provider: "a".to_string(),
                id: "1".to_string(),
                name: "1".to_string(),
                featured: false,
                raw: Value::Null,
            },
            ModelItemModel {
                provider: "a".to_string(),
                id: "2".to_string(),
                name: "2".to_string(),
                featured: false,
                raw: Value::Null,
            },
        ]);
        selector.load_models();
        selector.selected_index = 0;
        selector.handle_input("\u{1b}[A");
        assert_eq!(selector.selected_index, 1);
    }

    fn picker(models: Vec<ModelItemModel>) -> ModelSelectorComponent {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
        ModelSelectorComponent::new(
            None,
            vec![],
            ModelSelectorOptions {
                available_models: Some(models),
                configured_providers: Some(vec!["signed".into()]),
                get_rows: Some(Rc::new(|| 24.0)),
                ..Default::default()
            },
        )
    }

    #[test]
    fn rendered_picker_supports_search_focus_confirmation_and_empty_results() {
        let mut selector = picker(vec![
            item("signed", "alpha", "Alpha model").model,
            item("other", "beta", "Beta model").model,
        ]);
        selector.set_focused(true);
        let lines = selector.render(80.0);
        let text = pi_tui::utils::strip_ansi(&lines.join("\n"));
        assert!(text.contains("Models"));
        assert!(text.contains("Signed-in providers first"));
        assert!(text.contains("sign in"));
        assert!(lines.join("\n").contains(pi_tui::tui::CURSOR_MARKER));
        selector.handle_input("beta");
        assert_eq!(selector.filtered_models.len(), 1);
        assert_eq!(selector.get_search_value(), "beta");
        selector.handle_input("\r");
        assert_eq!(selector.selected_model.take().unwrap().provider, "other");
        selector.handle_input("-no-such-model");
        assert!(selector
            .render(80.0)
            .join("\n")
            .contains("No matching models"));
        selector.handle_input("\r");
        assert!(selector.selected_model.is_none());
        selector.handle_input("\x1b");
        assert!(selector.cancelled);
    }

    #[test]
    fn refresh_preserves_query_and_selection_and_accepts_an_empty_catalog() {
        let models = vec![
            item("signed", "model1", "One").model,
            item("signed", "model2", "Two").model,
        ];
        let mut selector = picker(models.clone());
        selector.handle_input("model");
        selector.handle_input("\x1b[B");
        selector.update_state(None, Some(models), Some(vec!["signed".into()]));
        assert_eq!(selector.get_search_value(), "model");
        assert_eq!(
            selector.get_selected_model_key().as_deref(),
            Some("signed/model2")
        );
        selector.update_state(None, Some(vec![]), Some(vec![]));
        assert_eq!(selector.get_search_value(), "model");
        assert!(selector.filtered_models.is_empty());
        assert!(selector
            .render(80.0)
            .join("\n")
            .contains("No matching models"));
    }

    #[test]
    fn ordering_prefers_signed_in_current_and_recent_models() {
        let models = vec![
            item("signed", "a-unseen", "Unseen").model,
            item("signed", "z-recent", "Recent").model,
            item("signed", "y-current", "Current").model,
            item("other", "a-unconfigured", "Other").model,
        ];
        let mut selector = picker(models.clone());
        selector.current_model = Some(models[2].clone());
        selector.recent_rank.insert("signed/z-recent".into(), 0);
        selector.load_models();
        let ids: Vec<_> = selector
            .filtered_models
            .iter()
            .map(|m| m.id.as_str())
            .collect();
        assert_eq!(ids, ["y-current", "z-recent", "a-unseen", "a-unconfigured"]);
        selector.handle_input(" \t ");
        assert_eq!(selector.filtered_models.len(), 4);
    }

    #[test]
    fn initial_search_ranks_exact_short_ids_and_provider_qualified_queries() {
        let models = vec![
            item("signed", "namespace/gpt-5", "Target").model,
            item("signed", "gpt-5-mini", "Mini").model,
            item("other", "gpt-5", "Other").model,
        ];
        let mut selector = ModelSelectorComponent::new(
            None,
            vec![],
            ModelSelectorOptions {
                available_models: Some(models),
                configured_providers: Some(vec!["signed".into()]),
                initial_search_input: Some("gpt5".into()),
                ..Default::default()
            },
        );
        assert_eq!(selector.filtered_models[0].id, "namespace/gpt-5");
        selector.set_search_value("other/gpt-5");
        assert_eq!(selector.filtered_models[0].provider, "other");
    }

    #[test]
    fn scoped_search_toggles_with_the_configured_binding() {
        let models = vec![
            item("signed", "alpha", "Alpha").model,
            item("signed", "beta", "Beta").model,
        ];
        let mut selector = picker(models.clone());
        selector.scoped_models = vec![ScopedModelItem {
            model: models[0].clone(),
            thinking_level: None,
        }];
        selector.load_models();
        selector.set_scope(ModelScope::Scoped);
        selector.handle_input("beta");
        assert!(selector.filtered_models.is_empty());
        selector.handle_input("\x1bs");
        assert_eq!(selector.scope, ModelScope::All);
        assert_eq!(selector.get_search_value(), "beta");
        assert_eq!(selector.filtered_models[0].id, "beta");
        selector.handle_input("\x1bs");
        assert!(selector.filtered_models.is_empty());
    }

    #[test]
    fn composed_option_s_toggles_scope_without_changing_search() {
        let models = vec![
            item("signed", "alpha", "Alpha").model,
            item("signed", "beta", "Beta").model,
        ];
        let mut selector = picker(models.clone());
        selector.scoped_models = vec![ScopedModelItem {
            model: models[0].clone(),
            thinking_level: None,
        }];
        selector.load_models();
        selector.set_scope(ModelScope::Scoped);
        selector.set_search_value("beta");
        selector.handle_input("ß");
        assert_eq!(selector.scope, ModelScope::All);
        assert_eq!(selector.get_search_value(), "beta");
        assert_eq!(selector.filtered_models[0].id, "beta");
        selector.handle_input("ß");
        assert_eq!(selector.scope, ModelScope::Scoped);
        assert_eq!(selector.get_search_value(), "beta");

        let mut kb = get_keybindings();
        kb.set_user_bindings(indexmap::IndexMap::from([
            ("app.model.toggleScope".into(), vec!["alt+a".into()]),
        ]));
        pi_tui::keybindings::set_keybindings(kb);
        selector.set_search_value("");
        selector.handle_input("ß");
        assert_eq!(selector.scope, ModelScope::Scoped);
        assert_eq!(selector.get_search_value(), "ß");
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
    }

    #[test]
    fn short_viewports_keep_search_and_selection_visible_after_resize() {
        let rows = Rc::new(std::cell::Cell::new(24.0));
        let mut selector = picker(
            (0..20)
                .map(|n| item("signed", &format!("model{n}"), "Model detail").model)
                .collect(),
        );
        let viewport = rows.clone();
        selector.viewport.get_rows = Some(Rc::new(move || viewport.get()));
        selector.set_focused(true);
        for height in [24, 12, 16, 24] {
            rows.set(height as f64);
            selector.handle_input("\x1b[A");
            let lines = selector.render(80.0);
            assert!(
                lines.len() <= height,
                "{height} rows: {} rendered",
                lines.len()
            );
            assert!(lines
                .iter()
                .all(|line| pi_tui::utils::visible_width(line) <= 80));
            assert!(lines
                .join("\n")
                .contains(&selector.filtered_models[selector.selected_index].id));
            assert!(lines.join("\n").contains(pi_tui::tui::CURSOR_MARKER));
        }
    }

    #[test]
    fn left_arrow_edits_search_before_cancelling_at_its_start() {
        let mut selector = picker(vec![]);
        selector.handle_input("ab");
        selector.handle_input("\x1b[D");
        assert!(!selector.cancelled);
        selector.handle_input("\x1b[D");
        assert!(!selector.cancelled);
        selector.handle_input("\x1b[D");
        assert!(selector.cancelled);
    }
}
