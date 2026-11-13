//! Bounded, presentation-only force layout over the existing roster contract.
use super::communication_activity::{ActivityState, PULSE};
use super::context_health_indicator;
use super::generation_throughput::{parent_keys, summary_keys};
use crate::modes::agents_view::agents_view_state::get_parent_keys;
use crate::modes::daemon::agent_roster::{
    classify_session_roster_status, AgentRosterStatus, RosterSummaryView,
};
use crate::modes::daemon::daemon_session_list::SessionSummary;
#[path = "textmas_activity_view.rs"]
mod textmas_activity_view;
use textmas_activity_view::Projection;
use crate::modes::interactive::components::subagent_summary_line::{
    linkage_row_for_roster_summary, RosterParent,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::{Duration, Instant};

const MAX_NODES: usize = 64;
const FRAME: Duration = Duration::from_millis(80);
const SETTLE: Duration = Duration::from_secs(6);

#[derive(Clone)]
struct Node {
    id: String,
    parent: Option<String>,
    working: bool,
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    target_x: f32,
    target_y: f32,
}

#[derive(Default)]
pub(super) struct Graph {
    root: String,
    nodes: Vec<Node>,
    activity: ActivityState,
    textmas: Projection,
    frame: Option<Instant>,
    started: Option<Instant>,
    settle_until: Option<Instant>,
    size: (usize, usize),
    omitted: usize,
}

impl Graph {
    pub(super) fn observe(
        &mut self,
        rows: &[SessionSummary],
        parent: &RosterParent,
        root_working: bool,
        now: Instant,
    ) {
        let root = parent.session_id.as_deref().unwrap_or("");
        if self.root != root {
            *self = Self {
                root: root.into(),
                started: Some(now),
                ..Self::default()
            };
        }
        if root.is_empty() {
            return;
        }
        // Index all aliases once, including saved parents and nested children.
        let mut children: BTreeMap<String, Vec<&SessionSummary>> = BTreeMap::new();
        for row in rows
            .iter()
            .filter(|row| row.runtime_kind.as_deref() == Some("subagent"))
        {
            for key in get_parent_keys(&linkage_row_for_roster_summary(row)) {
                children.entry(key).or_default().push(row);
            }
        }
        for siblings in children.values_mut() {
            siblings.sort_by_key(|row| (row.lifecycle != "live", row.session_id.as_str()));
        }
        let mut selected = vec![(root.to_string(), None, root_working)];
        let mut family = BTreeSet::from([root.to_string()]);
        let mut pending: VecDeque<_> = parent_keys(parent)
            .into_iter()
            .map(|key| (key, root.to_string()))
            .collect();
        let mut visited = BTreeSet::new();
        while let Some((key, owner)) = pending.pop_front() {
            if !visited.insert(key.clone()) {
                continue;
            }
            for row in children.remove(&key).unwrap_or_default() {
                if row.session_id.is_empty() || !family.insert(row.session_id.clone()) {
                    continue;
                }
                pending.extend(
                    summary_keys(row)
                        .into_iter()
                        .map(|key| (key, row.session_id.clone())),
                );
                if selected.len() < MAX_NODES {
                    let working = row.lifecycle == "live"
                        && row.roster_status.unwrap_or_else(|| {
                            classify_session_roster_status(
                                &RosterSummaryView {
                                    active_session_id: row.active_session_id.clone(),
                                    activity: Some(row.activity.clone()),
                                    is_session_active: Some(row.is_session_active),
                                },
                                false,
                            )
                        }) == AgentRosterStatus::Running;
                    selected.push((row.session_id.clone(), Some(owner.clone()), working));
                }
            }
        }
        self.omitted = family.len().saturating_sub(MAX_NODES);
        let changed = self.nodes.len() != selected.len()
            || selected.iter().any(|(id, parent, _)| {
                !self
                    .nodes
                    .iter()
                    .any(|node| node.id == *id && node.parent == *parent)
            });
        let mut previous: BTreeMap<_, _> = self
            .nodes
            .drain(..)
            .map(|node| (node.id.clone(), node))
            .collect();
        let mut added = BTreeSet::new();
        for (id, parent, working) in selected {
            let node = previous.remove(&id).unwrap_or_else(|| {
                added.insert(id.clone());
                Node {
                    id,
                    parent: parent.clone(),
                    working,
                    x: 0.5,
                    y: 0.5,
                    vx: 0.0,
                    vy: 0.0,
                    target_x: 0.5,
                    target_y: 0.5,
                }
            });
            self.nodes.push(Node {
                parent,
                working,
                ..node
            });
        }
        if let Some(main) = self.nodes.first_mut() {
            main.x = 0.5;
            main.y = 0.5;
        }
        if changed {
            self.arrange();
            for node in &mut self.nodes {
                if added.contains(&node.id) {
                    node.x = node.target_x;
                    node.y = node.target_y;
                }
            }
            self.settle_until = Some(now + SETTLE);
        }
        family.remove(root);
        // Same deduplication/reconnect rules as TXT/OCC, extended to descendants.
        self.activity.observe_scope(rows, parent, &family, now);
    }

    /// Observe the authoritative, attachment-scoped cache after the roster.
    /// No activity entry is inserted into rows, sessions or the usage ledger.
    pub(super) fn observe_activity(
        &mut self,
        activity: Option<&crate::core::occ::textmas_activity::TextMasActivitySnapshot>,
        now: Instant,
    ) {
        self.observe_activity_state(activity, activity.is_some_and(|activity| activity.active), now);
    }

    pub(super) fn observe_activity_state(
        &mut self,
        activity: Option<&crate::core::occ::textmas_activity::TextMasActivitySnapshot>,
        owner_active: bool,
        now: Instant,
    ) {
        // Preserve the legacy no-activity path: no root allocation and no traversal.
        if activity.is_none() && self.textmas.is_empty() { return; }
        let root = self.root.clone();
        if self.textmas.observe_with_owner(&root, activity, owner_active, now) {
            self.settle_until = Some(now + SETTLE);
            // Cancellation belongs to the existing graph frame task, not a new timer.
            self.frame = None;
        }
    }

    fn arrange(&mut self) {
        // Give each family branch its own sector. Larger subtrees get more
        // room; springs relax around these anchors instead of crossing branches.
        let mut children = vec![Vec::new(); self.nodes.len()];
        for (i, node) in self.nodes.iter().enumerate().skip(1) {
            if let Some(parent) = self.nodes.iter().position(|n| Some(&n.id) == node.parent.as_ref()) {
                children[parent].push(i);
            }
        }
        for siblings in &mut children {
            siblings.sort_by(|&a, &b| self.nodes[a].id.cmp(&self.nodes[b].id));
        }
        let mut weights = vec![1usize; self.nodes.len()];
        let mut levels = vec![0usize; self.nodes.len()];
        // observe() selects parents before children, including nested families.
        for i in (0..self.nodes.len()).rev() {
            if !children[i].is_empty() {
                weights[i] = children[i].iter().map(|&child| weights[child]).sum();
                levels[i] = 1 + children[i].iter().map(|&child| levels[child]).max().unwrap_or(0);
            }
        }
        let mut pending = VecDeque::from([(0, -std::f32::consts::PI, std::f32::consts::TAU, 0)]);
        while let Some((i, start, span, depth)) = pending.pop_front() {
            let total = children[i].iter().map(|&child| weights[child]).sum::<usize>().max(1);
            let mut angle = start;
            for &child in &children[i] {
                let sector = span * weights[child] as f32 / total as f32;
                let middle = angle + sector * 0.5;
                let radius = 0.44 * (depth + 1) as f32 / (depth + 1 + levels[child]) as f32;
                self.nodes[child].target_x = 0.5 + radius * middle.cos();
                self.nodes[child].target_y = 0.5 + radius * middle.sin();
                let fan = sector.min(std::f32::consts::PI * 0.75);
                pending.push_back((child, middle - fan * 0.5, fan, depth + 1));
                angle += sector;
            }
        }
    }

    fn step(&mut self, dt: f32, aspect: f32) {
        let mut forces = vec![(0.0f32, 0.0f32); self.nodes.len()];
        for i in 0..self.nodes.len() {
            for j in i + 1..self.nodes.len() {
                let dx = (self.nodes[i].x - self.nodes[j].x) * aspect;
                let dy = self.nodes[i].y - self.nodes[j].y;
                let d2 = (dx * dx + dy * dy).max(0.005);
                let repulsion = 0.012 * (8.0 / self.nodes.len() as f32).min(1.0) / d2;
                let (fx, fy) = (dx * repulsion, dy * repulsion);
                forces[i].0 += fx;
                forces[i].1 += fy;
                forces[j].0 -= fx;
                forces[j].1 -= fy;
            }
        }
        for (i, node) in self.nodes.iter().enumerate().skip(1) {
            let Some(j) = self
                .nodes
                .iter()
                .position(|n| Some(&n.id) == node.parent.as_ref())
            else {
                continue;
            };
            let dx = (self.nodes[j].x - node.x) * aspect;
            let dy = self.nodes[j].y - node.y;
            let length = (dx * dx + dy * dy).sqrt().max(0.01);
            let target_dx = (self.nodes[j].target_x - node.target_x) * aspect;
            let target_dy = self.nodes[j].target_y - node.target_y;
            let rest = (target_dx * target_dx + target_dy * target_dy).sqrt().max(0.1);
            let spring = (length - rest) * 2.0 / length;
            forces[i].0 += dx * spring;
            forces[i].1 += dy * spring;
            forces[j].0 -= dx * spring;
            forces[j].1 -= dy * spring;
        }
        let drag = (-8.0 * dt).exp();
        for (node, (fx, fy)) in self.nodes.iter_mut().zip(forces).skip(1) {
            node.vx = (node.vx + (fx / aspect + (node.target_x - node.x) * 4.0) * dt) * drag;
            node.vy = (node.vy + (fy + (node.target_y - node.y) * 4.0) * dt) * drag;
            node.x = (node.x + node.vx.clamp(-0.8, 0.8) * dt).clamp(0.06, 0.94);
            node.y = (node.y + node.vy.clamp(-0.8, 0.8) * dt).clamp(0.06, 0.94);
        }
    }

    pub(super) fn next_frame(&self, now: Instant) -> Option<Instant> {
        (self.nodes.iter().any(|node| node.working)
            || self.activity.live_links(now).next().is_some()
            || self.textmas.is_active()
            || self.settle_until.is_some_and(|until| now < until))
        .then(|| self.frame.unwrap_or(now) + FRAME)
    }

    pub(super) fn render(
        &mut self,
        root: &str,
        width: usize,
        height: usize,
        now: Instant,
    ) -> Vec<String> {
        if root != self.root || width < 8 || height == 0 {
            return Vec::new();
        }
        let (width, height) = (width.min(40), height.min(6));
        let activity_rows = self.textmas.lines(width, now, height.saturating_sub(2));
        let height = height.saturating_sub(activity_rows.len()).max(1);
        if self.size != (width, height) {
            self.size = (width, height);
            self.settle_until = Some(now + SETTLE);
        }
        let aspect = width as f32 / (height * 2) as f32;
        if self.settle_until.is_some_and(|until| now < until) {
            let dt = self
                .frame
                .map(|last| now.saturating_duration_since(last).as_secs_f32())
                .unwrap_or(0.16)
                .min(0.16);
            for _ in 0..4 {
                self.step(dt / 4.0, aspect);
            }
        }
        self.frame = Some(now);
        let mut canvas = Canvas::new(width, height);
        let mut points: BTreeMap<&str, (i32, i32)> = BTreeMap::new();
        let mut activity_points: BTreeMap<&str, (i32, i32)> = BTreeMap::new();
        let mut occupied = BTreeSet::new();
        // Reserve at most nine role/cohort cells before placing real agents.
        // Sixty-four maps can never evict roster nodes from this view.
        let activity_nodes = self.textmas.nodes();
        let columns = activity_nodes.len().div_ceil(height.min(2)).max(1);
        for (index, activity) in activity_nodes.iter().enumerate() {
            let row = index / columns;
            let column = index % columns;
            let x = (((column * 2 + 1) * width * 2 / (columns * 2)) as i32).clamp(1, (width * 2 - 2) as i32);
            let rows = activity_nodes.len().div_ceil(columns).max(1);
            let y = (((row * 2 + 1) * height / (rows * 2)).min(height - 1) * 4 + 2) as i32;
            activity_points.insert(activity.id.as_str(), (x, y));
            occupied.insert((x as usize / 2, y as usize / 4));
        }
        let mut crowding = vec![0; width * height];
        for node in &self.nodes {
            let x = 1 + (node.x * (width * 2 - 3) as f32).round() as i32;
            // Map to rows directly: rounding braille pixels first biases
            // symmetric positions toward the upper half of a short graph.
            let y = (node.y * height as f32).floor() as i32 * 4 + 2;
            // Leave a readable gap between markers where space permits.
            let cell = (0..height)
                .flat_map(|row| (0..width).map(move |col| (col, row)))
                .filter(|cell| !occupied.contains(cell))
                .min_by_key(|&(col, row)| {
                    let dx = col as i32 * 2 + 1 - x;
                    let dy = row as i32 * 4 + 2 - y;
                    // Equal-distance horizontal shifts retain vertical balance.
                    (dx * dx + dy * dy + crowding[row * width + col], dy.abs())
                })
                .unwrap_or((x as usize / 2, y as usize / 4));
            occupied.insert(cell);
            for col in [cell.0.checked_sub(1), cell.0.checked_add(1)].into_iter().flatten() {
                if col < width { crowding[cell.1 * width + col] += 48; }
            }
            points.insert(
                node.id.as_str(),
                (cell.0 as i32 * 2 + 1, cell.1 as i32 * 4 + 2),
            );
        }
        for node in &self.nodes {
            if let Some(parent) = node.parent.as_deref().and_then(|id| points.get(id)) {
                canvas.line(*parent, points[node.id.as_str()], 1);
            }
        }
        for activity in activity_nodes {
            let end = activity_points[activity.id.as_str()];
            for dependency in &activity.dependencies {
                if let Some(&start) = activity_points.get(dependency.as_str()) {
                    let unrelated_role_cells = activity_points
                        .iter()
                        .filter(|(id, _)| {
                            id.as_bytes() != activity.id.as_str().as_bytes()
                                && id.as_bytes() != dependency.as_str().as_bytes()
                        })
                        .map(|(_, point)| ((point.0 / 2) as usize, (point.1 / 4) as usize))
                        .collect();
                    canvas.activity_dependency(start, end, &unrelated_role_cells);
                }
            }
        }
        for (from, to, until) in self.activity.live_links(now) {
            if let (Some(&a), Some(&b)) = (points.get(from), points.get(to)) {
                canvas.line(a, b, 2);
                let progress = (1.0
                    - until.saturating_duration_since(now).as_secs_f32() / PULSE.as_secs_f32())
                .clamp(0.0, 1.0);
                let x = a.0 as f32 + (b.0 - a.0) as f32 * progress;
                let y = a.1 as f32 + (b.1 - a.1) as f32 * progress;
                canvas.dot(x.round() as i32, y.round() as i32, 3);
            }
        }
        let pulse = self
            .started
            .map(|at| now.saturating_duration_since(at).as_millis() % 1200 < 600)
            .unwrap_or(true);
        for (i, node) in self.nodes.iter().enumerate() {
            let (x, y) = points[node.id.as_str()];
            let light = if node.working {
                if pulse {
                    3
                } else {
                    2
                }
            } else {
                1
            };
            canvas.marker(
                x,
                y,
                if i == 0 {
                    '◆'
                } else if node.working {
                    '●'
                } else {
                    '○'
                },
                light,
            );
        }
        for activity in activity_nodes {
            let (x, y) = activity_points[activity.id.as_str()];
            canvas.marker_green(x, y, activity.glyph,
                activity.intensity(activity.counts.running > 0 && self.textmas.owner_active(), pulse));
        }
        let mut lines = canvas.render();
        if self.omitted > 0 {
            let label = format!("+{}", self.omitted);
            if let Some(line) = lines.last_mut() {
                *line = format!(
                    "{}\x1b[38;5;39m{label}\x1b[39m",
                    pi_tui::utils::truncate_to_width(
                        line,
                        width.saturating_sub(label.len()) as f64,
                        "",
                        true
                    )
                );
            }
        }
        for row in activity_rows {
            let padding = width.saturating_sub(pi_tui::utils::visible_width(&row));
            lines.push(format!("{row}{}", " ".repeat(padding)));
        }
        lines
    }
}

struct Canvas {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
    markers: BTreeMap<(usize, usize), (char, u8, bool)>,
}
impl Canvas {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; width * height * 8],
            markers: BTreeMap::new(),
        }
    }
    fn marker(&mut self, x: i32, y: i32, glyph: char, light: u8) {
        self.markers
            .insert((x as usize / 2, y as usize / 4), (glyph, light, false));
    }
    fn marker_green(&mut self, x: i32, y: i32, glyph: char, light: u8) {
        self.markers
            .insert((x as usize / 2, y as usize / 4), (glyph, light, true));
    }
    fn dot(&mut self, x: i32, y: i32, light: u8) {
        self.coloured_dot(x, y, light, false);
    }
    fn coloured_dot(&mut self, x: i32, y: i32, light: u8, green: bool) {
        if x >= 0 && y >= 0 && (x as usize) < self.width * 2 && (y as usize) < self.height * 4 {
            let pixel = &mut self.pixels[y as usize * self.width * 2 + x as usize];
            // The green bit is used only by recorded TextMAS dependencies.
            *pixel = (*pixel & 3).max(light) | if green || *pixel & 4 != 0 { 4 } else { 0 };
        }
    }
    fn line(&mut self, a: (i32, i32), b: (i32, i32), light: u8) {
        self.coloured_line(a, b, light, false);
    }
    fn activity_dependency(
        &mut self,
        start: (i32, i32),
        end: (i32, i32),
        unrelated_role_cells: &BTreeSet<(usize, usize)>,
    ) {
        let cell = |point: (i32, i32)| ((point.0 / 2) as usize, (point.1 / 4) as usize);
        let candidates = [
            vec![start, end],
            vec![start, (start.0, end.1), end],
            vec![start, (end.0, start.1), end],
        ];
        'candidate: for candidate in candidates {
            for segment in candidate.windows(2) {
                let (a, b) = (segment[0], segment[1]);
                let steps = (b.0 - a.0).abs().max((b.1 - a.1).abs()).max(1);
                for i in 0..=steps {
                    let point = (a.0 + (b.0 - a.0) * i / steps, a.1 + (b.1 - a.1) * i / steps);
                    if unrelated_role_cells.contains(&cell(point)) {
                        continue 'candidate;
                    }
                }
            }
            for segment in candidate.windows(2) {
                self.line_green(segment[0], segment[1], 1);
            }
            return;
        }
        // A crowded short canvas cannot draw every genuine cohort link without
        // implying a third role. Keep that link in Projection, not in its lines.
    }

    fn line_green(&mut self, a: (i32, i32), b: (i32, i32), light: u8) {
        self.coloured_line(a, b, light, true);
    }
    fn coloured_line(&mut self, a: (i32, i32), b: (i32, i32), light: u8, green: bool) {
        let steps = (b.0 - a.0).abs().max((b.1 - a.1).abs()).max(1);
        for i in 0..=steps {
            self.coloured_dot(
                a.0 + (b.0 - a.0) * i / steps,
                a.1 + (b.1 - a.1) * i / steps,
                light,
                green,
            );
        }
    }
    fn render(&self) -> Vec<String> {
        (0..self.height)
            .map(|row| {
                let mut line = String::new();
                let mut previous = 0;
                let mut previous_green = false;
                for col in 0..self.width {
                    let (mut bits, mut light, mut green) = (0u8, 0u8, false);
                    for (dx, dy, bit) in [
                        (0, 0, 1),
                        (0, 1, 2),
                        (0, 2, 4),
                        (1, 0, 8),
                        (1, 1, 16),
                        (1, 2, 32),
                        (0, 3, 64),
                        (1, 3, 128),
                    ] {
                        let pixel = self.pixels[(row * 4 + dy) * self.width * 2 + col * 2 + dx];
                        if pixel > 0 {
                            bits |= bit;
                            light = light.max(pixel & 3);
                            green |= pixel & 4 != 0;
                        }
                    }
                    let mut glyph = if bits == 0 {
                        ' '
                    } else {
                        char::from_u32(0x2800 + u32::from(bits)).unwrap_or(' ')
                    };
                    if let Some(&(marker, intensity, marker_green)) = self.markers.get(&(col, row)) {
                        glyph = marker;
                        light = intensity;
                        green = marker_green;
                    }
                    if light != previous || green != previous_green {
                        if green {
                            let strength = match light { 3 => "1", 1 => "2", _ => "0" };
                            line.push_str(&format!("\x1b[22;{strength};38;5;{}m",
                                super::context_health_indicator::CONTEXT_HEALTH_GREEN));
                        } else {
                            line.push_str(match light {
                                3 => "\x1b[22;1;38;5;39m",
                                2 => "\x1b[22;38;5;39m",
                                _ => "\x1b[22;2;38;5;39m",
                            });
                        }
                        previous = light;
                        previous_green = green;
                    }
                    line.push(glyph);
                }
                line.push_str("\x1b[22;39m");
                line
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "agent_graph_tests.rs"]
mod tests;
#[cfg(test)]
#[path = "agent_graph_activity_tests.rs"]
pub(super) mod activity_tests;

#[cfg(test)]
#[path = "agent_graph_activity_benchmark.rs"]
mod activity_benchmark;
