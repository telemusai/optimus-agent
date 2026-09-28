//! Port of packages/coding-agent/src/modes/interactive/components/oauth-selector.ts
//!
//! PARTIAL: `MenuPanel`, `MenuList`, `MenuRow`, `MenuSearchInput` live in
//! components/menu-panel.ts -> `super::menu_panel`. The providers, tabs,
//! filtering and status logic of this file is ported 1:1 against that real
//! module.

use crate::core::auth_storage::{AuthCredential, AuthStatus, AuthStorage};
use crate::core::prime_inference_auth::PRIME_INFERENCE_PROVIDER_ID;
use crate::modes::interactive::theme::theme::theme;

use super::menu_panel::{
    get_menu_list_layout, MenuListLayout, MenuListLayoutOptions, MenuViewportProvider,
};
use super::menu_panel::{
    MenuList, MenuPanel, MenuPanelOptions, MenuRow, MenuRowOptions, MenuSearchInput,
};
use pi_tui::components::{spacer::Spacer, truncated_text::TruncatedText};
use pi_tui::tui::{Component, Focusable};
use std::{cell::RefCell, rc::Rc};

use pi_tui::tui::Component as _;

/// `type AuthSelectorCategory = "provider" | "service"`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuthSelectorCategory {
    Provider,
    Service,
}

impl AuthSelectorCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            AuthSelectorCategory::Provider => "provider",
            AuthSelectorCategory::Service => "service",
        }
    }

    fn from_str(value: &str) -> Self {
        if value == "service" {
            AuthSelectorCategory::Service
        } else {
            AuthSelectorCategory::Provider
        }
    }

    fn label(self) -> &'static str {
        match self {
            AuthSelectorCategory::Provider => "Providers",
            AuthSelectorCategory::Service => "MCP Connections",
        }
    }
}

/// `AuthSelectorProvider`
#[derive(Debug, Clone, PartialEq)]
pub struct AuthSelectorProvider {
    pub id: String,
    pub name: String,
    /// `"oauth" | "api_key"`
    pub auth_type: String,
    /// Which tab the entry belongs to. Defaults to "provider" when omitted.
    pub category: Option<AuthSelectorCategory>,
}

impl AuthSelectorProvider {
    fn in_category(&self, category: AuthSelectorCategory) -> bool {
        self.category.unwrap_or(AuthSelectorCategory::Provider) == category
    }
}

/// `OAuthSelectorOptions`
#[derive(Default)]
pub struct OAuthSelectorOptions {
    pub get_rows: Option<Rc<dyn Fn() -> f64>>,
    pub initial_category: Option<AuthSelectorCategory>,
    pub header_rows: Option<f64>,
    pub header: Option<Rc<RefCell<dyn Component>>>,
    pub get_header_rows: Option<Box<dyn Fn() -> f64>>,
    pub title: Option<String>,
    pub subtitle: Option<String>,
    pub search_placeholder: Option<String>,
}

/// Port of `compareAuthSelectorProviders`.
pub fn compare_auth_selector_providers(
    a: &AuthSelectorProvider,
    b: &AuthSelectorProvider,
) -> std::cmp::Ordering {
    if a.auth_type != b.auth_type {
        return if a.auth_type == "oauth" {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        };
    }
    a.name.cmp(&b.name)
}

/// `PREFERRED_VISIBLE_PROVIDERS`
pub const PREFERRED_VISIBLE_PROVIDERS: usize = 8;
/// `PROVIDER_LIST_RESERVED_ROWS`
pub const PROVIDER_LIST_RESERVED_ROWS: usize = 7;
/// Extra fixed rows the Providers/MCP Connections tab bar (text + spacer) consumes.
pub const TAB_BAR_RESERVED_ROWS: f64 = 2.0;
/// `PROVIDER_SCROLL_INDICATOR_ROWS`
pub const PROVIDER_SCROLL_INDICATOR_ROWS: usize = 1;

/// The storage surface the selector reads.
pub trait AuthStorageLike {
    fn get(&self, provider: &str) -> Option<AuthCredential>;
    fn get_auth_status(&self, provider: &str) -> AuthStatus;
}

impl AuthStorageLike for AuthStorage {
    fn get(&self, provider: &str) -> Option<AuthCredential> {
        AuthStorage::get(self, provider)
    }

    fn get_auth_status(&self, provider: &str) -> AuthStatus {
        AuthStorage::get_auth_status(self, provider)
    }
}

pub struct OAuthSelectorComponent {
    pub search_value: String,
    pub cursor: usize,
    /// `"login" | "logout"`
    pub mode: String,
    pub all_providers: Vec<AuthSelectorProvider>,
    pub filtered_providers: Vec<AuthSelectorProvider>,
    pub selected_index: usize,
    pub search_query: String,
    pub categories: Vec<AuthSelectorCategory>,
    pub active_category: AuthSelectorCategory,
    /// `authStorage` - the credential store the selector reads.
    pub auth_storage: Box<dyn AuthStorageLike>,
    /// `getAuthStatus` override; when `None` the storage is read directly.
    pub get_auth_status: Option<Box<dyn Fn(&str) -> AuthStatus>>,
    pub list_layout: MenuListLayout,
    pub viewport: MenuViewportProvider,
    pub get_header_rows: Box<dyn Fn() -> f64>,
    pub has_tab_bar: bool,
    pub has_header: bool,
    pub search_input: Rc<RefCell<MenuSearchInput>>,
    header: Option<Rc<RefCell<dyn Component>>>,
    /// Set when the user picked a provider (`onSelectCallback`).
    pub selected_provider: Option<AuthSelectorProvider>,
    /// Set when the user cancelled (`onCancelCallback`).
    pub cancelled: bool,
    pub title: String,
    pub subtitle: String,
    pub search_placeholder: String,
    /// The visible slice `updateList` computed.
    pub visible_range: (usize, usize),
}

impl OAuthSelectorComponent {
    /// `constructor(mode, authStorage, providers, onSelect, onCancel, getAuthStatus?, options?)`.
    pub fn new(
        mode: &str,
        auth_storage: Box<dyn AuthStorageLike>,
        providers: Vec<AuthSelectorProvider>,
        get_auth_status: Option<Box<dyn Fn(&str) -> AuthStatus>>,
        options: OAuthSelectorOptions,
    ) -> Self {
        let mut component = Self {
            search_value: String::new(),
            cursor: 0,
            mode: mode.to_string(),
            all_providers: Vec::new(),
            filtered_providers: Vec::new(),
            selected_index: 0,
            search_query: String::new(),
            categories: Vec::new(),
            active_category: AuthSelectorCategory::Provider,
            auth_storage,
            get_auth_status,
            list_layout: MenuListLayout {
                compact: false,
                visible_items: 0,
            },
            viewport: MenuViewportProvider {
                get_rows: options.get_rows.clone(),
            },
            get_header_rows: {
                let has_header = options.header_rows.is_some() || options.header.is_some();
                let rows = options.header_rows.unwrap_or(TAB_BAR_RESERVED_ROWS);
                options
                    .get_header_rows
                    .unwrap_or_else(|| Box::new(move || if has_header { rows } else { 0.0 }))
            },
            has_tab_bar: false,
            has_header: options.header_rows.is_some(),
            search_input: Rc::new(RefCell::new(MenuSearchInput::new(
                options
                    .search_placeholder
                    .clone()
                    .unwrap_or_else(|| "Search providers".into()),
            ))),
            header: options.header,
            selected_provider: None,
            cancelled: false,
            title: options.title.clone().unwrap_or_else(|| {
                if mode == "login" {
                    "Providers".to_string()
                } else {
                    "Saved Credentials".to_string()
                }
            }),
            subtitle: options.subtitle.clone().unwrap_or_else(|| {
                if mode == "login" {
                    "Connect with a subscription or API key.".to_string()
                } else {
                    "Choose a credential to remove.".to_string()
                }
            }),
            search_placeholder: options
                .search_placeholder
                .clone()
                .unwrap_or_else(|| "Search providers".to_string()),
            visible_range: (0, 0),
        };

        component.list_layout = get_menu_list_layout(MenuListLayoutOptions {
            preferred_visible_items: PREFERRED_VISIBLE_PROVIDERS,
            reserved_rows: PROVIDER_LIST_RESERVED_ROWS,
            comfortable_item_rows: 3,
            compact_item_rows: Some(2),
            ..Default::default()
        });
        component.all_providers = component.sort_providers(&component.all_providers, &providers);
        component.filtered_providers = component.all_providers.clone();

        let mut categories: Vec<AuthSelectorCategory> = Vec::new();
        for provider in &providers {
            let category = provider.category.unwrap_or(AuthSelectorCategory::Provider);
            if !categories.contains(&category) {
                let present: std::collections::HashSet<AuthSelectorCategory> = providers
                    .iter()
                    .map(|p| p.category.unwrap_or(AuthSelectorCategory::Provider))
                    .collect();
                categories = [
                    AuthSelectorCategory::Provider,
                    AuthSelectorCategory::Service,
                ]
                .into_iter()
                .filter(|c| present.contains(c))
                .collect();
                let _ = category;
                break;
            }
        }
        component.categories = categories;
        component.active_category = match options.initial_category {
            Some(category) if component.categories.contains(&category) => category,
            _ => component
                .categories
                .first()
                .copied()
                .unwrap_or(AuthSelectorCategory::Provider),
        };
        component.has_tab_bar = component.categories.len() > 1;

        component.filter_providers("");
        component
    }

    fn in_active_category(&self, provider: &AuthSelectorProvider) -> bool {
        provider.in_category(self.active_category)
    }

    /// Port of `switchCategory(direction)`.
    pub fn switch_category(&mut self, direction: i32) {
        if self.categories.len() < 2 {
            return;
        }
        let current = self
            .categories
            .iter()
            .position(|category| *category == self.active_category)
            .unwrap_or(0) as i32;
        let next =
            (current + direction + self.categories.len() as i32) % self.categories.len() as i32;
        self.active_category = self.categories[next as usize];
        self.selected_index = 0;
        self.search_input.borrow_mut().set_value("");
        self.filter_providers("");
    }

    /// Port of `updateTabBar`.
    pub fn update_tab_bar(&self) -> String {
        if !self.has_tab_bar {
            return String::new();
        }
        let rendered = self
            .categories
            .iter()
            .map(|category| {
                if *category == self.active_category {
                    theme().bold(&theme().fg("accent", category.label()))
                } else {
                    theme().fg("muted", category.label())
                }
            })
            .collect::<Vec<String>>()
            .join(&theme().fg("muted", "  \u{b7}  "));
        format!(
            "{rendered}   {}",
            theme().fg("muted", "\u{2190}/\u{2192} switch")
        )
    }

    /// Port of `filterProviders`.
    pub fn filter_providers(&mut self, query: &str) {
        let query_changed = query != self.search_query;
        self.search_query = query.to_string();
        let in_category: Vec<AuthSelectorProvider> = self
            .all_providers
            .iter()
            .filter(|provider| self.in_active_category(provider))
            .cloned()
            .collect();
        self.filtered_providers = if query.is_empty() {
            in_category
        } else {
            pi_tui::fuzzy::fuzzy_filter(&in_category, query, &|provider| {
                format!("{} {} {}", provider.name, provider.id, provider.auth_type)
            })
        };
        self.selected_index = if query_changed {
            0
        } else {
            self.selected_index
                .min(self.filtered_providers.len().saturating_sub(1))
        };
        self.update_list();
    }

    /// `[...providers].sort(...)` - the input array is copied, never reordered in place.
    fn sort_providers(
        &self,
        _previous: &[AuthSelectorProvider],
        providers: &[AuthSelectorProvider],
    ) -> Vec<AuthSelectorProvider> {
        let mut sorted = providers.to_vec();
        sorted.sort_by(|a, b| {
            let rank_delta =
                self.get_provider_sort_rank(a) as i32 - self.get_provider_sort_rank(b) as i32;
            if rank_delta != 0 {
                return rank_delta.cmp(&0);
            }
            if self.mode == "login" && a.id != b.id {
                if a.id == PRIME_INFERENCE_PROVIDER_ID {
                    return std::cmp::Ordering::Less;
                }
                if b.id == PRIME_INFERENCE_PROVIDER_ID {
                    return std::cmp::Ordering::Greater;
                }
            }
            compare_auth_selector_providers(a, b)
        });
        sorted
    }

    /// Port of `refresh`.
    pub fn refresh(&mut self) {
        let selected = self.filtered_providers.get(self.selected_index).cloned();
        let providers = std::mem::take(&mut self.all_providers);
        self.all_providers = self.sort_providers(&[], &providers);
        let query = self.search_input.borrow().get_value().to_string();
        self.filter_providers(&query);
        if let Some(selected) = selected {
            if let Some(index) = self.filtered_providers.iter().position(|provider| {
                provider.id == selected.id && provider.auth_type == selected.auth_type
            }) {
                self.selected_index = index;
            }
        }
    }

    fn get_provider_sort_rank(&self, provider: &AuthSelectorProvider) -> i32 {
        if self.is_provider_configured(provider) {
            return 0;
        }
        if self.is_provider_stale(provider) {
            return 1;
        }
        2
    }

    fn is_provider_stale(&self, provider: &AuthSelectorProvider) -> bool {
        let status = self.status(&provider.id);
        let credential = self.credential(provider);
        let storage_status = self.storage_status(provider);
        status.source.as_deref() == Some("stale")
            || (storage_status.source.as_deref() == Some("stale")
                && credential
                    .as_ref()
                    .map(|credential| credential_type(credential) == provider.auth_type)
                    .unwrap_or(false))
    }

    fn is_provider_configured(&self, provider: &AuthSelectorProvider) -> bool {
        let status = self.status(&provider.id);
        if status.source.as_deref() == Some("claude_code") {
            return status.configured && provider.auth_type == "oauth";
        }
        let credential = self.credential(provider);
        if self.is_provider_stale(provider) {
            return false;
        }

        if let Some(source) = status.source.as_deref() {
            if source != "stored" {
                return provider.auth_type == "api_key";
            }
        }

        if credential.is_some() {
            return true;
        }
        if provider.auth_type != "api_key" {
            return false;
        }
        status.source.is_some()
    }

    fn credential(&self, provider: &AuthSelectorProvider) -> Option<AuthCredential> {
        self.auth_storage.get(&provider.id)
    }

    fn storage_status(&self, provider: &AuthSelectorProvider) -> AuthStatus {
        self.auth_storage.get_auth_status(&provider.id)
    }

    /// Port of `getAuthStatus(providerId)`: the injected callback wins, else the
    /// storage is read directly (`this.getAuthStatus ?? ((id) => this.authStorage.getAuthStatus(id))`).
    fn status(&self, provider_id: &str) -> AuthStatus {
        match &self.get_auth_status {
            Some(callback) => callback(provider_id),
            None => self.auth_storage.get_auth_status(provider_id),
        }
    }

    /// Port of `handleInput`.
    pub fn handle_input(&mut self, key_data: &str) {
        let kb = pi_tui::keybindings::get_keybindings();
        if kb.matches(key_data, "tui.select.up") {
            if self.filtered_providers.is_empty() {
                return;
            }
            self.selected_index = self.selected_index.saturating_sub(1);
            self.update_list();
        } else if kb.matches(key_data, "tui.select.down") {
            if self.filtered_providers.is_empty() {
                return;
            }
            self.selected_index = (self.selected_index + 1).min(self.filtered_providers.len() - 1);
            self.update_list();
        }
        // Only steal left/right for tabs when the search field is empty, so cursor
        // editing still works while filtering.
        else if self.categories.len() > 1
            && self.search_input.borrow().get_value().is_empty()
            && kb.matches(key_data, "tui.editor.cursorLeft")
        {
            self.switch_category(-1);
        } else if self.categories.len() > 1
            && self.search_input.borrow().get_value().is_empty()
            && kb.matches(key_data, "tui.editor.cursorRight")
        {
            self.switch_category(1);
        } else if kb.matches(key_data, "tui.select.confirm") {
            if let Some(selected) = self.filtered_providers.get(self.selected_index).cloned() {
                self.selected_provider = Some(selected);
            }
        } else if kb.matches(key_data, "tui.select.cancel") {
            self.cancelled = true;
        } else {
            self.search_input.borrow_mut().handle_input(key_data);
            let value = self.search_input.borrow().get_value().to_string();
            self.filter_providers(&value);
        }
    }

    fn reserved_rows(&self) -> f64 {
        PROVIDER_LIST_RESERVED_ROWS as f64
            + (self.get_header_rows)()
            + if self.has_tab_bar {
                TAB_BAR_RESERVED_ROWS
            } else {
                0.0
            }
    }

    /// Port of `updateLayout`.
    pub fn update_layout(&mut self) {
        self.list_layout = get_menu_list_layout(MenuListLayoutOptions {
            get_rows: self.viewport.get_rows.clone(),
            preferred_visible_items: PREFERRED_VISIBLE_PROVIDERS,
            total_items: Some(self.filtered_providers.len()),
            reserved_rows: self.reserved_rows() as usize,
            comfortable_item_rows: 3,
            compact_item_rows: Some(2),
            scroll_indicator_rows: Some(PROVIDER_SCROLL_INDICATOR_ROWS),
            ..Default::default()
        });
    }

    /// Port of `updateList`'s visible window.
    pub fn update_list(&mut self) {
        self.update_layout();
        let max_visible = self.list_layout.visible_items;
        let start_index = (self.selected_index as i64 - (max_visible / 2) as i64)
            .min(self.filtered_providers.len() as i64 - max_visible as i64)
            .max(0) as usize;
        self.visible_range = (
            start_index,
            (start_index + max_visible).min(self.filtered_providers.len()),
        );
    }

    /// Port of `formatStatusIndicator`.
    pub fn format_status_indicator(&self, provider: &AuthSelectorProvider) -> String {
        let status = self.status(&provider.id);
        if status.source.as_deref() == Some("claude_code") {
            return theme().fg(if status.configured { "success" } else { "warning" },
                status.label.as_deref().unwrap_or("Claude Code OAuth"));
        }
        if self.is_provider_stale(provider) {
            return theme().fg("warning", status.label.as_deref().unwrap_or("expired"));
        }

        if let Some(source) = status.source.as_deref() {
            if source != "stored" {
                return if provider.auth_type == "api_key" {
                    format_api_key_status_indicator(&status)
                } else {
                    theme().fg("muted", "unconfigured")
                };
            }
        }

        if let Some(credential) = self.credential(provider) {
            if credential_type(&credential) == provider.auth_type {
                return theme().fg("success", "configured");
            }
            let label = if credential_type(&credential) == "oauth" {
                "subscription configured"
            } else {
                "API key configured"
            };
            return theme().fg("warning", label);
        }
        if provider.auth_type != "api_key" {
            return theme().fg("muted", "unconfigured");
        }

        format_api_key_status_indicator(&status)
    }

    /// Port of `getSearchInput()`.
    pub fn search_input(&self) -> Rc<RefCell<MenuSearchInput>> {
        self.search_input.clone()
    }
}

impl Component for OAuthSelectorComponent {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.update_list();
        let mut panel = MenuPanel::new(MenuPanelOptions {
            title: self.title.clone(),
            subtitle: Some(self.subtitle.clone()),
        });
        if let Some(header) = &self.header {
            panel.add_child(header.clone());
            panel.add_child(Rc::new(RefCell::new(Spacer::new(1))));
        }
        if self.has_tab_bar {
            panel.add_child(Rc::new(RefCell::new(TruncatedText::new(
                self.update_tab_bar(),
                0,
                0,
            ))));
            panel.add_child(Rc::new(RefCell::new(Spacer::new(1))));
        }
        panel.add_full_width_child(self.search_input.clone());
        panel.add_child(Rc::new(RefCell::new(Spacer::new(1))));
        let compact = self.list_layout.compact;
        let mut list = MenuList::new(Some(Box::new(move || compact)));
        let (start, end) = self.visible_range;
        for (index, provider) in self
            .filtered_providers
            .iter()
            .enumerate()
            .take(end)
            .skip(start)
        {
            list.add_row(Rc::new(RefCell::new(MenuRow::new(MenuRowOptions {
                primary: provider.name.clone(),
                secondary: Some(
                    if provider.auth_type == "oauth" {
                        "subscription"
                    } else {
                        "api key"
                    }
                    .into(),
                ),
                meta: Some(self.format_status_indicator(provider)),
                selected: index == self.selected_index,
            }))));
        }
        if start > 0 || end < self.filtered_providers.len() {
            list.add_child(
                Rc::new(RefCell::new(TruncatedText::new(
                    theme().fg(
                        "muted",
                        &format!(
                            "  ({}/{})",
                            self.selected_index + 1,
                            self.filtered_providers.len()
                        ),
                    ),
                    1,
                    0,
                ))),
                false,
            );
        }
        if self.filtered_providers.is_empty() {
            let text = if self.all_providers.is_empty() {
                if self.mode == "login" {
                    "No providers available"
                } else {
                    "No providers logged in. Use /login first."
                }
            } else {
                "No matching providers"
            };
            list.add_child(
                Rc::new(RefCell::new(TruncatedText::new(
                    theme().fg("muted", text),
                    1,
                    0,
                ))),
                false,
            );
        }
        panel.add_full_width_child(Rc::new(RefCell::new(list)));
        panel.render(width)
    }
    fn handle_input(&mut self, data: &str) {
        OAuthSelectorComponent::handle_input(self, data);
    }
    fn invalidate(&mut self) {
        self.search_input.borrow_mut().invalidate();
    }
    fn as_focusable(&mut self) -> Option<&mut dyn Focusable> {
        Some(self)
    }
}
impl Focusable for OAuthSelectorComponent {
    fn focused(&self) -> bool {
        self.search_input.borrow().focused()
    }
    fn set_focused(&mut self, focused: bool) {
        self.search_input.borrow_mut().set_focused(focused);
    }
}

/// `AuthCredential["type"]`.
pub fn credential_type(credential: &AuthCredential) -> String {
    match credential {
        AuthCredential::ApiKey { .. } => "api_key".to_string(),
        AuthCredential::OAuth { .. } => "oauth".to_string(),
    }
}

/// Port of `formatApiKeyStatusIndicator`.
pub fn format_api_key_status_indicator(status: &AuthStatus) -> String {
    match status.source.as_deref() {
        Some("environment") => theme().fg(
            "success",
            &format!("env: {}", status.label.as_deref().unwrap_or("API key")),
        ),
        Some("prime_cli") => theme().fg("success", status.label.as_deref().unwrap_or("Prime CLI")),
        Some("runtime") => theme().fg("success", "runtime API key"),
        Some("fallback") => theme().fg("success", "custom API key"),
        Some("models_json_key") => theme().fg("success", "key in models.json"),
        Some("models_json_command") => theme().fg("success", "command in models.json"),
        _ => theme().fg("muted", "unconfigured"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct StubStorage {
        statuses: std::collections::HashMap<String, AuthStatus>,
        credentials: std::collections::HashMap<String, AuthCredential>,
    }

    impl AuthStorageLike for StubStorage {
        fn get(&self, provider: &str) -> Option<AuthCredential> {
            self.credentials.get(provider).cloned()
        }

        fn get_auth_status(&self, provider: &str) -> AuthStatus {
            self.statuses.get(provider).cloned().unwrap_or_default()
        }
    }

    fn provider(id: &str, name: &str, auth_type: &str) -> AuthSelectorProvider {
        AuthSelectorProvider {
            id: id.to_string(),
            name: name.to_string(),
            auth_type: auth_type.to_string(),
            category: None,
        }
    }

    fn storage() -> StubStorage {
        StubStorage {
            statuses: std::collections::HashMap::new(),
            credentials: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn oauth_providers_sort_before_api_key_providers() {
        let mut providers = vec![provider("b", "B", "api_key"), provider("a", "A", "oauth")];
        providers.sort_by(compare_auth_selector_providers);
        assert_eq!(providers[0].auth_type, "oauth");
    }

    #[test]
    fn prime_inference_ranks_first_in_login_mode() {
        let storage = storage();
        let component = OAuthSelectorComponent::new(
            "login",
            Box::new(storage),
            vec![
                provider("anthropic", "Anthropic", "oauth"),
                provider(PRIME_INFERENCE_PROVIDER_ID, "Prime", "oauth"),
            ],
            None,
            OAuthSelectorOptions::default(),
        );
        assert_eq!(component.all_providers[0].id, PRIME_INFERENCE_PROVIDER_ID);
    }

    #[test]
    fn tab_bar_appears_only_with_more_than_one_category() {
        let storage = storage();
        let single = OAuthSelectorComponent::new(
            "login",
            Box::new(storage.clone()),
            vec![provider("a", "A", "oauth")],
            None,
            OAuthSelectorOptions::default(),
        );
        assert!(!single.has_tab_bar);

        let service = AuthSelectorProvider {
            category: Some(AuthSelectorCategory::Service),
            ..provider("b", "B", "api_key")
        };
        let two = OAuthSelectorComponent::new(
            "login",
            Box::new(storage),
            vec![provider("a", "A", "oauth"), service],
            None,
            OAuthSelectorOptions::default(),
        );
        assert!(two.has_tab_bar);
        assert_eq!(two.active_category, AuthSelectorCategory::Provider);
    }

    #[test]
    fn switching_categories_wraps_and_resets_the_selection() {
        let storage = storage();
        let service = AuthSelectorProvider {
            category: Some(AuthSelectorCategory::Service),
            ..provider("b", "B", "api_key")
        };
        let mut component = OAuthSelectorComponent::new(
            "login",
            Box::new(storage),
            vec![provider("a", "A", "oauth"), service],
            None,
            OAuthSelectorOptions::default(),
        );
        component.selected_index = 0;
        component.switch_category(1);
        assert_eq!(component.active_category, AuthSelectorCategory::Service);
        assert_eq!(component.selected_index, 0);
        component.switch_category(1);
        assert_eq!(component.active_category, AuthSelectorCategory::Provider);
    }

    #[test]
    fn filtering_is_scoped_to_the_active_category() {
        let storage = storage();
        let service = AuthSelectorProvider {
            category: Some(AuthSelectorCategory::Service),
            ..provider("mcp", "MCP", "api_key")
        };
        let mut component = OAuthSelectorComponent::new(
            "login",
            Box::new(storage),
            vec![provider("anthropic", "Anthropic", "oauth"), service],
            None,
            OAuthSelectorOptions::default(),
        );
        assert_eq!(component.filtered_providers.len(), 1);
        component.switch_category(1);
        assert_eq!(component.filtered_providers[0].id, "mcp");
    }

    #[test]
    fn api_key_status_indicator_labels_every_source() {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        let status = AuthStatus {
            configured: true,
            source: Some("environment".to_string()),
            label: Some("MY_KEY".to_string()),
        };
        assert!(format_api_key_status_indicator(&status).contains("env: MY_KEY"));
        let unconfigured = AuthStatus::default();
        assert!(format_api_key_status_indicator(&unconfigured).contains("unconfigured"));
    }

    #[test]
    fn claude_code_is_shown_as_the_active_subscription() {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        let mut storage = storage();
        storage.statuses.insert("anthropic".into(), AuthStatus {
            configured: true, source: Some("claude_code".into()), label: Some("Claude Code 2.1.283 OAuth".into()),
        });
        let oauth = provider("anthropic", "Claude", "oauth");
        let api_key = provider("anthropic", "Anthropic API", "api_key");
        let component = OAuthSelectorComponent::new("login", Box::new(storage),
            vec![api_key.clone(), oauth.clone()], None, OAuthSelectorOptions::default());
        assert!(component.is_provider_configured(&oauth));
        assert!(!component.is_provider_configured(&api_key));
        assert!(component.format_status_indicator(&oauth).contains("Claude Code 2.1.283 OAuth"));
    }
}
