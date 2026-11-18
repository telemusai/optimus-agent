//! Native configuration-menu mounting with the persisted provider registry.
use super::*;
use crate::core::{auth_storage::AuthStorage, model_registry::ModelRegistry};
use crate::modes::interactive::components::configuration_menu::{
    AuthSelectorProvider, ConfigurationMenuComponent, ConfigurationMenuOptions,
    ConfigurationMenuScopedModel,
};

/// The verdict the model-selection path needs for one provider.
///
/// Port of `ensureModelProviderConfigured` (interactive-mode.ts:8015-8040)
/// together with `isModelProviderConfigured` (interactive-mode.ts:8042-8044).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ModelProviderReadiness {
    /// `isModelProviderConfigured(model)`: the daemon already reports the
    /// provider as configured, or the model registry holds auth for it
    /// (`this.connectionConfiguredProviders.has(model.provider) ||
    /// this.modelRegistry.hasConfiguredAuth(model)`, interactive-mode.ts:8043).
    Configured,
    /// A provider-category login option exists, so `authFlows.loginProvider`
    /// can run (`interactive-mode.ts:8033-8035`).
    Loginable(AuthSelectorProvider),
    /// No provider-category login option exists: `Authentication for <id> must
    /// be configured externally.` (`interactive-mode.ts:8027-8029`).
    ExternallyConfigured,
}

/// Decides what the model-selection path does for `provider`.
///
/// The lookup is the TypeScript one verbatim
/// (`providerOptions.find((option) => option.id === model.provider &&
/// (option.category ?? "provider") === "provider")`, interactive-mode.ts:8022-8025):
/// it consults the SAME login-option list the Providers tab renders, so a
/// provider offered for API-key login is always accepted here. The Rust host
/// used to test `built_in_provider_display_names()` instead, which refused every
/// custom API-key provider that `provider_options` had just offered.
pub(super) fn model_provider_readiness(
    provider_options: &[AuthSelectorProvider],
    provider: &str,
    daemon_configured: bool,
    registry_has_auth: bool,
) -> ModelProviderReadiness {
    if daemon_configured || registry_has_auth {
        return ModelProviderReadiness::Configured;
    }
    provider_options
        .iter()
        .find(|option| {
            option.id == provider && option.category.as_deref().unwrap_or("provider") == "provider"
        })
        .cloned()
        .map(ModelProviderReadiness::Loginable)
        .unwrap_or(ModelProviderReadiness::ExternallyConfigured)
}

/// The action the model-selection loop takes for one selected model.
///
/// Port of `ensureModelProviderConfigured` + `completeModelSelection`
/// (interactive-mode.ts:8015-8040, :7990-7995).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ModelSelectionAction {
    /// `await this.completeModelSelection(model)` (interactive-mode.ts:8436).
    Switch,
    /// `await authFlows.loginProvider(provider)` (interactive-mode.ts:8033-8035);
    /// `oauth` picks the subscription dialog over the API-key dialog
    /// (auth-flows.ts:173-185).
    BeginLogin { oauth: bool },
    /// `showError("Authentication for <provider> must be configured externally.")`
    /// (interactive-mode.ts:8027-8029).
    ExternallyConfigured,
}

/// Resolves the loop action for a selected model.
pub(super) fn model_selection_action(
    provider_options: &[AuthSelectorProvider],
    provider: &str,
    daemon_configured: bool,
    registry_has_auth: bool,
) -> ModelSelectionAction {
    match model_provider_readiness(
        provider_options,
        provider,
        daemon_configured,
        registry_has_auth,
    ) {
        ModelProviderReadiness::Configured => ModelSelectionAction::Switch,
        ModelProviderReadiness::Loginable(option) => ModelSelectionAction::BeginLogin {
            oauth: option.auth_type == "oauth",
        },
        ModelProviderReadiness::ExternallyConfigured => ModelSelectionAction::ExternallyConfigured,
    }
}

pub(super) fn create(
    mode: &InteractiveMode,
    ui: Rc<RefCell<TUI>>,
    tab: &'static str,
    models: &[wire::AgentConnectionModel],
    configured: &std::collections::HashSet<String>,
    search: Option<String>,
    rows: Rc<Cell<f64>>,
    selection: mpsc::Sender<Option<String>>,
    send: HostEventSender,
) -> Result<ConfigurationMenuComponent, String> {
    let auth = Rc::new(RefCell::new(AuthStorage::create(None, None)));
    let registry = Rc::new(RefCell::new(ModelRegistry::create(
        AuthStorage::create(None, None),
        None,
    )));
    let providers = provider_options(&auth.borrow(), &registry.borrow(), models);
    let selected_provider = selection.clone();
    let selected_service = selection.clone();
    let selected_model = selection.clone();
    let settings = mode.settings_manager().lock().map_err(|e| e.to_string())?;
    Ok(ConfigurationMenuComponent::new(ConfigurationMenuOptions {
        initial_tab: tab,
        tui: ui,
        auth_storage: auth,
        model_registry: registry,
        provider_options: providers,
        current_model: mode.get_current_model().cloned(),
        scoped_models: mode
            .get_scoped_model_state()
            .into_iter()
            .map(|s| ConfigurationMenuScopedModel {
                model: s.model,
                thinking_level: None,
            })
            .collect(),
        available_models: models.to_vec(),
        configured_providers: configured.clone(),
        recent_models: Some(settings.get_recent_models()),
        initial_model_search: search,
        get_rows: Some(Box::new(move || rows.get())),
        request_render: Box::new(move || {
            let _ = send.send(HostEvent::Render);
        }),
        on_select_provider: Box::new(move |provider| {
            let _ = selected_provider.send(Some(login_key(provider)));
        }),
        on_select_mcp_connection: Box::new(move |provider| {
            let _ = selected_service.send(Some(login_key(provider)));
        }),
        on_select_model: Box::new(move |model| {
            let _ = selected_model.send(Some(format!("{}/{}", model.provider, model.id)));
        }),
        on_cancel: Box::new(move || {
            let _ = selection.send(None);
        }),
    }))
}

fn login_key(provider: &AuthSelectorProvider) -> String {
    format!(
        "login:{}:{}",
        if provider.auth_type == "oauth" {
            "oauth"
        } else {
            "api"
        },
        provider.id
    )
}

/// The inputs `getLoginProviderOptions` (auth-flows.ts:233-268) reads.
///
/// Both the Providers tab and the model-selection eligibility check build their
/// list through [`login_provider_options`], so the two paths cannot drift - the
/// TypeScript shares one `authFlows.getLoginProviderOptions()` call between
/// `showConfigurationMenu` (interactive-mode.ts:8327-8329) and
/// `handleModelCommand` (interactive-mode.ts:7950-7953).
pub(super) struct LoginProviderSources<'a> {
    /// `this.host.modelRegistry.authStorage`.
    pub auth: &'a AuthStorage,
    /// `new Set(this.host.modelRegistry.getAll().map((model) => model.provider))`
    /// (auth-flows.ts:243), unioned with the models the daemon catalog shows.
    pub registered_models: &'a [wire::AgentConnectionModel],
    /// `this.host.modelRegistry.getProviderDisplayName(providerId)`
    /// (auth-flows.ts:250).
    pub display_name: &'a dyn Fn(&str) -> String,
}

/// Port of `getLoginProviderOptions` (auth-flows.ts:233-268).
pub(super) fn login_provider_options(
    sources: LoginProviderSources<'_>,
) -> Vec<AuthSelectorProvider> {
    let oauth = sources.auth.get_oauth_providers();
    let ids: std::collections::HashSet<String> = oauth.iter().map(|p| p.id.clone()).collect();
    let mut options: Vec<_> = oauth
        .into_iter()
        .map(|p| AuthSelectorProvider {
            // MCP integrations (mcp:<server>) are services, not model providers.
            category: p.id.starts_with("mcp:").then(|| "service".into()),
            id: p.id,
            name: p.name,
            auth_type: "oauth".into(),
        })
        .collect();
    let providers: std::collections::BTreeSet<_> = sources
        .registered_models
        .iter()
        .map(|m| m.provider.clone())
        .collect();
    for provider in providers {
        // `if (!isApiKeyLoginProvider(providerId, oauthProviderIds)) continue;`
        // (auth-flows.ts:248-250). Custom (non-built-in, non-OAuth) providers
        // pass this test, which is exactly what defect 3 dropped.
        if crate::modes::interactive::auth_flows::is_api_key_login_provider(&provider, &ids, None) {
            options.push(AuthSelectorProvider {
                name: (sources.display_name)(&provider),
                id: provider,
                auth_type: "api_key".into(),
                category: None,
            });
        }
    }
    // Serper is a skill credential, not a model provider, so add it manually
    // (auth-flows.ts:256-261).
    options.push(AuthSelectorProvider {
        id: crate::core::websearch_credential::SERPER_CREDENTIAL_ID.into(),
        name: crate::core::websearch_credential::SERPER_CREDENTIAL_NAME.into(),
        auth_type: "api_key".into(),
        category: Some("service".into()),
    });
    options
}

/// The menu surface `apply_post_login_catalog` drives.
///
/// The TypeScript calls these four methods on the open menu after a successful
/// login (`interactive-mode.ts:8384-8400`): `menu.refreshAuthentication()`,
/// `menu.updateModels(current, candidates, configuredProviders)`,
/// `menu.setActiveTab("models")`, and the overlay `focus()`. The trait keeps the
/// production call site and its test on the same path.
pub(super) trait LoginCatalogMenu {
    /// `menu.refreshAuthentication()` (interactive-mode.ts:8381).
    fn refresh_authentication(&mut self);
    /// `menu.updateModels(...)` (interactive-mode.ts:8396-8400).
    fn update_models(
        &mut self,
        current_model: Option<&wire::AgentConnectionModel>,
        models: &[wire::AgentConnectionModel],
        configured_providers: &std::collections::HashSet<String>,
    );
    /// `menu.setActiveTab("models")` (interactive-mode.ts:8401).
    fn set_active_tab_to_models(&mut self);
}

impl LoginCatalogMenu for ConfigurationMenuComponent {
    fn refresh_authentication(&mut self) {
        ConfigurationMenuComponent::refresh_authentication(self);
    }
    fn update_models(
        &mut self,
        current_model: Option<&wire::AgentConnectionModel>,
        models: &[wire::AgentConnectionModel],
        configured_providers: &std::collections::HashSet<String>,
    ) {
        ConfigurationMenuComponent::update_models(
            self,
            current_model,
            Some(models),
            Some(configured_providers),
        );
    }
    fn set_active_tab_to_models(&mut self) {
        self.set_active_tab("models");
    }
}

/// The state the post-login refresh replaces.
///
/// These are the outer `models`/`configured_providers` the host's
/// model-selection loop reads (`interactive-mode.ts:8042-8044` reads
/// `this.connectionModelCatalog` / `this.connectionConfiguredProviders`).
#[derive(Debug)]
pub(super) struct PostLoginCatalog {
    pub models: Vec<wire::AgentConnectionModel>,
    pub configured_providers: std::collections::HashSet<String>,
}

/// The catalog fetch the post-login refresh performs.
///
/// `invalidateConnectionModels(); await this.getConnectionAvailableModels()`
/// through `onAuthChanged` (interactive-mode.ts:8123-8126, :8855-8858), which
/// reaches `this.agentConnection.getModelCatalog()` (interactive-mode.ts:8066-8068).
/// The trait exists so the flow can be driven by a recording source in tests
/// while production passes the live connection.
pub(super) trait LoginCatalogSource {
    fn get_model_catalog(
        &self,
    ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionModelCatalog, String>>;
}

impl<T: wire::AgentConnection + ?Sized> LoginCatalogSource for T {
    fn get_model_catalog(
        &self,
    ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionModelCatalog, String>> {
        wire::AgentConnection::get_model_catalog(self)
    }
}

/// Refreshes the daemon catalog and applies it to the host and the open menu.
///
/// Port of the post-login ordering in `authenticate`
/// (interactive-mode.ts:8378-8402): the catalog is re-fetched first, then
/// `menu.refreshAuthentication()`, then `menu.updateModels(current, candidates,
/// connectionConfiguredProviders)`, then `menu.setActiveTab("models")`.
/// `getCachedModelCandidates` (interactive-mode.ts:8085-8094) starts from the
/// scoped models and then overwrites from the connection catalog, so scoped
/// entries are unioned in first.
pub(super) async fn refresh_after_login<M, C>(
    source: &C,
    menu: Option<&Rc<RefCell<M>>>,
    current_model: Option<&wire::AgentConnectionModel>,
    scoped_models: &[wire::AgentConnectionModel],
) -> Result<PostLoginCatalog, String>
where
    M: LoginCatalogMenu,
    C: LoginCatalogSource + ?Sized,
{
    // `menu.refreshAuthentication()` runs BEFORE the model refresh
    // (`handle?.focus(); menu.refreshAuthentication();` at
    // interactive-mode.ts:8379-8381, and again at :8432-8433 after
    // `ensureModelProviderConfigured`), so the Providers tab already shows the
    // saved credential when the model list is replaced.
    if let Some(menu) = menu {
        menu.borrow_mut().refresh_authentication();
    }
    let catalog = source.get_model_catalog().await?;
    Ok(apply_post_login_catalog(
        menu,
        current_model,
        scoped_models,
        catalog,
    ))
}

/// Applies a re-fetched daemon catalog to the host and the open menu.
pub(super) fn apply_post_login_catalog<M: LoginCatalogMenu>(
    menu: Option<&Rc<RefCell<M>>>,
    current_model: Option<&wire::AgentConnectionModel>,
    scoped_models: &[wire::AgentConnectionModel],
    catalog: wire::AgentConnectionModelCatalog,
) -> PostLoginCatalog {
    let mut candidates = PostLoginCatalog {
        models: Vec::new(),
        configured_providers: catalog.configured_providers.into_iter().collect(),
    };
    let mut seen = std::collections::HashSet::new();
    for model in scoped_models.iter().chain(catalog.models.iter()) {
        if seen.insert((model.provider.clone(), model.id.clone())) {
            candidates.models.push(model.clone());
        }
    }
    if let Some(menu) = menu {
        let mut menu = menu.borrow_mut();
        menu.update_models(
            current_model,
            &candidates.models,
            &candidates.configured_providers,
        );
        menu.set_active_tab_to_models();
    }
    candidates
}

/// The login-option list for a model catalog, from a fresh registry.
///
/// The Providers tab (`create`) and the model-selection eligibility check both
/// call this, so the two paths read the identical list - the TypeScript shares
/// one `authFlows.getLoginProviderOptions()` result between
/// `showConfigurationMenu` (interactive-mode.ts:8328) and `handleModelCommand`
/// (interactive-mode.ts:7952).
pub(super) fn login_options_for(
    models: &[wire::AgentConnectionModel],
) -> Vec<AuthSelectorProvider> {
    let auth = AuthStorage::create(None, None);
    let registry = ModelRegistry::create(AuthStorage::create(None, None), None);
    provider_options(&auth, &registry, models)
}

/// The list the Providers tab renders, from the menu's own registry.
fn provider_options(
    auth: &AuthStorage,
    registry: &ModelRegistry,
    models: &[wire::AgentConnectionModel],
) -> Vec<AuthSelectorProvider> {
    let registered: Vec<wire::AgentConnectionModel> =
        models.iter().cloned().chain(registry.get_all()).collect();
    let display = |provider: &str| registry.get_provider_display_name(provider);
    login_provider_options(LoginProviderSources {
        auth,
        registered_models: &registered,
        display_name: &display,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Runs one post-login refresh inside a local runtime, the same way the host
    /// loop awaits it.
    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
            .block_on(future)
    }

    fn model(provider: &str, id: &str) -> wire::AgentConnectionModel {
        pi_ai::types::Model::new(id, id, "openai-completions", provider, "http://127.0.0.1")
    }

    /// The provider entry the Providers tab offers for a custom provider is the
    /// same entry the model-selection path accepts.
    #[test]
    fn a_custom_api_key_provider_is_loginable_in_both_paths() {
        let models = vec![model("my-proxy", "proxy-model")];
        let options = login_options_for(&models);
        assert!(
            options
                .iter()
                .any(|option| option.id == "my-proxy" && option.auth_type == "api_key"),
            "the Providers tab must offer the custom provider: {options:?}"
        );
        assert_eq!(
            model_selection_action(&options, "my-proxy", false, false),
            ModelSelectionAction::BeginLogin { oauth: false },
            "the model-selection path must accept the provider it just offered"
        );
        // The old check only queried `built_in_provider_display_names()`.
        assert!(
            !crate::core::provider_display_names::built_in_provider_display_names()
                .contains_key("my-proxy"),
            "the custom provider is not a built-in display name, so the old              check refused it even though provider_options offered it"
        );
    }

    /// A built-in provider keeps API-key login; the display-name shortcut and
    /// the shared list must agree.
    #[test]
    fn a_built_in_provider_stays_loginable() {
        let models = vec![model("deepseek", "deepseek-chat")];
        let options = login_options_for(&models);
        assert_eq!(
            model_selection_action(&options, "deepseek", false, false),
            ModelSelectionAction::BeginLogin { oauth: false }
        );
    }

    /// An OAuth-only provider routes to the subscription dialog, not the API-key one.
    #[test]
    fn an_oauth_provider_routes_to_the_subscription_dialog() {
        let options = vec![AuthSelectorProvider {
            id: "openai-codex".into(),
            name: "OpenAI Codex".into(),
            auth_type: "oauth".into(),
            category: None,
        }];
        assert_eq!(
            model_selection_action(&options, "openai-codex", false, false),
            ModelSelectionAction::BeginLogin { oauth: true }
        );
    }

    /// A provider with no login option is the externally-configured case
    /// (interactive-mode.ts:8027-8029), not a silent no-op.
    #[test]
    fn a_provider_without_a_login_option_reports_external_configuration() {
        let options = vec![AuthSelectorProvider {
            id: "some-other-provider".into(),
            name: "Some Other Provider".into(),
            auth_type: "api_key".into(),
            category: None,
        }];
        assert_eq!(
            model_selection_action(&options, "mystery-gateway", false, false),
            ModelSelectionAction::ExternallyConfigured,
            "a provider absent from the login options needs external credentials"
        );
        assert_eq!(
            model_selection_action(&options, "mystery-gateway", false, true),
            ModelSelectionAction::Switch,
            "registry-held auth still wins"
        );
    }

    /// The live list agrees with the registry: Bedrock is genuinely offered for
    /// API-key login because it has a built-in display name
    /// (`BUILT_IN_PROVIDER_DISPLAY_NAMES`, provider-display-names.ts:3) and
    /// `authFlows.loginProvider` sends it to the AWS setup dialog
    /// (auth-flows.ts:181-183).
    #[test]
    fn the_live_login_list_offers_every_built_in_provider() {
        let options = login_options_for(&[model("my-proxy", "proxy-model")]);
        for provider in ["amazon-bedrock", "deepseek", "openai", "my-proxy"] {
            assert!(
                options.iter().any(|option| option.id == provider
                    && option.category.as_deref().unwrap_or("provider") == "provider"),
                "the login options must hold a provider entry for {provider}"
            );
        }
    }

    /// A configured provider switches without any auth dialog
    /// (`isModelProviderConfigured`, interactive-mode.ts:8042-8044).
    #[test]
    fn a_configured_provider_switches_without_a_prompt() {
        let options = login_options_for(&[model("my-proxy", "proxy-model")]);
        assert_eq!(
            model_selection_action(&options, "my-proxy", true, false),
            ModelSelectionAction::Switch,
            "daemon-reported configuration wins"
        );
        assert_eq!(
            model_selection_action(&options, "my-proxy", false, true),
            ModelSelectionAction::Switch,
            "registry-held auth wins"
        );
    }

    /// Only provider-category entries can be logged in: the Serper service entry
    /// must never satisfy a model provider (interactive-mode.ts:8022-8025).
    #[test]
    fn a_service_entry_never_satisfies_a_model_provider() {
        let models = vec![model(
            crate::core::websearch_credential::SERPER_CREDENTIAL_ID,
            "serper-model",
        )];
        let options = login_options_for(&models);
        assert!(
            options.iter().any(|option| option.id
                == crate::core::websearch_credential::SERPER_CREDENTIAL_ID
                && option.category.as_deref() == Some("service")),
            "precondition: Serper is listed as a service"
        );
        assert_eq!(
            model_selection_action(
                &options,
                crate::core::websearch_credential::SERPER_CREDENTIAL_ID,
                false,
                false
            ),
            ModelSelectionAction::BeginLogin { oauth: false },
            "the provider-category entry wins over the service entry of the same id"
        );
    }

    /// Records the menu calls the post-login refresh performs, in order.
    struct RecordingMenu {
        calls: Vec<String>,
    }

    impl LoginCatalogMenu for RecordingMenu {
        fn refresh_authentication(&mut self) {
            self.calls.push("refreshAuthentication".into());
        }
        fn update_models(
            &mut self,
            current_model: Option<&wire::AgentConnectionModel>,
            models: &[wire::AgentConnectionModel],
            configured_providers: &HashSet<String>,
        ) {
            let current = current_model
                .map(|model| format!("{}/{}", model.provider, model.id))
                .unwrap_or_else(|| "none".into());
            let mut providers: Vec<_> = configured_providers.iter().cloned().collect();
            providers.sort();
            self.calls.push(format!(
                "updateModels(current={current},models={},configured={})",
                models.len(),
                providers.join("|")
            ));
        }
        fn set_active_tab_to_models(&mut self) {
            self.calls.push("setActiveTab(models)".into());
        }
    }

    /// Records the catalog fetch, and how many times it happened.
    struct RecordingSource {
        catalog: Result<wire::AgentConnectionModelCatalog, String>,
        calls: std::cell::RefCell<Vec<String>>,
    }

    impl LoginCatalogSource for RecordingSource {
        fn get_model_catalog(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionModelCatalog, String>> {
            self.calls.borrow_mut().push("getModelCatalog".into());
            let catalog = self.catalog.clone();
            Box::pin(async move { catalog })
        }
    }

    /// DEFECT 2: after a successful login the flow re-fetches the daemon catalog
    /// and hands it to the menu, so the newly authenticated provider is visible
    /// to the selection loop that runs next.
    ///
    /// `invalidateConnectionModels(); await this.getConnectionAvailableModels()`
    /// (interactive-mode.ts:8123-8126) via `onAuthChanged` (interactive-mode.ts:8855-8858),
    /// then `menu.refreshAuthentication(); menu.updateModels(...);
    /// menu.setActiveTab("models")` (interactive-mode.ts:8381-8401).
    #[test]
    fn a_successful_login_refreshes_the_catalog_and_the_menu_in_order() {
        let source = RecordingSource {
            catalog: Ok(wire::AgentConnectionModelCatalog {
                models: vec![model("my-proxy", "proxy-model")],
                configured_providers: vec!["my-proxy".into()],
            }),
            calls: std::cell::RefCell::new(Vec::new()),
        };
        let menu = Rc::new(RefCell::new(RecordingMenu { calls: Vec::new() }));
        let current = model("openai", "gpt-5.6");
        let refreshed = block_on(refresh_after_login(
            &source,
            Some(&menu),
            Some(&current),
            &[],
        ))
        .expect("catalog");

        assert_eq!(
            source.calls.borrow().as_slice(),
            ["getModelCatalog"],
            "the daemon catalog must be re-fetched exactly once"
        );
        assert_eq!(
            menu.borrow().calls.as_slice(),
            [
                "refreshAuthentication",
                "updateModels(current=openai/gpt-5.6,models=1,configured=my-proxy)",
                "setActiveTab(models)",
            ],
            "the TS ordering is refreshAuthentication then updateModels then setActiveTab"
        );
        // `refreshAuthentication()` precedes the catalog fetch
        // (interactive-mode.ts:8379-8381), so the provider list is refreshed even
        // when the daemon call fails.
        let source_calls = source.calls.borrow();
        assert_eq!(source_calls.as_slice(), ["getModelCatalog"]);
        // The outer state the selection loop reads.
        assert_eq!(refreshed.models.len(), 1);
        assert!(refreshed.configured_providers.contains("my-proxy"));
    }

    /// DEFECT 2 end to end: the refreshed catalog makes the selection loop treat
    /// the new provider as configured, so selecting the new model does NOT
    /// re-prompt for login. Before the fix the outer state stayed stale and the
    /// second selection took the login branch.
    #[test]
    fn selecting_the_new_model_after_login_does_not_re_prompt() {
        let source = RecordingSource {
            catalog: Ok(wire::AgentConnectionModelCatalog {
                models: vec![model("my-proxy", "proxy-model")],
                configured_providers: vec!["my-proxy".into()],
            }),
            calls: std::cell::RefCell::new(Vec::new()),
        };
        let refreshed = block_on(refresh_after_login::<RecordingMenu, RecordingSource>(
            &source,
            None,
            None,
            &[],
        ))
        .expect("catalog");
        let options = login_options_for(&refreshed.models);

        assert_eq!(
            model_selection_action(
                &options,
                "my-proxy",
                refreshed.configured_providers.contains("my-proxy"),
                false,
            ),
            ModelSelectionAction::Switch,
            "the selection loop must switch, not re-prompt for login"
        );
        // The stale outer state (the pre-fix bug) still prompts.
        assert_eq!(
            model_selection_action(&options, "my-proxy", false, false),
            ModelSelectionAction::BeginLogin { oauth: false },
            "precondition: with the stale outer state the flow prompts again"
        );
    }

    /// Scoped models stay candidates: `getCachedModelCandidates` starts from
    /// `getScopedModelState()` (interactive-mode.ts:8085-8094).
    #[test]
    fn post_login_candidates_union_the_scoped_models() {
        let menu = Rc::new(RefCell::new(RecordingMenu { calls: Vec::new() }));
        let refreshed = apply_post_login_catalog(
            Some(&menu),
            None,
            &[
                model("scoped-only", "scoped-1"),
                model("my-proxy", "proxy-model"),
            ],
            wire::AgentConnectionModelCatalog {
                models: vec![model("my-proxy", "proxy-model"), model("other", "other-1")],
                configured_providers: vec!["my-proxy".into()],
            },
        );
        let ids: Vec<_> = refreshed
            .models
            .iter()
            .map(|model| format!("{}/{}", model.provider, model.id))
            .collect();
        assert_eq!(
            ids,
            [
                "scoped-only/scoped-1",
                "my-proxy/proxy-model",
                "other/other-1"
            ],
            "scoped first, then the catalog, deduplicated on provider/id"
        );
    }

    /// A failed catalog fetch still fails loudly, and the menu holds no false
    /// model claim. Only `refreshAuthentication` ran, because it is ordered
    /// before the fetch (interactive-mode.ts:8379-8381).
    #[test]
    fn a_failed_catalog_refresh_reports_the_error() {
        let source = RecordingSource {
            catalog: Err("daemon unavailable".into()),
            calls: std::cell::RefCell::new(Vec::new()),
        };
        let menu = Rc::new(RefCell::new(RecordingMenu { calls: Vec::new() }));
        let error = block_on(refresh_after_login::<RecordingMenu, RecordingSource>(
            &source,
            Some(&menu),
            None,
            &[],
        ))
        .expect_err("must fail");
        assert_eq!(error, "daemon unavailable");
        assert_eq!(
            menu.borrow().calls.as_slice(),
            ["refreshAuthentication"],
            "a failed refresh must not claim the model list was updated"
        );
    }

    /// DEFECT 3 at the production call site: the host's selection loop must use
    /// the shared list, never the built-in display-name table.
    #[test]
    fn the_host_selection_path_has_no_display_name_gate() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/modes/interactive/native_host.rs"
        ))
        .expect("native_host.rs");
        let region: Vec<&str> = source
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.contains("//"))
            .map(|(_, line)| line)
            .collect();
        let selection = region
            .iter()
            .position(|line| line.contains("model_selection_action"))
            .expect("the selection path calls model_selection_action");
        let window = &region[selection.saturating_sub(40)..=selection];
        assert!(
            !window
                .iter()
                .any(|line| line.contains("built_in_provider_display_names")),
            "the model-selection path must not gate on built-in display names"
        );
    }
}
