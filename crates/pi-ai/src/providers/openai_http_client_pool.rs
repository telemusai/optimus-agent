//! Transport-only clients. Request credentials and affinity never enter this pool.
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

const MAX_CLIENTS: usize = 16;
const MAX_CLIENT_AGE: Duration = Duration::from_secs(300);
const POLICY_ENV: [&str; 9] = [
    "ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
    "NO_PROXY", "no_proxy", "REQUEST_METHOD",
];
const TLS_ENV: [&str; 4] = ["SSL_CERT_FILE", "SSL_CERT_DIR", "OPENSSL_CONF", "OPENSSL_MODULES"];

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum RedirectPolicy { Generation, Count }

#[derive(Clone, PartialEq, Eq)]
struct Key {
    runtime: tokio::runtime::Id,
    origin: [u8; 32],
    environment: [u8; 32],
    redirect: RedirectPolicy,
}

struct Entry { key: Key, created: Instant, client: reqwest::Client }
#[derive(Default)]
struct Pool { entries: VecDeque<Entry> }

#[derive(PartialEq, Eq)]
struct Environment { digest: [u8; 32], fallback: Option<&'static str> }

fn environment() -> Environment { environment_from(|name| std::env::var_os(name)) }

fn environment_from(read: impl Fn(&str) -> Option<std::ffi::OsString>) -> Environment {
    static SALT: OnceLock<[u8; 32]> = OnceLock::new();
    let mut hash = Sha256::new();
    hash.update(SALT.get_or_init(rand::random));
    for name in POLICY_ENV.into_iter().chain(TLS_ENV) {
        hash.update(name.as_bytes());
        match read(name) {
            Some(value) => {
                let bytes = value.as_encoded_bytes();
                hash.update([1]);
                hash.update((bytes.len() as u64).to_le_bytes());
                hash.update(bytes);
            }
            None => hash.update([0]),
        }
    }
    let nonempty = |upper, lower| read(upper).and_then(|v| v.into_string().ok())
        .or_else(|| read(lower).and_then(|v| v.into_string().ok()))
        .map(|v| !v.is_empty()).unwrap_or(false);
    // reqwest's default system matcher can fill empty HTTP/HTTPS/NO_PROXY
    // from Windows/macOS settings. Do not freeze unobservable mutable policy.
    let os_proxy_unobserved = cfg!(any(windows, target_os = "macos"))
        && read("REQUEST_METHOD").is_none()
        && !(nonempty("HTTP_PROXY", "http_proxy") && nonempty("HTTPS_PROXY", "https_proxy")
            && nonempty("NO_PROXY", "no_proxy"));
    // Native TLS may load mutable files. Keep its existing per-request behavior
    // instead of caching trust-file contents or weakening verification.
    let tls_files = TLS_ENV.iter().any(|name| read(name).is_some());
    let first = |upper, lower| read(upper).and_then(|v| v.into_string().ok())
        .or_else(|| read(lower).and_then(|v| v.into_string().ok()));
    // A domain wildcard does not bypass numeric IP hosts in hyper-util. Do not
    // infer TLS bypass from NO_PROXY; an HTTPS proxy always uses fresh policy.
    let proxy_may_use_tls = read("REQUEST_METHOD").is_none()
        && [first("HTTP_PROXY", "http_proxy"), first("ALL_PROXY", "all_proxy")]
            .into_iter().flatten().any(|value| value.to_ascii_lowercase().starts_with("https://"));
    Environment {
        digest: hash.finalize().into(),
        fallback: if tls_files { Some("fresh_tls_policy") }
            else if os_proxy_unobserved { Some("fresh_system_proxy_policy") }
            else if proxy_may_use_tls { Some("fresh_proxy_tls_policy") } else { None },
    }
}

fn build(redirect: RedirectPolicy) -> Result<reqwest::Client, reqwest::Error> {
    let mut builder = reqwest::Client::builder();
    if redirect == RedirectPolicy::Count {
        builder = builder.redirect(reqwest::redirect::Policy::none()).pool_max_idle_per_host(4);
    }
    // TLS verification and the default proxy/redirect rules are unchanged.
    builder.build()
}

impl Pool {
    fn take(&mut self, key: &Key, now: Instant) -> Option<reqwest::Client> {
        self.entries.retain(|entry| now.duration_since(entry.created) < MAX_CLIENT_AGE);
        let index = self.entries.iter().position(|entry| &entry.key == key)?;
        let entry = self.entries.remove(index)?;
        let client = entry.client.clone();
        self.entries.push_back(entry);
        Some(client)
    }

    fn insert(&mut self, key: Key, client: reqwest::Client, now: Instant) -> reqwest::Client {
        if let Some(existing) = self.take(&key, now) { return existing; }
        while self.entries.len() >= MAX_CLIENTS { self.entries.pop_front(); }
        self.entries.push_back(Entry { key, created: now, client: client.clone() });
        client
    }
}

pub(super) struct Lease {
    pub client: reqwest::Client,
    pub phase: &'static str,
}

pub(super) fn acquire(base_url: &str, redirect: RedirectPolicy) -> Result<Lease, reqwest::Error> {
    static POOL: OnceLock<Mutex<Pool>> = OnceLock::new();
    // Generation follows redirects, which can reach TLS even from HTTP. Native
    // trust stores are not observable here; keep current-default construction.
    if redirect == RedirectPolicy::Generation {
        return Ok(Lease { client: build(redirect)?, phase: "fresh_generation_policy" });
    }
    let url = reqwest::Url::parse(base_url).ok();
    if url.as_ref().is_some_and(|url| url.scheme() == "https") {
        return Ok(Lease { client: build(redirect)?, phase: "fresh_tls_policy" });
    }
    let policy = environment();
    let origin = url.filter(|url| url.scheme() == "http" && url.host_str().is_some())
        .map(|url| Sha256::digest(url.origin().ascii_serialization().as_bytes()).into());
    let runtime = tokio::runtime::Handle::try_current().ok().map(|handle| handle.id());
    let fallback = policy.fallback.or(if origin.is_none() || runtime.is_none() { Some("fresh_unknown_policy") } else { None });
    if let Some(phase) = fallback { return Ok(Lease { client: build(redirect)?, phase }); }
    let key = Key { runtime: runtime.unwrap(), origin: origin.unwrap(), environment: policy.digest, redirect };
    let pool = POOL.get_or_init(|| Mutex::new(Pool::default()));
    if let Some(client) = pool.lock().unwrap_or_else(|p| p.into_inner()).take(&key, Instant::now()) {
        if environment() == policy { return Ok(Lease { client, phase: "http_client_reused" }); }
        return Ok(Lease { client: build(redirect)?, phase: "fresh_changed_policy" });
    }
    // Client construction is outside the pool lock. Cloned handles keep active
    // requests alive across eviction; only idle ownership is bounded here.
    let client = build(redirect)?;
    if environment() != policy { return Ok(Lease { client, phase: "fresh_changed_policy" }); }
    let client = pool.lock().unwrap_or_else(|p| p.into_inner()).insert(key, client, Instant::now());
    Ok(Lease { client, phase: "http_client_created" })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bounded_eviction_age_runtime_and_redirect_keys() {
        let runtime = tokio::runtime::Handle::current().id();
        let key = |n| Key { runtime, origin: [n; 32], environment: [0; 32], redirect: RedirectPolicy::Generation };
        let mut pool = Pool::default();
        let now = Instant::now();
        let active = build(RedirectPolicy::Generation).unwrap();
        for n in 0..=MAX_CLIENTS { pool.insert(key(n as u8), active.clone(), now); }
        assert_eq!(pool.entries.len(), MAX_CLIENTS);
        assert!(pool.take(&key(0), now).is_none());
        assert!(pool.take(&key(1), now).is_some());
        let mut count = key(1); count.redirect = RedirectPolicy::Count;
        assert!(pool.take(&count, now).is_none());
        let other = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let mut different_runtime = key(1); different_runtime.runtime = other.handle().id();
        assert!(pool.take(&different_runtime, now).is_none());
        other.shutdown_background();
        assert!(pool.take(&key(1), now + MAX_CLIENT_AGE).is_none());
        assert!(pool.entries.is_empty());
        // The independently held client remains usable after pool eviction.
        assert!(active.get("http://127.0.0.1:1").build().is_ok());
    }

    #[test]
    fn environment_rotation_is_not_stored_as_plaintext_and_tls_files_fall_back() {
        let mut values = std::collections::BTreeMap::<&str, std::ffi::OsString>::new();
        let initial = environment_from(|name| values.get(name).cloned());
        values.insert("HTTP_PROXY", "http://fixture-user:fixture-secret@127.0.0.1:1".into());
        values.insert("HTTPS_PROXY", "http://fixture-user:fixture-secret@127.0.0.1:1".into());
        values.insert("NO_PROXY", "*".into());
        let first = environment_from(|name| values.get(name).cloned());
        assert!(first.digest != initial.digest);
        assert!(first.fallback.is_none());
        values.insert("HTTP_PROXY", "http://rotated-user:rotated-secret@127.0.0.1:1".into());
        assert!(environment_from(|name| values.get(name).cloned()).digest != first.digest);
        values.insert("SSL_CERT_FILE", "nonexistent-fixture.pem".into());
        assert_eq!(environment_from(|name| values.get(name).cloned()).fallback, Some("fresh_tls_policy"));
        values.remove("SSL_CERT_FILE");
        values.remove("HTTP_PROXY");
        if cfg!(any(windows, target_os = "macos")) {
            assert_eq!(environment_from(|name| values.get(name).cloned()).fallback, Some("fresh_system_proxy_policy"));
        }
        values.insert("HTTP_PROXY", "https://fixture-proxy.invalid".into());
        values.insert("NO_PROXY", "127.0.0.1".into());
        assert_eq!(environment_from(|name| values.get(name).cloned()).fallback, Some("fresh_proxy_tls_policy"));
        values.insert("NO_PROXY", "*".into());
        assert_eq!(environment_from(|name| values.get(name).cloned()).fallback, Some("fresh_proxy_tls_policy"), "domain wildcard cannot prove TLS bypass for IP hosts");
        values.insert("REQUEST_METHOD", "POST".into());
        assert!(environment_from(|name| values.get(name).cloned()).fallback.is_none());
    }

}
