//! Transport-only clients. Request credentials and affinity never enter this pool.
//! Pooling keys include a fingerprint of every frozen-matcher input reqwest's
//! default client reads: the proxy/TLS environment plus, on Windows, the
//! registry values hyper-util's system matcher consults.
use std::collections::VecDeque;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
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

/// native-tls re-reads system trust per handshake on schannel (per-connection
/// `CertGetCertificateChain` against the default chain engine, schannel
/// `tls_stream.rs`) and on security-framework (per-handshake trust evaluation,
/// `security_framework.rs`), while the OpenSSL backend probes and freezes its
/// trust at connector build (`imp/openssl.rs` `TlsConnector::new`). Only the
/// per-handshake platforms may pool TLS-bearing reuse.
const PER_HANDSHAKE_TRUST: bool = cfg!(any(windows, target_os = "macos"));

/// What this process can observe about the OS proxy store that reqwest's
/// default matcher consults beyond the environment (hyper-util
/// `client-proxy-system`): the registry on Windows, the SystemConfiguration
/// store on macOS. The store feeds every pooled client's frozen matcher, so
/// reuse needs a fingerprint of its consultable values or a fresh client.
#[derive(Clone, PartialEq, Eq, Debug)]
enum OsProxyStore {
    /// The default matcher consults no OS store (Linux): env is everything.
    /// Only tests construct it on Windows/macOS, where an OS store is real.
    #[cfg_attr(any(windows, target_os = "macos"), allow(dead_code))]
    None,
    /// The store exists but its values are not observable here (macOS
    /// SystemConfiguration; a Windows registry key that failed to open).
    Unobserved,
    /// The Windows registry store, carrying exactly the values hyper-util
    /// `win::with_system` reads, presence included.
    Windows(WindowsProxySettings),
}

/// The registry values under HKCU `Software\Microsoft\Windows\
/// CurrentVersion\Internet Settings` that hyper-util 0.1.20 `win::with_system`
/// consults: `ProxyEnable` gates the whole contribution, `ProxyServer` fills
/// empty env http/https slots, `ProxyOverride` fills an empty env NO_PROXY
/// slot after its `;`-join and `*.`-strip transform.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
struct WindowsProxySettings {
    enable: u32,
    server: Option<String>,
    r#override: Option<String>,
}

/// Read the store hyper-util would read, without writing anything. An
/// unopenable key is `Unobserved` (a fresh client could still read it later,
/// so the store is not provably empty), not `None`.
#[cfg(windows)]
fn observe_os_proxy() -> OsProxyStore {
    let Ok(settings) = windows_registry::CURRENT_USER
        .open("Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings")
    else {
        return OsProxyStore::Unobserved;
    };
    OsProxyStore::Windows(WindowsProxySettings {
        enable: settings.get_u32("ProxyEnable").unwrap_or(0),
        server: settings.get_string("ProxyServer").ok(),
        r#override: settings.get_string("ProxyOverride").ok(),
    })
}

#[cfg(not(windows))]
fn observe_os_proxy() -> OsProxyStore {
    if cfg!(target_os = "macos") { OsProxyStore::Unobserved } else { OsProxyStore::None }
}

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

#[derive(Debug)]
struct Environment { digest: [u8; 32], fallback: Option<&'static str>, no_proxy: Option<String> }

// `no_proxy` is a decision input for loopback coverage, not an identity field:
// its value is the env string or the registry override, both hashed into
// `digest`, so equality stays rotation-sensitive.
impl PartialEq for Environment {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest && self.fallback == other.fallback
    }
}
impl Eq for Environment {}

fn environment_from(
    read: impl Fn(&str) -> Option<std::ffi::OsString>, os: &OsProxyStore,
) -> Environment {
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
    // The registry store feeds the same frozen matcher. Hash exactly what
    // hyper-util `win::with_system` consults: ProxyEnable always (its early
    // return), then ProxyServer/ProxyOverride only while enabled (the only
    // state in which they can fill a matcher slot).
    if let OsProxyStore::Windows(settings) = os {
        hash.update(b"windows:ProxyEnable");
        hash.update(settings.enable.to_le_bytes());
        if settings.enable != 0 {
            for (name, value) in [
                ("windows:ProxyServer", &settings.server),
                ("windows:ProxyOverride", &settings.r#override),
            ] {
                hash.update(name);
                match value {
                    Some(value) => {
                        let bytes = value.as_bytes();
                        hash.update([1]);
                        hash.update((bytes.len() as u64).to_le_bytes());
                        hash.update(bytes);
                    }
                    None => hash.update([0]),
                }
            }
        }
    }
    let nonempty = |upper, lower| read(upper).and_then(|v| v.into_string().ok())
        .or_else(|| read(lower).and_then(|v| v.into_string().ok()))
        .map(|v| !v.is_empty()).unwrap_or(false);
    // reqwest's default system matcher can fill empty HTTP/HTTPS/NO_PROXY from
    // the OS store. The store is either fingerprinted above (Windows registry)
    // or provably unable to contribute: CGI mode empties the whole matcher, and
    // a fully covering environment leaves no empty slot to fill.
    let store_unobserved = match os {
        OsProxyStore::Unobserved => true,
        OsProxyStore::None | OsProxyStore::Windows(_) => false,
    };
    let os_proxy_unobserved = store_unobserved
        && read("REQUEST_METHOD").is_none()
        && !(nonempty("HTTP_PROXY", "http_proxy") && nonempty("HTTPS_PROXY", "https_proxy")
            && nonempty("NO_PROXY", "no_proxy"));
    // Native TLS may load mutable files. Keep its existing per-request behavior
    // instead of caching trust-file contents or weakening verification.
    let tls_files = TLS_ENV.iter().any(|name| read(name).is_some());
    let first = |upper, lower| read(upper).and_then(|v| v.into_string().ok())
        .or_else(|| read(lower).and_then(|v| v.into_string().ok()));
    // hyper-util fills an empty env NO_PROXY slot from the registry override
    // while ProxyEnable is set, after its `;`-join and `*.`-strip transform
    // (`win::with_system`); replicate the gate and the transform exactly so
    // coverage decisions track the frozen matcher.
    let no_proxy = first("NO_PROXY", "no_proxy").filter(|value| !value.is_empty()).or_else(|| {
        match os {
            OsProxyStore::Windows(settings) if settings.enable != 0 => settings.r#override.as_ref()
                .map(|value| {
                    value.split(';').map(str::trim).collect::<Vec<&str>>().join(",")
                        .replace("*.", "")
                }),
            _ => None,
        }
    });
    // The registry server fills only empty env http/https slots while enabled;
    // ALL_PROXY fills only what is still empty at build time.
    let http_slot = first("HTTP_PROXY", "http_proxy").filter(|value| !value.is_empty());
    let https_slot = first("HTTPS_PROXY", "https_proxy").filter(|value| !value.is_empty());
    let all_slot = first("ALL_PROXY", "all_proxy").filter(|value| !value.is_empty());
    let registry_server = match os {
        OsProxyStore::Windows(settings) if settings.enable != 0 => settings.server.clone(),
        _ => None,
    };
    let registry_fills_http = http_slot.is_none();
    let registry_fills_https = https_slot.is_none();
    let http_proxy = http_slot
        .or_else(|| registry_server.clone().filter(|_| registry_fills_http))
        .or(all_slot.clone());
    let https_proxy = https_slot
        .or_else(|| registry_server.filter(|_| registry_fills_https))
        .or(all_slot);
    // A domain wildcard does not bypass numeric IP hosts in hyper-util. Do not
    // infer TLS bypass from NO_PROXY; an HTTPS proxy always uses fresh policy.
    let proxy_may_use_tls = read("REQUEST_METHOD").is_none()
        && [http_proxy, https_proxy].into_iter().flatten()
            .any(|value| value.to_ascii_lowercase().starts_with("https://"));
    Environment {
        digest: hash.finalize().into(),
        no_proxy,
        fallback: if tls_files { Some("fresh_tls_policy") }
            else if os_proxy_unobserved { Some("fresh_system_proxy_policy") }
            else if proxy_may_use_tls { Some("fresh_proxy_tls_policy") } else { None },
    }
}

fn build(redirect: RedirectPolicy) -> Result<reqwest::Client, reqwest::Error> {
    let mut builder = reqwest::Client::builder();
    if redirect == RedirectPolicy::Count {
        // 32 matches the campaign's highest measured agent concurrency: hyper-util only
        // drops idle connections at return-time, so a cap below concurrent streams forces
        // reconnect churn in burst-gap patterns (measured: 340 accepts vs 32 at conc 32).
        // The guarded pool itself stays bounded at MAX_CLIENTS; idle sockets cap at 32.
        builder = builder.redirect(reqwest::redirect::Policy::none()).pool_max_idle_per_host(32);
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

// ---------------------------------------------------------------------------
// hyper-util NO_PROXY coverage (the frozen-matcher bypass decision)
// ---------------------------------------------------------------------------

/// hyper-util's `NoProxy::contains` strips the RFC 3986 brackets of raw IPv6
/// hosts before classifying; `reqwest::Url::host_str` keeps them.
fn unbracketed(host: &str) -> &str {
    host.trim_matches(|c| c == '[' || c == ']')
}

/// The local-origin hosts this pool treats as loopback. Exact literals only
/// (the url crate lowercases special-scheme hosts), so the decision never
/// depends on resolver behavior.
fn is_loopback_host(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "[::1]" | "localhost")
}

enum Net { V4 { addr: Ipv4Addr, prefix: u8 }, V6 { addr: Ipv6Addr, prefix: u8 } }

impl Net {
    /// ipnet compares against the truncated network and broadcast addresses, so
    /// a non-canonical entry like `127.0.0.1/8` still covers the whole /8.
    fn contains(&self, ip: IpAddr) -> bool {
        match (self, ip) {
            (Net::V4 { addr, prefix }, IpAddr::V4(ip)) => {
                let mask = u32::MAX.checked_shl(32 - u32::from(*prefix)).unwrap_or(0);
                let network = u32::from(*addr) & mask;
                let ip = u32::from(ip);
                network <= ip && ip <= network | !mask
            }
            (Net::V6 { addr, prefix }, IpAddr::V6(ip)) => {
                let mask = u128::MAX.checked_shl(128 - u32::from(*prefix)).unwrap_or(0);
                let network = u128::from_be_bytes(addr.octets()) & mask;
                let ip = u128::from_be_bytes(ip.octets());
                network <= ip && ip <= network | !mask
            }
            _ => false,
        }
    }
}

/// Replicates ipnet 2.12.2's prefix parsing (`Parser::read_number`): decimal
/// digits only, at most two digits for IPv4 and three for IPv6, value below
/// 33 and 129 respectively. Longer or out-of-range prefixes make the whole
/// entry a domain, exactly like `IpNet::from_str`.
fn parse_cidr(left: &str, prefix: &str) -> Option<Net> {
    let prefix = |max_digits: usize, below: u8| -> Option<u8> {
        if prefix.len() > max_digits || !prefix.bytes().all(|b| b.is_ascii_digit()) { return None; }
        let value: u8 = prefix.parse().ok()?;
        (value < below).then_some(value)
    };
    if let Ok(addr) = left.parse::<Ipv4Addr>() {
        return prefix(2, 33).map(|prefix| Net::V4 { addr, prefix });
    }
    if let Ok(addr) = left.parse::<Ipv6Addr>() {
        return prefix(3, 129).map(|prefix| Net::V6 { addr, prefix });
    }
    None
}

enum NoProxyEntry { Net(Net), Addr(IpAddr), Domain(String) }

/// Classifies one comma-separated NO_PROXY entry exactly like hyper-util 0.1.20
/// `NoProxy::from_string` (client/proxy/matcher.rs): `IpNet` first, then
/// `IpAddr`, else domain. One deliberate divergence: an IPv4 address with
/// leading zeros parses in ipnet's lenient CIDR parser but not with std, so
/// such entries count as domains here. That can only make this pool MORE
/// conservative, never less.
fn classify_entry(entry: &str) -> NoProxyEntry {
    if let Some((left, prefix)) = entry.split_once('/') {
        // A '/' entry never parses as a bare IP address.
        return match parse_cidr(left, prefix) {
            Some(net) => NoProxyEntry::Net(net),
            None => NoProxyEntry::Domain(entry.to_string()),
        };
    }
    match entry.parse::<IpAddr>() {
        Ok(addr) => NoProxyEntry::Addr(addr),
        Err(_) => NoProxyEntry::Domain(entry.to_string()),
    }
}

/// Replicates hyper-util 0.1.20 `NoProxy::contains` for `host` (the string
/// `reqwest::Url::host_str` reports) against a NO_PROXY list value. Coverage
/// decides whether a pooled client's frozen proxy matcher provably bypasses
/// this host, so this must never report coverage hyper-util would not: an
/// IP host consults only the IP matcher and a domain host only the domain
/// matcher (matcher.rs `contains`).
fn no_proxy_covers(host: &str, no_proxy: &str) -> bool {
    let host = unbracketed(host);
    match host.parse::<IpAddr>() {
        Ok(ip) => no_proxy.split(',').map(str::trim).any(|entry| match classify_entry(entry) {
            NoProxyEntry::Addr(addr) => addr == ip,
            NoProxyEntry::Net(net) => net.contains(ip),
            NoProxyEntry::Domain(_) => false,
        }),
        Err(_) => no_proxy.split(',').map(str::trim).any(|entry| match classify_entry(entry) {
            NoProxyEntry::Domain(domain) => domain_covers(host, &domain),
            _ => false,
        }),
    }
}

/// hyper-util `DomainMatcher::contains`, including the suffix dot-boundary rule
/// and the whole-entry-only `*` wildcard.
fn domain_covers(host: &str, domain: &str) -> bool {
    if domain.eq_ignore_ascii_case(host) { return true; }
    if domain.strip_prefix('.').is_some_and(|stripped| stripped.eq_ignore_ascii_case(host)) {
        return true;
    }
    let suffix_matches = host.get(host.len().saturating_sub(domain.len())..)
        .is_some_and(|suffix| suffix.eq_ignore_ascii_case(domain));
    if suffix_matches {
        if domain.starts_with('.') { return true; }
        return host.len().checked_sub(domain.len() + 1)
            .and_then(|start| host.as_bytes().get(start))
            .is_some_and(|byte| *byte == b'.');
    }
    domain == "*"
}

// ---------------------------------------------------------------------------
// Acquisition
// ---------------------------------------------------------------------------

/// Whether this acquisition may consult the pool, from the origin facts and
/// the environment snapshot. Pure so every branch is unit-testable.
///
/// Security contract: a pooled `reqwest::Client` freezes its proxy matcher at
/// build (reqwest `async_impl/client.rs` `ClientBuilder::build` ->
/// `ProxyMatcher::system()`; hyper-util `matcher.rs` `from_system`). Reuse is
/// allowed only when every input of that frozen matcher is either hashed into
/// the environment digest (a fallback-free, fully observed environment) or
/// provably not consulted for this request (a NO_PROXY that covers the host is
/// checked before any proxy match in `Matcher::intercept`).
fn poolable(
    redirect: RedirectPolicy, scheme: Option<&str>, loopback: bool,
    covered: bool, fallback: Option<&'static str>,
) -> bool {
    let fully_observed = fallback.is_none();
    let tls_env = fallback == Some("fresh_tls_policy");
    match (redirect, scheme) {
        // Redirectless counting to an https origin: only the frozen-matcher
        // concern applies, and only per-handshake-trust platforms clear it.
        (RedirectPolicy::Count, Some("https")) => PER_HANDSHAKE_TRUST && fully_observed,
        // Redirectless plain http. Loopback origins may also rely on NO_PROXY
        // coverage, which bypasses the frozen matcher before any proxy or TLS
        // is consulted; the TLS-env guard is never overridden.
        (RedirectPolicy::Count, Some("http")) => {
            fully_observed || (loopback && covered && !tls_env)
        }
        // Generation follows redirects that can leave the covered host for TLS
        // or proxied origins, so it needs a fully observed environment (every
        // frozen matcher input digest-tracked) plus coverage plus per-handshake
        // trust for the redirect hops that may reach TLS.
        (RedirectPolicy::Generation, Some("http")) if loopback => {
            PER_HANDSHAKE_TRUST && fully_observed && covered
        }
        // Redirect-following generation over TLS: the environment digest now
        // pins every frozen-matcher input (env vars plus, on Windows, the
        // registry values hyper-util consults), and per-handshake trust keeps
        // schannel/security-framework trust roots current, so a pooled client
        // observes exactly the policy a fresh build would.
        (RedirectPolicy::Generation, Some("https")) => PER_HANDSHAKE_TRUST && fully_observed,
        _ => false,
    }
}

/// Pool phases for a poolable acquisition. Only called when `poolable` is true.
fn phases(redirect: RedirectPolicy, scheme: Option<&str>, loopback: bool) -> (&'static str, &'static str) {
    match (redirect, scheme, loopback) {
        (RedirectPolicy::Count, Some("https"), _) => ("http_client_reused_https", "http_client_created_https"),
        (RedirectPolicy::Generation, Some("https"), _) => {
            ("http_client_reused_generation_https", "http_client_created_generation_https")
        }
        (RedirectPolicy::Generation, Some("http"), true) => {
            ("http_client_reused_generation_loopback", "http_client_created_generation_loopback")
        }
        (RedirectPolicy::Count, Some("http"), true) => ("http_client_reused_loopback", "http_client_created_loopback"),
        _ => ("http_client_reused", "http_client_created"),
    }
}

/// Fresh-phase name for a non-pooled acquisition.
fn fresh_phase(
    redirect: RedirectPolicy, scheme: Option<&str>, origin_known: bool,
    fallback: Option<&'static str>,
) -> &'static str {
    if redirect == RedirectPolicy::Generation {
        match fallback {
            Some("fresh_tls_policy") => "fresh_generation_tls_policy",
            Some("fresh_system_proxy_policy") => "fresh_generation_system_proxy_policy",
            Some("fresh_proxy_tls_policy") => "fresh_generation_proxy_tls_policy",
            // Fully observed but still fresh: no per-handshake-trust platform,
            // or an origin that never pooled.
            _ => "fresh_generation_policy",
        }
    } else if scheme == Some("https") { "fresh_tls_policy" }
    else if !origin_known { "fresh_unknown_policy" }
    else { fallback.unwrap_or("fresh_unknown_policy") }
}

pub(super) struct Lease {
    pub client: reqwest::Client,
    pub phase: &'static str,
}

pub(super) fn acquire(base_url: &str, redirect: RedirectPolicy) -> Result<Lease, reqwest::Error> {
    acquire_with(&|name| std::env::var_os(name), &observe_os_proxy, base_url, redirect)
}

fn acquire_with(
    read: &dyn Fn(&str) -> Option<std::ffi::OsString>,
    os_read: &dyn Fn() -> OsProxyStore,
    base_url: &str,
    redirect: RedirectPolicy,
) -> Result<Lease, reqwest::Error> {
    static POOL: OnceLock<Mutex<Pool>> = OnceLock::new();
    let url = reqwest::Url::parse(base_url).ok();
    let scheme = url.as_ref().map(|url| url.scheme());
    let host = url.as_ref().and_then(|url| url.host_str());
    let runtime = tokio::runtime::Handle::try_current().ok().map(|handle| handle.id());
    let policy = environment_from(read, &os_read());
    let origin = url.as_ref()
        .filter(|url| matches!(url.scheme(), "http" | "https") && url.host_str().is_some())
        .map(|url| Sha256::digest(url.origin().ascii_serialization().as_bytes()).into());
    let loopback = scheme == Some("http") && host.is_some_and(is_loopback_host);
    let covered = host.is_some_and(|host| {
        policy.no_proxy.as_deref().is_some_and(|value| no_proxy_covers(host, value))
    });
    let known = origin.is_some() && runtime.is_some();
    let (reused, created) = if poolable(redirect, scheme, loopback, covered, policy.fallback) && known {
        phases(redirect, scheme, loopback)
    } else {
        let phase = fresh_phase(redirect, scheme, origin.is_some(), policy.fallback);
        return Ok(Lease { client: build(redirect)?, phase });
    };
    let key = Key { runtime: runtime.unwrap(), origin: origin.unwrap(), environment: policy.digest, redirect };
    let pool = POOL.get_or_init(|| Mutex::new(Pool::default()));
    if let Some(client) = pool.lock().unwrap_or_else(|p| p.into_inner()).take(&key, Instant::now()) {
        if environment_from(read, &os_read()) == policy { return Ok(Lease { client, phase: reused }); }
        return Ok(Lease { client: build(redirect)?, phase: "fresh_changed_policy" });
    }
    // Client construction is outside the pool lock. Cloned handles keep active
    // requests alive across eviction; only idle ownership is bounded here.
    let client = build(redirect)?;
    if environment_from(read, &os_read()) != policy { return Ok(Lease { client, phase: "fresh_changed_policy" }); }
    let client = pool.lock().unwrap_or_else(|p| p.into_inner()).insert(key, client, Instant::now());
    Ok(Lease { client, phase: created })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn reader(pairs: &[(&'static str, &'static str)]) -> impl Fn(&str) -> Option<std::ffi::OsString> {
        let pairs: Vec<(&'static str, std::ffi::OsString)> =
            pairs.iter().map(|(name, value)| (*name, (*value).into())).collect();
        move |name| pairs.iter().find(|(key, _)| *key == name).map(|(_, value)| value.clone())
    }

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
        let initial = environment_from(|name| values.get(name).cloned(), &OsProxyStore::None);
        assert_eq!(initial.no_proxy, None);
        values.insert("HTTP_PROXY", "http://fixture-user:fixture-secret@127.0.0.1:1".into());
        values.insert("HTTPS_PROXY", "http://fixture-user:fixture-secret@127.0.0.1:1".into());
        values.insert("NO_PROXY", "*".into());
        let first = environment_from(|name| values.get(name).cloned(), &OsProxyStore::None);
        assert!(first.digest != initial.digest);
        assert!(first.fallback.is_none());
        assert_eq!(first.no_proxy.as_deref(), Some("*"));
        values.insert("HTTP_PROXY", "http://rotated-user:rotated-secret@127.0.0.1:1".into());
        assert!(environment_from(|name| values.get(name).cloned(), &OsProxyStore::None).digest != first.digest);
        values.insert("NO_PROXY", "127.0.0.1".into());
        let rotated = environment_from(|name| values.get(name).cloned(), &OsProxyStore::None);
        assert!(rotated.digest != first.digest, "a NO_PROXY rotation must change the digest");
        assert_eq!(rotated.no_proxy.as_deref(), Some("127.0.0.1"));
        values.insert("SSL_CERT_FILE", "nonexistent-fixture.pem".into());
        assert_eq!(environment_from(|name| values.get(name).cloned(), &OsProxyStore::None).fallback, Some("fresh_tls_policy"));
        values.remove("SSL_CERT_FILE");
        values.remove("HTTP_PROXY");
        // An unobservable OS store (macOS SystemConfiguration, or a Windows
        // registry key that failed to open) keeps the fresh fallback.
        assert_eq!(
            environment_from(|name| values.get(name).cloned(), &OsProxyStore::Unobserved).fallback,
            Some("fresh_system_proxy_policy")
        );
        values.insert("HTTP_PROXY", "https://fixture-proxy.invalid".into());
        values.insert("NO_PROXY", "127.0.0.1".into());
        assert_eq!(environment_from(|name| values.get(name).cloned(), &OsProxyStore::None).fallback, Some("fresh_proxy_tls_policy"));
        values.insert("NO_PROXY", "*".into());
        assert_eq!(environment_from(|name| values.get(name).cloned(), &OsProxyStore::None).fallback, Some("fresh_proxy_tls_policy"), "domain wildcard cannot prove TLS bypass for IP hosts");
        values.insert("REQUEST_METHOD", "POST".into());
        assert!(environment_from(|name| values.get(name).cloned(), &OsProxyStore::Unobserved).fallback.is_none());
    }

    #[test]
    fn windows_store_rotation_changes_the_digest_only_while_enabled() {
        let read = reader(&[]);
        let base = WindowsProxySettings {
            enable: 1, server: Some("http://a.invalid:1".into()), r#override: Some("localhost".into()),
        };
        let baseline = environment_from(&read, &OsProxyStore::Windows(base.clone()));
        // Any consultable rotation while enabled changes the digest.
        for rotated in [
            WindowsProxySettings { enable: 1, server: Some("http://b.invalid:2".into()), r#override: Some("localhost".into()) },
            WindowsProxySettings { enable: 1, server: Some("http://a.invalid:1".into()), r#override: Some("127.0.0.1".into()) },
            WindowsProxySettings { enable: 1, server: None, r#override: Some("localhost".into()) },
            WindowsProxySettings { enable: 1, server: Some("http://a.invalid:1".into()), r#override: None },
            WindowsProxySettings { enable: 0, server: Some("http://a.invalid:1".into()), r#override: Some("localhost".into()) },
        ] {
            assert!(
                environment_from(&read, &OsProxyStore::Windows(rotated)).digest != baseline.digest,
                "rotation must invalidate the frozen-matcher fingerprint"
            );
        }
        // While disabled, server/override cannot fill any matcher slot, so
        // rotating them leaves the digest unchanged: exactly the values
        // hyper-util would consult.
        let disabled = WindowsProxySettings::default();
        let disabled_digest = environment_from(&read, &OsProxyStore::Windows(disabled.clone())).digest;
        for ignored in [
            WindowsProxySettings { enable: 0, server: Some("http://b.invalid:2".into()), r#override: None },
            WindowsProxySettings { enable: 0, server: None, r#override: Some("127.0.0.1".into()) },
        ] {
            assert_eq!(environment_from(&read, &OsProxyStore::Windows(ignored)).digest, disabled_digest);
        }
        // Store presence itself never leaks into the digest, only its values.
        assert_eq!(
            environment_from(&read, &OsProxyStore::Unobserved).digest,
            environment_from(&read, &OsProxyStore::None).digest
        );
        // Presence of the store is not plaintext either: equal observations
        // are equal environments.
        assert_eq!(environment_from(&read, &OsProxyStore::Windows(base)), baseline);
    }

    #[test]
    fn no_proxy_coverage_matches_hyper_util_semantics() {
        // Hosts exactly as `reqwest::Url::host_str` reports them.
        for (host, value, expected) in [
            ("127.0.0.1", "127.0.0.1", true),
            ("127.0.0.1", "127.0.0.2", false),
            ("127.0.0.1", "127.0.0.0/8", true),
            ("127.0.0.1", "127.0.0.1/8", true),
            ("127.0.0.1", "127.0.0.1/32", true),
            ("127.0.0.1", "127.0.0.1/33", false),
            ("127.0.0.1", "127.0.0.1/033", false),
            ("127.0.0.1", "0.0.0.0/0", true),
            ("127.0.0.1", "::/0", false),
            ("127.0.0.1", "*", false),
            ("127.0.0.1", "localhost", false),
            ("127.0.0.1", "127.0.0.1,localhost", true),
            ("127.0.0.1", " 127.0.0.1 , localhost ", true),
            ("127.0.0.1", "127.000.0.1", false),
            ("127.0.0.1", "127.000.0.1/8", false),
            ("localhost", "localhost", true),
            ("localhost", "LOCALHOST", true),
            ("localhost", ".localhost", true),
            ("localhost", "sub.localhost", false),
            ("localhost", "ocalhost", false),
            ("localhost", "*.localhost", false),
            ("localhost", "*", true),
            ("localhost", "example.com", false),
            ("localhost", "127.0.0.1", false),
            ("[::1]", "::1", true),
            ("[::1]", "::1/128", true),
            ("[::1]", "::1/129", false),
            ("[::1]", "::1/0128", false),
            ("[::1]", "::/0", true),
            ("[::1]", "0.0.0.0/0", false),
            ("[::1]", "[::1]", false),
            ("[::1]", "localhost", false),
        ] {
            assert_eq!(no_proxy_covers(host, value), expected, "host {host} value {value:?}");
        }
        assert!(!no_proxy_covers("127.0.0.1", ""));
        assert!(!no_proxy_covers("localhost", ""));
    }

    /// End-to-end equivalence against the real frozen matcher: for every gated
    /// loopback host, a reqwest client whose proxy matcher was built from the
    /// same NO_PROXY string (public `Proxy::no_proxy` -> hyper-util
    /// `Builder::no(..).build()`, the identical parse `Matcher::from_system`
    /// uses, consulted per request like the pooled client would) must bypass
    /// the proxy exactly when `no_proxy_covers` reports coverage. The one
    /// documented divergence (ipnet's lenient leading-zero CIDR left side)
    /// is asserted in the safe direction: this pool stays fresh.
    #[tokio::test]
    async fn no_proxy_coverage_matches_a_frozen_reqwest_matcher() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let equivalent: Vec<(&str, &str)> = vec![
            ("127.0.0.1", "127.0.0.1"), ("127.0.0.1", "127.0.0.2"), ("127.0.0.1", "127.0.0.0/8"),
            ("127.0.0.1", "127.0.0.1/8"), ("127.0.0.1", "127.0.0.1/33"), ("127.0.0.1", "127.0.0.1/033"),
            ("127.0.0.1", "0.0.0.0/0"), ("127.0.0.1", "*"), ("127.0.0.1", "localhost"),
            ("127.0.0.1", " 127.0.0.1 , localhost "), ("127.0.0.1", "127.000.0.1"),
            ("localhost", "localhost"), ("localhost", "LOCALHOST"), ("localhost", ".localhost"),
            ("localhost", "sub.localhost"), ("localhost", "ocalhost"), ("localhost", "*"),
            ("localhost", "*.localhost"), ("localhost", "127.0.0.1"),
            ("[::1]", "::1"), ("[::1]", "::1/128"), ("[::1]", "::1/129"), ("[::1]", "::1/0128"),
            ("[::1]", "::/0"), ("[::1]", "0.0.0.0/0"), ("[::1]", "*"), ("[::1]", "[::1]"),
        ];
        let stricter: Vec<(&str, &str)> = vec![("127.0.0.1", "127.000.0.1/8")];
        for (host, value) in equivalent.iter().chain(stricter.iter()) {
            let proxy = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let target_bind = if *host == "[::1]" { "[::1]:0" } else { "127.0.0.1:0" };
            let target = tokio::net::TcpListener::bind(target_bind).await.unwrap();
            let proxy_address = proxy.local_addr().unwrap();
            let target_port = target.local_addr().unwrap().port();
            let intercepted = std::sync::Arc::new(AtomicUsize::new(0));
            let direct = std::sync::Arc::new(AtomicUsize::new(0));
            let hits = intercepted.clone();
            let proxy_task = tokio::spawn(async move {
                loop {
                    let (mut socket, _) = match tokio::time::timeout(Duration::from_secs(1), proxy.accept()).await {
                        Ok(Ok(accepted)) => accepted,
                        _ => continue,
                    };
                    hits.fetch_add(1, Ordering::SeqCst);
                    let _ = tokio::time::timeout(Duration::from_millis(500), socket.read(&mut [0u8; 512])).await;
                }
            });
            let hits = direct.clone();
            let target_task = tokio::spawn(async move {
                loop {
                    let (mut socket, _) = match tokio::time::timeout(Duration::from_secs(1), target.accept()).await {
                        Ok(Ok(accepted)) => accepted,
                        _ => continue,
                    };
                    hits.fetch_add(1, Ordering::SeqCst);
                    let mut buffer = Vec::new();
                    let mut chunk = [0u8; 1024];
                    while !buffer.windows(4).any(|w| w == b"\r\n\r\n") {
                        match tokio::time::timeout(Duration::from_secs(5), socket.read(&mut chunk)).await {
                            Ok(Ok(0)) | Err(_) | Ok(Err(_)) => break,
                            Ok(Ok(n)) => {
                                buffer.extend_from_slice(&chunk[..n]);
                                if buffer.len() > 8192 { break; }
                            }
                        }
                    }
                    let _ = socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok").await;
                }
            });
            // The system matcher fills its `http` entry from env for http targets;
            // an explicit `Proxy::http` with the same NO_PROXY string makes the
            // identical bypass decision for these http:// probes.
            let client = reqwest::Client::builder()
                .no_proxy()
                .proxy(
                    reqwest::Proxy::http(format!("http://{proxy_address}"))
                        .unwrap()
                        .no_proxy(reqwest::NoProxy::from_string(value)),
                )
                .build()
                .unwrap();
            let url = if *host == "[::1]" {
                format!("http://{host}:{target_port}/probe")
            } else {
                format!("http://{host}:{target_port}/probe")
            };
            let _ = tokio::time::timeout(Duration::from_secs(5), client.get(&url).send()).await;
            tokio::time::sleep(Duration::from_millis(50)).await;
            let covered = no_proxy_covers(host, value);
            let bypassed = direct.load(Ordering::SeqCst) > 0;
            let proxied = intercepted.load(Ordering::SeqCst) > 0;
            if stricter.contains(&(*host, *value)) {
                assert!(!covered, "the lenient left side must stay uncovered here");
                assert!(bypassed && !proxied, "ipnet accepts the left side; this pool must only be stricter");
            } else {
                assert_eq!(covered, bypassed, "host {host} value {value:?}");
                assert_eq!(covered, !proxied, "host {host} value {value:?}");
            }
            proxy_task.abort();
            target_task.abort();
        }
    }

    #[tokio::test]
    async fn acquisition_branches_and_phases() {
        let observed = reader(&[
            ("HTTP_PROXY", "http://fixture.invalid:8080"),
            ("HTTPS_PROXY", "http://fixture.invalid:8080"),
            ("NO_PROXY", "127.0.0.1,localhost,::1"),
        ]);
        let no_store = || OsProxyStore::None;
        // Fully observed env, plain-http non-loopback: existing base phases.
        assert_eq!(acquire_with(&observed, &no_store, "http://10.0.0.1:59001/v1", RedirectPolicy::Count).unwrap().phase, "http_client_created");
        assert_eq!(acquire_with(&observed, &no_store, "http://10.0.0.1:59001/v1", RedirectPolicy::Count).unwrap().phase, "http_client_reused");
        // Loopback detection: the three covered hosts pool, remote hosts do not.
        assert_eq!(acquire_with(&observed, &no_store, "http://127.0.0.1:59002/v1", RedirectPolicy::Count).unwrap().phase, "http_client_created_loopback");
        assert_eq!(acquire_with(&observed, &no_store, "http://127.0.0.1:59002/v1", RedirectPolicy::Count).unwrap().phase, "http_client_reused_loopback");
        assert_eq!(acquire_with(&observed, &no_store, "http://localhost:59002/v1", RedirectPolicy::Count).unwrap().phase, "http_client_created_loopback");
        assert_eq!(acquire_with(&observed, &no_store, "http://[::1]:59002/v1", RedirectPolicy::Count).unwrap().phase, "http_client_created_loopback");
        // https://127.0.0.1 is the https branch, not loopback.
        let https_loopback = acquire_with(&observed, &no_store, "https://127.0.0.1:59003/v1", RedirectPolicy::Count).unwrap().phase;
        if PER_HANDSHAKE_TRUST {
            assert_eq!(https_loopback, "http_client_created_https");
        } else {
            assert_eq!(https_loopback, "fresh_tls_policy");
        }
        // Generation: pooled only for covered loopback with fully observed env.
        let generation = acquire_with(&observed, &no_store, "http://127.0.0.1:59004/v1", RedirectPolicy::Generation).unwrap();
        if PER_HANDSHAKE_TRUST {
            assert_eq!(generation.phase, "http_client_created_generation_loopback");
            assert_eq!(acquire_with(&observed, &no_store, "http://127.0.0.1:59004/v1", RedirectPolicy::Generation).unwrap().phase, "http_client_reused_generation_loopback");
        } else {
            assert_eq!(generation.phase, "fresh_generation_policy");
        }
        assert_eq!(acquire_with(&observed, &no_store, "http://10.0.0.1:59005/v1", RedirectPolicy::Generation).unwrap().phase, "fresh_generation_policy");
        // https Count: per-handshake-trust platforms pool, OpenSSL platforms stay fresh.
        let https = acquire_with(&observed, &no_store, "https://fixture.invalid:59006/v1", RedirectPolicy::Count).unwrap().phase;
        if PER_HANDSHAKE_TRUST {
            assert_eq!(https, "http_client_created_https");
            assert_eq!(acquire_with(&observed, &no_store, "https://fixture.invalid:59006/v1", RedirectPolicy::Count).unwrap().phase, "http_client_reused_https");
        } else {
            assert_eq!(https, "fresh_tls_policy");
        }
        // https Generation: the fully observed environment plus per-handshake
        // trust now pools the redirect-following TLS generation client too.
        let generation_https = acquire_with(&observed, &no_store, "https://fixture.invalid:59006/v1", RedirectPolicy::Generation).unwrap();
        if PER_HANDSHAKE_TRUST {
            assert_eq!(generation_https.phase, "http_client_created_generation_https");
            assert_eq!(acquire_with(&observed, &no_store, "https://fixture.invalid:59006/v1", RedirectPolicy::Generation).unwrap().phase, "http_client_reused_generation_https");
        } else {
            assert_eq!(generation_https.phase, "fresh_generation_policy");
        }
        // Only NO_PROXY set: the unobserved Windows/macOS system matcher is
        // provably bypassed for a covered loopback host, so Count still pools.
        let no_proxy_only = reader(&[("NO_PROXY", "127.0.0.1,localhost,::1")]);
        assert_eq!(acquire_with(&no_proxy_only, &no_store, "http://localhost:59007/v1", RedirectPolicy::Count).unwrap().phase, "http_client_created_loopback");
        assert_eq!(acquire_with(&no_proxy_only, &no_store, "http://localhost:59007/v1", RedirectPolicy::Count).unwrap().phase, "http_client_reused_loopback");
        // Generation with an unobserved system matcher stays fresh.
        assert_eq!(acquire_with(&no_proxy_only, &|| OsProxyStore::Unobserved, "http://localhost:59008/v1", RedirectPolicy::Generation).unwrap().phase, "fresh_generation_system_proxy_policy");
        // An uncovered loopback host keeps the fresh fallbacks when the store
        // is unobservable, and pools when the store is fingerprinted.
        let uncovered = reader(&[("NO_PROXY", "10.0.0.1")]);
        assert_eq!(
            acquire_with(&uncovered, &|| OsProxyStore::Unobserved, "http://127.0.0.1:59009/v1", RedirectPolicy::Count).unwrap().phase,
            "fresh_system_proxy_policy"
        );
        assert_eq!(
            acquire_with(&uncovered, &no_store, "http://127.0.0.1:59009/v1", RedirectPolicy::Count).unwrap().phase,
            "http_client_created_loopback"
        );
        // The TLS-env guard is never overridden by coverage.
        let tls_env = reader(&[("NO_PROXY", "127.0.0.1"), ("SSL_CERT_FILE", "fixture.pem")]);
        assert_eq!(acquire_with(&tls_env, &no_store, "http://127.0.0.1:59010/v1", RedirectPolicy::Count).unwrap().phase, "fresh_tls_policy");
        // An https:// proxy is bypassed for a covered loopback host.
        let https_proxy = reader(&[("HTTP_PROXY", "https://fixture.invalid"), ("NO_PROXY", "127.0.0.1")]);
        assert_eq!(acquire_with(&https_proxy, &no_store, "http://127.0.0.1:59011/v1", RedirectPolicy::Count).unwrap().phase, "http_client_created_loopback");
    }

    #[tokio::test]
    async fn windows_registry_store_pools_and_invalidates_https_generation() {
        let partial = reader(&[("NO_PROXY", "127.0.0.1,localhost,::1")]);
        let url = "https://dgx-fixture.invalid:59020/v1";
        let disabled = || OsProxyStore::Windows(WindowsProxySettings::default());
        // A readable registry with ProxyEnable=0 is fully observed: https
        // generation pools without any env proxy variables at all.
        if PER_HANDSHAKE_TRUST {
            assert_eq!(acquire_with(&partial, &disabled, url, RedirectPolicy::Generation).unwrap().phase, "http_client_created_generation_https");
            assert_eq!(acquire_with(&partial, &disabled, url, RedirectPolicy::Generation).unwrap().phase, "http_client_reused_generation_https");
        } else {
            assert_eq!(acquire_with(&partial, &disabled, url, RedirectPolicy::Generation).unwrap().phase, "fresh_generation_policy");
        }
        // Enabling the store rotates the fingerprint: the disabled-store client
        // is not handed out; returning to it reuses its own earlier client.
        let enabled = || OsProxyStore::Windows(WindowsProxySettings {
            enable: 1, server: Some("http://proxy-fixture.invalid:8080".into()), r#override: None,
        });
        if PER_HANDSHAKE_TRUST {
            assert_eq!(acquire_with(&partial, &enabled, url, RedirectPolicy::Generation).unwrap().phase, "http_client_created_generation_https");
            assert_eq!(acquire_with(&partial, &disabled, url, RedirectPolicy::Generation).unwrap().phase, "http_client_reused_generation_https");
        }
        // An https:// registry proxy keeps TLS-bearing generation fresh.
        let https_proxy = || OsProxyStore::Windows(WindowsProxySettings {
            enable: 1, server: Some("https://proxy-fixture.invalid:8443".into()), r#override: None,
        });
        assert_eq!(acquire_with(&partial, &https_proxy, url, RedirectPolicy::Generation).unwrap().phase, "fresh_generation_proxy_tls_policy");
        // An unobservable store with a partially covering env stays fresh.
        let unobserved = || OsProxyStore::Unobserved;
        assert_eq!(acquire_with(&partial, &unobserved, url, RedirectPolicy::Generation).unwrap().phase, "fresh_generation_system_proxy_policy");
        // A fully covering env leaves no empty slot for the store to fill, so
        // even an unobservable store (macOS shape) pools.
        let observed = reader(&[
            ("HTTP_PROXY", "http://fixture.invalid:8080"),
            ("HTTPS_PROXY", "http://fixture.invalid:8080"),
            ("NO_PROXY", "127.0.0.1,localhost,::1"),
        ]);
        if PER_HANDSHAKE_TRUST {
            assert_eq!(acquire_with(&observed, &unobserved, url, RedirectPolicy::Generation).unwrap().phase, "http_client_created_generation_https");
        } else {
            assert_eq!(acquire_with(&observed, &unobserved, url, RedirectPolicy::Generation).unwrap().phase, "fresh_generation_policy");
        }
        // CGI mode empties the matcher entirely: the store is not consulted.
        let cgi = reader(&[("NO_PROXY", "127.0.0.1"), ("REQUEST_METHOD", "POST")]);
        if PER_HANDSHAKE_TRUST {
            assert_eq!(acquire_with(&cgi, &unobserved, url, RedirectPolicy::Generation).unwrap().phase, "http_client_created_generation_https");
        } else {
            assert_eq!(acquire_with(&cgi, &unobserved, url, RedirectPolicy::Generation).unwrap().phase, "fresh_generation_policy");
        }
    }

    #[tokio::test]
    async fn registry_override_extends_loopback_coverage() {
        // No env NO_PROXY: the effective coverage comes from the registry
        // override while ProxyEnable is set, after hyper-util's `;`-join and
        // `*.`-strip transform (`win::with_system`).
        let empty = reader(&[]);
        let covered = || OsProxyStore::Windows(WindowsProxySettings {
            enable: 1,
            server: Some("http://proxy-fixture.invalid:8080".into()),
            r#override: Some("127.0.0.1;localhost".into()),
        });
        let policy = environment_from(&empty, &covered());
        assert_eq!(policy.no_proxy.as_deref(), Some("127.0.0.1,localhost"));
        assert_eq!(policy.fallback, None);
        assert_eq!(acquire_with(&empty, &covered, "http://127.0.0.1:59021/v1", RedirectPolicy::Count).unwrap().phase, "http_client_created_loopback");
        assert_eq!(acquire_with(&empty, &covered, "http://127.0.0.1:59021/v1", RedirectPolicy::Count).unwrap().phase, "http_client_reused_loopback");
        if PER_HANDSHAKE_TRUST {
            assert_eq!(acquire_with(&empty, &covered, "http://localhost:59022/v1", RedirectPolicy::Generation).unwrap().phase, "http_client_created_generation_loopback");
        } else {
            assert_eq!(acquire_with(&empty, &covered, "http://localhost:59022/v1", RedirectPolicy::Generation).unwrap().phase, "fresh_generation_policy");
        }
        // Windows' `<local>` convention is NOT a hyper-util bypass, and `*.`
        // entries are stripped to their bare domain form.
        let local = OsProxyStore::Windows(WindowsProxySettings {
            enable: 1,
            server: Some("http://proxy-fixture.invalid:8080".into()),
            r#override: Some("<local>;*.intranet".into()),
        });
        let policy = environment_from(&empty, &local);
        assert_eq!(policy.no_proxy.as_deref(), Some("<local>,intranet"));
        assert!(!no_proxy_covers("127.0.0.1", policy.no_proxy.as_deref().unwrap()));
        assert!(no_proxy_covers("host.intranet", policy.no_proxy.as_deref().unwrap()));
        // A disabled store contributes no override even when the value exists.
        let disabled = OsProxyStore::Windows(WindowsProxySettings {
            enable: 0,
            server: Some("http://proxy-fixture.invalid:8080".into()),
            r#override: Some("127.0.0.1".into()),
        });
        assert_eq!(environment_from(&empty, &disabled).no_proxy, None);
        // Env NO_PROXY wins over the registry override.
        let env_wins = reader(&[("NO_PROXY", "10.0.0.1")]);
        assert_eq!(environment_from(&env_wins, &covered()).no_proxy.as_deref(), Some("10.0.0.1"));
    }

    /// The production observer reads the real HKCU Internet Settings key. Two
    /// consecutive observations must agree, and whatever the machine store
    /// says, a fully covering environment keeps the fallback clear.
    #[cfg(windows)]
    #[test]
    fn live_registry_observation_is_stable() {
        let first = observe_os_proxy();
        let second = observe_os_proxy();
        assert_eq!(first, second);
        let read = reader(&[
            ("HTTP_PROXY", "http://fixture.invalid:8080"),
            ("HTTPS_PROXY", "http://fixture.invalid:8080"),
            ("NO_PROXY", "127.0.0.1,localhost,::1"),
        ]);
        assert_eq!(environment_from(&read, &first).fallback, None);
        assert_eq!(
            environment_from(&read, &first).digest,
            environment_from(&read, &second).digest
        );
    }

    #[tokio::test]
    async fn loopback_env_rotation_never_hands_out_a_stale_client() {
        let before = reader(&[
            ("HTTP_PROXY", "http://fixture.invalid:1"),
            ("HTTPS_PROXY", "http://fixture.invalid:1"),
            ("NO_PROXY", "127.0.0.1"),
        ]);
        let after = reader(&[
            ("HTTP_PROXY", "http://fixture.invalid:1"),
            ("HTTPS_PROXY", "http://fixture.invalid:1"),
            ("NO_PROXY", "localhost,::1"),
        ]);
        let url = "http://127.0.0.1:59012/v1";
        let no_store = || OsProxyStore::None;
        assert_eq!(acquire_with(&before, &no_store, url, RedirectPolicy::Count).unwrap().phase, "http_client_created_loopback");
        // The rotated digest is part of the key: the old client is not handed out.
        assert_eq!(acquire_with(&after, &no_store, url, RedirectPolicy::Count).unwrap().phase, "http_client_created_loopback");
        assert_eq!(acquire_with(&after, &no_store, url, RedirectPolicy::Count).unwrap().phase, "http_client_reused_loopback");
        // Returning to the earlier environment legitimately reuses its client.
        assert_eq!(acquire_with(&before, &no_store, url, RedirectPolicy::Count).unwrap().phase, "http_client_reused_loopback");
    }

    #[tokio::test]
    async fn concurrent_loopback_acquisition_is_bounded() {
        let read: std::sync::Arc<dyn Fn(&str) -> Option<std::ffi::OsString> + Send + Sync> =
            std::sync::Arc::new(|name| {
                let pairs = [
                    ("HTTP_PROXY", "http://fixture.invalid:1"),
                    ("HTTPS_PROXY", "http://fixture.invalid:1"),
                    ("NO_PROXY", "127.0.0.1"),
                ];
                pairs.iter().find(|(key, _)| *key == name).map(|(_, value)| (*value).into())
            });
        let mut tasks = Vec::new();
        for _ in 0..32 {
            let read = read.clone();
            tasks.push(tokio::spawn(async move {
                acquire_with(&*read, &|| OsProxyStore::None, "http://127.0.0.1:59013/v1", RedirectPolicy::Count).unwrap().phase
            }));
        }
        let mut phases = Vec::new();
        for task in tasks { phases.push(task.await.unwrap()); }
        assert!(phases.iter().all(|phase| matches!(*phase, "http_client_created_loopback" | "http_client_reused_loopback")));
        assert!(phases.contains(&"http_client_created_loopback"));
        assert!(phases.contains(&"http_client_reused_loopback"));
        assert_eq!(acquire_with(&*read, &|| OsProxyStore::None, "http://127.0.0.1:59013/v1", RedirectPolicy::Count).unwrap().phase, "http_client_reused_loopback");
    }

    // -----------------------------------------------------------------------
    // Benchmarks: `cargo test -p pi-ai --release http_client_pool -- --ignored --nocapture`
    // -----------------------------------------------------------------------

    fn percentile(sorted: &[u128], fraction: f64) -> u128 {
        if sorted.is_empty() { return 0; }
        let index = ((sorted.len() as f64 - 1.0) * fraction).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    fn latency_summary(mut samples: Vec<u128>) -> serde_json::Value {
        samples.sort_unstable();
        serde_json::json!({
            "samples": samples.len(),
            "p50_ns": percentile(&samples, 0.50),
            "p95_ns": percentile(&samples, 0.95),
            "p99_ns": percentile(&samples, 0.99),
            "mean_ns": if samples.is_empty() { 0 } else { samples.iter().sum::<u128>() / samples.len() as u128 },
        })
    }

    /// One keep-alive HTTP/1.1 echo fixture: counts TCP accepts. The returned
    /// handle aborts the accept loop; connections stay served until the tasks
    /// finish and their sockets drop.
    async fn echo_fixture() -> (std::sync::Arc<AtomicUsize>, String, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let accepts = std::sync::Arc::new(AtomicUsize::new(0));
        let counted = accepts.clone();
        let server = tokio::spawn(async move {
            loop {
                let (mut socket, _) = match listener.accept().await {
                    Ok(accepted) => accepted,
                    Err(_) => return,
                };
                counted.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    let mut buffer = Vec::new();
                    let mut chunk = [0u8; 2048];
                    loop {
                        while !buffer.windows(4).any(|w| w == b"\r\n\r\n") {
                            match socket.read(&mut chunk).await {
                                Ok(0) | Err(_) => return,
                                Ok(n) => {
                                    buffer.extend_from_slice(&chunk[..n]);
                                    if buffer.len() > 16384 { return; }
                                }
                            }
                        }
                        buffer.clear();
                        if socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nok").await.is_err() {
                            return;
                        }
                    }
                });
            }
        });
        (accepts, address, server)
    }

    async fn run_scenario(
        pooled: bool, concurrency: usize, requests: usize,
        read: impl Fn(&str) -> Option<std::ffi::OsString> + Send + Sync + 'static,
    ) -> serde_json::Value {
        let (accepts, address, server) = echo_fixture().await;
        let url = format!("http://{address}/bench");
        let read = std::sync::Arc::new(read);
        let latencies: std::sync::Arc<std::sync::Mutex<Vec<u128>>> = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut tasks = Vec::new();
        let quota = requests / concurrency.max(1);
        for _ in 0..concurrency {
            let latencies = latencies.clone();
            let url = url.clone();
            let read = read.clone();
            tasks.push(tokio::spawn(async move {
                for _ in 0..quota {
                    let started = Instant::now();
                    if pooled {
                        let lease = acquire_with(&*read, &|| OsProxyStore::None, &url, RedirectPolicy::Count).unwrap();
                        let response = lease.client.get(&url).timeout(Duration::from_secs(10)).send().await.unwrap();
                        assert_eq!(response.status(), reqwest::StatusCode::OK);
                        assert_eq!(response.text().await.unwrap(), "ok");
                    } else {
                        let client = build(RedirectPolicy::Count).unwrap();
                        let response = client.get(&url).timeout(Duration::from_secs(10)).send().await.unwrap();
                        assert_eq!(response.status(), reqwest::StatusCode::OK);
                        assert_eq!(response.text().await.unwrap(), "ok");
                    }
                    latencies.lock().unwrap().push(started.elapsed().as_micros());
                }
            }));
        }
        let started = Instant::now();
        for task in tasks { task.await.unwrap(); }
        let elapsed = started.elapsed();
        server.abort();
        let _ = server.await;
        let samples = std::sync::Arc::try_unwrap(latencies).unwrap().into_inner().unwrap();
        let accepts = accepts.load(Ordering::SeqCst);
        serde_json::json!({
            "requests": requests,
            "concurrency": concurrency,
            "tcp_accepts": accepts,
            "accepts_per_request": accepts as f64 / requests as f64,
            "wall_ms": elapsed.as_millis(),
            "throughput_rps": requests as f64 / elapsed.as_secs_f64(),
            "latency": latency_summary(samples),
        })
    }

    /// Real measurements for the client-pool change. Ignored in the default
    /// suite; writes research/network_latency/provider/results/client-pool-bench.json.
    #[ignore]
    #[tokio::test]
    async fn client_pool_bench() {
        let read = reader(&[
            ("HTTP_PROXY", "http://fixture.invalid:1"),
            ("HTTPS_PROXY", "http://fixture.invalid:1"),
            ("NO_PROXY", "127.0.0.1"),
        ]);
        // (a) Client construction cost, the per-request price of freshness.
        let mut build_samples = Vec::new();
        for _ in 0..1000 {
            let started = Instant::now();
            let client = build(RedirectPolicy::Count).unwrap();
            build_samples.push(started.elapsed().as_nanos());
            std::hint::black_box(&client);
        }
        let build_cost = latency_summary(build_samples);
        // (a') Pool-hit acquisition cost (digest + lock + take), same loop.
        let url = "http://127.0.0.1:59014/v1";
        acquire_with(&read, &|| OsProxyStore::None, url, RedirectPolicy::Count).unwrap();
        let mut acquire_samples = Vec::new();
        for _ in 0..1000 {
            let started = Instant::now();
            let lease = acquire_with(&read, &|| OsProxyStore::None, url, RedirectPolicy::Count).unwrap();
            acquire_samples.push(started.elapsed().as_nanos());
            std::hint::black_box(&lease);
        }
        let acquire_hit = latency_summary(acquire_samples);
        // (b) End-to-end fresh vs pooled against the echo fixture.
        const REQUESTS: usize = 320;
        let mut end_to_end = serde_json::Map::new();
        for (name, pooled) in [("fresh_client_per_request", false), ("pooled_client", true)] {
            let mut scenarios = serde_json::Map::new();
            for concurrency in [1usize, 8, 32] {
                scenarios.insert(concurrency.to_string(), run_scenario(pooled, concurrency, REQUESTS, reader(&[
            ("HTTP_PROXY", "http://fixture.invalid:1"),
            ("HTTPS_PROXY", "http://fixture.invalid:1"),
            ("NO_PROXY", "127.0.0.1"),
        ])).await);
            }
            end_to_end.insert(name.to_string(), serde_json::Value::Object(scenarios));
        }
        let report = serde_json::json!({
            "build_count_policy": build_cost,
            "acquire_pool_hit": acquire_hit,
            "end_to_end": end_to_end,
        });
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../research/network_latency/provider/results");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("client-pool-bench.json"), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    }
}
