# Client-pool reuse for the guarded generation transport

Scope: `openai_http_client_pool.rs` (`acquire`), used by
`post_chat_completions` (guarded/audited SGLang-budget dispatch,
`RedirectPolicy::Generation`) and `openai_sglang_budget.rs`
(`request_count`, `RedirectPolicy::Count`). All measurements are from
`cargo test -p pi-ai --release --lib http_client_pool -- --ignored --nocapture`
on this Windows worktree (rustc 1.98.1, reqwest 0.12.28, hyper-util 0.1.20,
native-tls 0.2.18 + schannel 0.1.29); raw JSON: `results/client-pool-bench.json`.

Measurement conditions disclosure: this bench ran while up to four sibling
cargo jobs were saturating all CPUs (the contention that later triggered the
campaign-wide build gate). Both arms of every comparison executed under the
same concurrent load, so the fresh-vs-pooled ratios and TCP-accept counts are
directly comparable; the absolute latencies are inflated by that contention
and an idle machine should be faster in both arms. No numbers were re-measured
after the gate directive arrived (results JSON written 01:46:50Z, gate
effective later that minute).

## Why fresh clients were expensive

A reqwest `Client::ClientBuilder::build()` costs p50 25.5 us / p95 31.1 us /
p99 47.3 us (mean 59.8 us, 1000 iterations; the tail is TLS-backend
initialization; independently cross-checked at 23.6 us p50 by the provider
transport bench). A pool hit on the same code path (env digest + pool lock +
`take`) costs p50 5.1 us / p95 5.5 us / p99 7.7 us. Every guarded request
also pays a fresh TCP connect because the fresh client's connection pool is
empty.

Unit-labeling note: the committed results JSON recorded these nanosecond
samples under `*_us` keys; the code now emits `*_ns` keys. Values are
unchanged; only the label was wrong.

## End-to-end fresh vs pooled (320 requests, loopback echo fixture)

| mode | conc | TCP accepts | p50 us | p95 us | p99 us | wall ms | rps |
|---|---|---|---|---|---|---|---|
| fresh client/request | 1 | 320 | 579 | 1011 | 13174 | 326 | 980 |
| fresh client/request | 8 | 320 | 4323 | 16423 | 33673 | 245 | 1302 |
| fresh client/request | 32 | 320 | 19306 | 33309 | 33899 | 229 | 1393 |
| pooled | 1 | 1 | 134 | 170 | 268 | 44 | 7145 |
| pooled | 8 | 8 | 750 | 919 | 2622 | 33 | 9651 |
| pooled | 32 | 32 | 2870 | 10420 | 13252 | 41 | 7684 |

Pooled requests make exactly `concurrency` TCP accepts (one per worker
connection), not one per request: accepts/request 0.003/0.025/0.1 vs 1.0.
Sequential latency drops 4.3x (579 us -> 134 us) and throughput rises 7.3x
(980 -> 7145 rps) for the fully serialized count-then-dispatch shape; under
concurrency 32 the fresh path degrades (p50 19.3 ms, contention on client
construction) while the pooled path stays 10x faster at the p50. TLS
handshake overhead was not measured offline: the in-repo certificate fixtures
available to this bench do not include a server-side TLS fixture, and no new
dependencies may be added, so that cell is recorded as blocked rather than
invented.

## Which origins may pool now

The pool still never pools when any environment fallback is set
(TLS env files, unobserved Windows/macOS system proxy, https:// proxy URL),
when the URL has no host, or outside a tokio runtime. The additions:

- `Count` + `http://` loopback (`127.0.0.1`, `localhost`, `[::1]`) with the
  observed `NO_PROXY` provably covering that host: pools even when the
  Windows/macOS system proxy is unobserved or an https:// proxy is
  configured, because hyper-util checks `NoProxy` before any proxy match
  (`Matcher::intercept`, matcher.rs) and this origin never does TLS.
- `Count` + `http://` non-loopback with a fully observed environment
  (existing behavior, unchanged).
- `Count` + `https://` with a fully observed environment, only on
  per-handshake-trust platforms (`PER_HANDSHAKE_TRUST`: schannel and
  security-framework re-resolve system trust on every handshake; OpenSSL
  freezes its probed trust at connector build, so Linux stays fresh).
- `Generation` + `http://` loopback with a fully observed environment AND
  NO_PROXY coverage AND per-handshake trust: a redirect hop can leave the
  covered host, so both the frozen proxy matcher and TLS staleness must be
  excluded. Any other `Generation` origin stays fresh exactly as before.
- `Generation` + `https://` with a fully observed environment on
  per-handshake-trust platforms (added 2026-11-03, see the addendum): the
  environment digest now also pins the Windows registry values the frozen
  matcher consults, so the TLS origin no longer needs NO_PROXY coverage.

## How the no-proxy guard avoids divergence from hyper-util

`no_proxy_covers(host, value)` reimplements hyper-util 0.1.20
`NoProxy::contains` semantics entry by entry (IpNet parse via ipnet 2.12.2
prefix rules, bare IpAddr, else domain with the suffix dot-boundary rule;
IP hosts consult only the IP matcher and domain hosts only the domain
matcher; `*` covers domains only). Beyond the unit table
(`no_proxy_coverage_matches_hyper_util_semantics`),
`no_proxy_coverage_matches_a_frozen_reqwest_matcher` builds a real reqwest
client whose `Proxy::http` matcher was produced through the public
`Proxy::no_proxy(NoProxy::from_string(..))` path - the same
hyper-util `Builder::no(..).build()` parse `Matcher::from_system` uses -
against a live local proxy and target, and asserts the client bypasses the
proxy exactly when `no_proxy_covers` reports coverage, for every host/value
pair the pool gates on. One deliberate divergence: an entry like
`127.000.0.1/8` parses as a CIDR in ipnet's lenient IPv4 parser (leading
zeros allowed on the left side) but is classified as a domain here; the test
asserts that direction only makes this pool more conservative (stay fresh),
never less.

## Observable phases

New `Lease.phase` strings, all visible through `observe_local_phase`:
`http_client_reused_loopback` / `http_client_created_loopback`,
`http_client_reused_https` / `http_client_created_https`,
`http_client_reused_generation_loopback` /
`http_client_created_generation_loopback`. Unchanged:
`fresh_generation_policy`, `fresh_tls_policy`, `fresh_unknown_policy`,
`fresh_system_proxy_policy`, `fresh_proxy_tls_policy`, `fresh_changed_policy`,
`http_client_reused`, `http_client_created`.


## Addendum 2026-11-03: HTTPS generation pooling and the Windows registry fingerprint

Two changes to the pooling gate, both offline-verified (unit tests in
`openai_http_client_pool.rs`; no new live bench yet, so no new numbers):

1. `Generation` + `https://` now pools on per-handshake-trust platforms
   whenever the environment is fully observed. New phases
   `http_client_created_generation_https` /
   `http_client_reused_generation_https`. Fresh generation fallbacks are now
   reason-specific: `fresh_generation_system_proxy_policy`,
   `fresh_generation_tls_policy`, `fresh_generation_proxy_tls_policy`, with
   `fresh_generation_policy` kept for the fully-observed-but-fresh cases
   (non-per-handshake platforms, unknown origin).
2. The Windows `fresh_system_proxy_policy` blanket is replaced by a real
   fingerprint: the pool reads exactly the HKCU `Internet Settings` values
   hyper-util 0.1.20 `win::with_system` consults (`ProxyEnable`, `ProxyServer`,
   `ProxyOverride`, presence included) through `windows-registry` (already in
   the lock via hyper-util) and hashes them into the environment digest. A
   readable registry makes the store observed; rotation invalidates pooled
   clients through the existing digest key and double-read checks. macOS
   SystemConfiguration and unopenable registry keys keep
   `fresh_system_proxy_policy`. The fingerprint replicates hyper-util's
   `;`-join and `*.`-strip `ProxyOverride` transform, so registry-only
   NO_PROXY coverage gates loopback pooling exactly like the frozen matcher
   would, and an `https://` registry proxy keeps TLS-bearing clients fresh
   (`fresh_proxy_tls_policy`). Windows' `<local>` override convention is NOT a
   hyper-util bypass and never counts as coverage.

Live impact path (not yet measured end-to-end): the DGX GLM 5.3 model is
configured with the HTTPS public LaunchPad base URL, so its audited
generation traffic previously took the fresh-client-per-request path on
Windows; it now takes the pooled branch whenever the registry fingerprint is
stable.
