# Native hardening and clipboard follow-through

Target baseline: Optimus main through PR #87, including all PR #80 provenance, sidebar, input, session identity, and native ownership behavior. No provider payload, model catalog, compaction retry, sidebar layout, daemon wire shape, or release provenance contracts are deliberately changed.

## Included
- Migration source cleanup requires a successful durable destination write.
- Auth/settings transactions use stable OS-owned side-file locks, not age-based ownership stealing. Cancellation and process death release ownership. Unreadable settings fail writes closed.
- Session tail repair returns failure to writable opens. Parent cycles terminate without rewriting valid stored rows.
- Slow/unknown live worker adoption enters existing bounded recovery instead of parking or discarding registrations. Failed reconnect only clears its own client.
- Clipboard tools must accept all input and exit successfully. Native selection uses the same backend as `/copy`. SSH skips host clipboard tools and forwards OSC 52; tmux passthrough is encoded. Terminal forwarding is explicitly unconfirmed, not clipboard success.
- Local kernel PATH includes managed search tools unless an explicit caller PATH overrides it. Search guidance retains Jev-supplied candidates.
- Python `bash()` adds bounded accident checks for supported literal destructive Git and recursive-force removal commands, with explicit per-call overrides. This is not a shell sandbox; see `bash-accident-guards.md` for exact coverage and exclusions.

See `persistence-store-lock-review.md`, `native-clipboard-delivery.md`, `native-worker-adoption-retry.md`, and `managed-kernel-search-path.md` for contracts and limitations.

## Cutover and rollback
The new permanent `<store>.lock` regular files are not compatible with old directory-lock writers. Stop all old daemons/workers sharing the profile before starting the new build. Never remove a lock while any writer may still be running. Old legacy lock directories fail closed; remove them only after all writers are confirmed stopped. A rollback to an old directory-lock build likewise requires stopping every new writer before removing the new regular lock side files. Store data is not migrated by this lock change.

## Deliberately not claimed
- Migration is not a cross-store atomic transaction; concurrent auth/settings migration coordination remains a follow-up.
- No complete power-loss fsync upgrade or cross-process repair/append transaction is claimed.
- No full schema-validation or recovery-budget rewrite; existing bounded recovery, MCP protections and stop finalization are reused.
- No native macOS/Windows runtime verification. Platform-specific workstream 6 is deferred.
- OSC 52 delivery cannot certify acceptance by a user's terminal or clipboard manager. Live SSH/client-terminal acceptance is a remaining manual check.
- Tests use disposable stores, fake processes/servers and synthetic credentials. No paid provider or real profile fixtures.

## Verification and review
Focused local tests and `scripts/check.sh` results are recorded in the PR. Protected coverage includes PR #80 provenance and sidebar/selection tests and the FW-Kimi-K3 request contract. Earlier failing attempts, if any, must be distinguished from the final passing result. GitHub Actions/checks are out of scope.

### Recorded local results
- `scripts/check.sh`: passed (workspace/all-target check, 33 packaging/launcher tests, native layout gate).
- Focused native hardening plus sidebar/compaction regressions: 138 passed; one intentionally ignored subprocess fixture is exercised through its parent test.
- Kimi K3 request-contract regression: 1 passed. Build-provenance integration tests: 4 passed.
- `pi-tui` fullscreen regressions: 15 passed.
- Frozen Python Bash suite: 102 passed, including 34 guard tests. Earlier broad runs intermittently failed the existing process-journal assertion; the unchanged HEAD Bash implementation reproduced that failure in its original 68-test suite. The precise interleaving remains a follow-up, not a claimed fix.
- The first focused attempt exposed long Unix socket fixture paths and missing test theme setup; both were fixed before the passing run.
- Broader native UI run: 18 passed after reconciling two stale expectations with committed `95c61aeb82` behavior. Tests now require the exact versioned brand caption and retained header across themes; geometry, selection, dock and history checks remain. Production UI rendering was not changed.
