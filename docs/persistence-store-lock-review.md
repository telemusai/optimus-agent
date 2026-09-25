# Settings and credential persistence review

## Guarantees

- Auth migration does not retire `oauth.json`, remove settings API keys, or
  report migrated providers when the destination write or durability sync fails.
  A post-rename sync failure can leave both destination and sources; preserve
  those sources for review rather than deleting them automatically.
- Settings read errors (including invalid UTF-8) abort writes. Only a missing
  file means no current settings. The late first-write recheck uses the same rule.
- Auth and settings use stable OS-owned `.lock` files. Never delete these files
  during normal operation. Their existence does not mean the store is locked.
  Ownership releases when the descriptor closes or its process dies.
- OAuth refresh still rereads credentials under lock, holds that lock across the
  network refresh and commit, and retains existing provider/retry semantics.
  Cancellation and unwinding release ownership. Lock age never authorizes takeover.

## Upgrade and recovery

Legacy directory locks remain fail-closed, regardless of their age. Their format
cannot prove whether a slow writer is alive. Stop **all** old and new writers
before inspecting and manually removing a confirmed legacy lock directory.
Do not delete regular lock files or perform recovery while a writer is running.
Old writers cannot acquire the new regular-file lock through `mkdir`; mixed
versions fail closed rather than running concurrent transactions.

## Security review and remaining scope

The threat addressed is cooperating-process concurrency, crashes, failed reads,
and failed persistence, not an attacker who can replace files in the profile
directory. New Unix side files are mode 0600 and opened with `O_NOFOLLOW`;
Windows opens the reparse point itself. Non-regular files and symlinks are rejected.
No credential format, provider identity, logging payload, or migration schema changes.
Tests use temporary stores and synthetic data, without provider requests.

Migration still lacks a transaction spanning auth and settings. Its initial
absence check and source snapshots can race concurrent writers; a shared ordered
cross-store protocol and no-clobber destination publication remain separate work.
Auth/settings atomic replacement fsync defaults are unchanged; this patch does
not claim complete power-loss durability. OS locks require a filesystem with
working lock support. Adversarial directory replacement and hard-link aliases
remain outside this local trusted-profile contract.

Regression coverage includes destination failure before/after simulated commit,
invalid-byte settings reads and late rechecks, live aged auth ownership,
refresh cancellation, callback failure, process death, and lock symlink rejection.
