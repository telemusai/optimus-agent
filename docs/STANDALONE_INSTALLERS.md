# Publishing the standalone installers

The maintained download sources are `installers/install.sh`, `installers/install.ps1`, and `installers/install.py`. The shell/PowerShell entry points download the shared Python helper over HTTPS. The helper queries Git refs directly from `https://github.com/telemusai/optimus-agent.git`; it does not depend on the GitHub Releases API, release assets, Actions, or a GitHub token.

Stable means the highest numeric `vMAJOR.MINOR.PATCH` tag, excluding prerelease/build suffixes. An annotated tag is resolved to its peeled commit. After cloning only the selected tag, the helper checks `HEAD` against that commit before building. A moved tag causes a failure instead of silently installing different source. Publish new stable versions as new tags; do not move existing release tags. A source tag is sufficient; a GitHub Release entry is optional.

## Release and deployment

1. Align the workspace, package resource, and Python runtime versions and lockfiles. Run `bash scripts/check.sh` and the installer tests.
2. Commit/push the approved source and publish its matching stable tag. Keep GitHub Actions disabled. The first supported standalone tag is `v0.1.1`; older tags lack the Rust-only release installer.
3. Publish only the three installer objects below using the authenticated website AWS profile. Do not sync or delete unrelated website content.
4. Invalidate `/optimus-agent/*` and compare the public downloads with their committed sources.

```bash
aws s3 cp installers/install.py s3://telemus.ai/optimus-agent/install.py \
  --profile telemus --region ap-southeast-2 --content-type 'text/plain; charset=utf-8' \
  --cache-control 'public, max-age=300, must-revalidate'
aws s3 cp installers/install.sh s3://telemus.ai/optimus-agent/install.sh \
  --profile telemus --region ap-southeast-2 --content-type 'text/plain; charset=utf-8' \
  --cache-control 'public, max-age=300, must-revalidate'
aws s3 cp installers/install.ps1 s3://telemus.ai/optimus-agent/install.ps1 \
  --profile telemus --region ap-southeast-2 --content-type 'text/plain; charset=utf-8' \
  --cache-control 'public, max-age=300, must-revalidate'
aws cloudfront create-invalidation --profile telemus \
  --distribution-id E32UW2M6VBOZHF --paths '/optimus-agent/*'
```

Later source tags are discovered automatically without changing these download endpoints. Redeploy the three files together when the bootstrap itself changes, maintaining compatibility with the tagged `scripts/rust_release.py` interface. The checkout-level `install.sh` is a different script: it installs that checkout, not the latest remote tag.

## Installation properties and validation

- Git HTTPS authenticates the source host; the tag commit check prevents a ref changing between discovery and cloning. This does not provide independent publisher signature verification.
- Builds use `Cargo.lock` with `--locked`. All downloaded tag code is trusted publisher code and runs with the installing user's permissions. No elevated privilege or package-manager changes are requested.
- The helper uses argument arrays, private build directories, and the tagged release installer's explicit bundle allowlist. Failed builds are removed before activation. A successful version probe precedes activation; launcher/pointer failures restore the previous launchers and retain the old release.
- The installer does not delete older releases, alter sessions or credentials, restart daemons, or enable Actions. PowerShell updates only the user's PATH after a successful install. POSIX users add the documented bin directory to their shell PATH.
- Offline tests exercise real local Git tags and cloned source with a synthetic compiler. They cover tag ordering, annotated tags, prerelease exclusion, moved refs, installation/update/force, repeat-install idempotence, cleanup after build failure, paths with spaces, and Windows launcher rollback. No provider API calls are used.
- Run a real Linux build from the public endpoint in an isolated prefix for release smoke coverage. PowerShell parsing and simulated Windows activation are useful checks, but a native Windows build/launch and a native macOS build require their respective hosts before claiming those platforms have been smoke-tested.

These are distribution and documentation changes. The terminal command gains a Windows launcher; application behavior and daemon protocol are unchanged. The separate Telemus 3D and Web app surfaces are unaffected.
