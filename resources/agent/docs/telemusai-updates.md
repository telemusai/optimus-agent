# TelemusAI main updates

`/update` and `prime-agent update` install the latest published build of
[`telemusai/prime-agent` main](https://github.com/telemusai/prime-agent/tree/main).
Existing stable, beta, and custom Astra installations all follow this channel.
Model settings and authentication stay in the existing agent directory.

The updater reads `main.json` from the fork's latest GitHub release. The manifest
must identify the main channel and point to that version's `prime-agent` tarball
in the same repository. All four internal packages use immutable release URLs.
When discovery fails, the update stops before installation or daemon restart.
Startup notices remain quiet when offline or when GitHub is unavailable.

Pushes to the fork's `main` run **Publish TelemusAI main**, which checks, tests,
builds, and publishes a version named
`<package-version>-telemusai.main.<commit-count>.g<commit-sha>`.
The GitHub **Latest** release denotes the latest published main build. Publication
can lag the branch while checks run. Other branches and tags do not publish it.

`PRIME_AGENT_DOWNLOAD_BASE_URL` remains an explicit mirror override. A mirror
must serve `main.json` with the same channel, package, version, and tarball fields.
`PI_OFFLINE` and `PI_SKIP_VERSION_CHECK` disable discovery.

To prepare GitHub release assets locally after compiling the workspace:

```sh
node scripts/pack-prime-agent-release.mjs \
  --github-repository telemusai/prime-agent \
  --channel main --version <main-build-version> \
  --out-dir packages/coding-agent/release/telemusai-main
```

Publish all four tarballs, `SHA256SUMS`, and `main.json` together under the
matching `v<main-build-version>` tag. Keep older release assets available because
the CLI tarball's internal dependencies reference those exact URLs.
