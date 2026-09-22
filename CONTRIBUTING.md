# Contributing to Optimus Agent

Thank you for your interest in contributing to Optimus Agent.

We welcome code contributions, feature suggestions, documentation improvements, general feedback, and bug reports.

## Getting Started

Before starting significant work, please check the existing Issues and Discussions to see whether the topic has already been raised.

For questions, ideas, or proposals, use GitHub Discussions:

https://github.com/telemusai/optimus-agent/discussions

For confirmed bugs or clearly defined tasks, you may open a GitHub Issue:

https://github.com/telemusai/optimus-agent/issues

When reporting a problem, please include enough information to reproduce and understand the issue, including relevant environment details, logs, or screenshots where appropriate.

Do not include API keys, access tokens, credentials, personal information, or other sensitive data.

For security vulnerabilities, please follow [SECURITY.md](SECURITY.md) rather than reporting the issue publicly.

## Pull Requests

Pull requests are welcome.

Before submitting a pull request:

1. Keep changes focused and reasonably scoped.
2. Follow the existing project structure and coding conventions.
3. Add or update tests where appropriate.
4. Run the relevant checks locally before submitting.
5. Clearly describe what the change does and why it is needed.
6. Avoid unrelated refactoring or dependency changes unless they are necessary for the contribution.

For larger changes or new functionality, we recommend opening an Issue or Discussion first so the proposed approach can be considered before significant development work is undertaken.

Contributors using coding agents or other AI-assisted development tools are welcome. Contributors remain responsible for reviewing, understanding, testing, and validating the code they submit.

## Development

Development setup, build instructions, and relevant commands are documented in the [development guide](resources/agent/docs/development.md).

Please follow the repository's existing development and formatting conventions when making changes.

## Changelog Entries

Add `.changes/<slug>.md` for user-visible Rust or Python runtime changes. Use one bullet per change. Preserve historical release notes in `resources/agent/CHANGELOG.md`. The `no-changelog` label is available when an entry is not appropriate.

## Review

Pull requests are reviewed based on correctness, scope, maintainability, compatibility with the project, and successful validation.

Maintainers may request changes before merging or close contributions that are no longer applicable or do not align with the direction of the project.

Thank you for helping improve Optimus Agent.
