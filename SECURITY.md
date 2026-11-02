# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in Optimus Agent, please do not report it through a public GitHub Issue, Discussion, or pull request.

Please report security vulnerabilities directly to **support@telemus.ai**.

Where possible, include:

- The affected version or commit
- The affected component or environment
- Steps to reproduce the issue
- The expected and observed behaviour
- The potential security impact
- Any known workaround or mitigation

Please do not include real API keys, access tokens, credentials, personal information, or other sensitive data. Use redacted or temporary test values where required.

## Vulnerability Review and Resolution

Reported vulnerabilities will be reviewed to determine their scope, severity, and appropriate remediation.

Please allow reasonable time for investigation and remediation before publicly disclosing details that could affect users.

Security fixes will generally target the current version of Optimus Agent. Older versions may not receive security updates.

## Credential and Secret Handling

Never commit credentials to this repository or include them in issues, discussions, or pull requests. The repository ships without credentials: bundled model registrations, including the DGX models, contain public metadata only.

A DGX auth file produced by `optimus-agent dgx-auth export` is itself a credential. It grants access to the exporter's DGX allocation until the gateway session expires — up to about 48 hours — and it cannot be revoked before that expiry. Treat exported auth files like passwords: transfer them only through trusted channels, never commit them to version control, and delete them once they are no longer needed.

## General Issues and Support

For ordinary bugs, feature requests, or other project questions, please use the Optimus Agent GitHub repository:

https://github.com/telemusai/optimus-agent
