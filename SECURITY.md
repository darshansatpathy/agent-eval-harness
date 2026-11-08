# Security policy

Report a vulnerability privately through [GitHub security advisories](https://github.com/darshansatpathy/agent-eval-harness/security/advisories/new). Please do not open a public issue. You can expect a first reply within five working days.

## What is in scope

- Prompt injection reaching the reference agent's answer. The knowledge base contains a planted injection on purpose (`kb/release-notes-import.md`) and three critical cases test it; a way past the guard that the gate does not catch is in scope.
- API keys. The OpenAI-compatible adapters read the key from `EVALGATE_API_KEY` only and never write it to runs, reports or logs.
- The CI workflows: actions are pinned to major versions, run with read-only permissions, and the gate job runs no code from the golden set.

## Supported versions

The latest release receives fixes.
