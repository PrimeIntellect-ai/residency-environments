# Maintainer notes: repo security model

This repo is **private** (pre-release residency work) with semi-trusted outside
collaborators (RL residents). The setup enforces:

## Access

- Residents: outside collaborators with **write** role. They push branches and
  open PRs directly; they cannot merge to `main` (code-owner review required)
  and cannot reach integration secrets (environment-gated).
- Maintainers: company members in `@PrimeIntellect-ai/residency-maintainers`
  (maintain role; referenced by CODEOWNERS).
- Forking is disabled; visibility changes are admin-only.

## Branch protection (ruleset `protect-main`)

- Require PR, >=1 approval, dismiss stale approvals, require code owner review.
- Required status checks: `Ruff`, `Unit tests`.
- Block force pushes and deletions. **No bypass actors** — nobody pushes to
  `main` directly, admins included.

## CI trust boundaries

- `style.yaml` / `tests.yaml`: run on every PR with no maintainer gate. MUST
  never reference `secrets.*`.
- `integration-tests.yaml`: needs secrets. All its secrets live on the
  `integration` GitHub Environment with **required reviewers**, so every run
  waits for a maintainer's approval before secrets are injected — even if a
  contributor edits the workflow on their branch. Never move these secrets to
  repo level: with write-access contributors, repo-level secrets are readable
  by any workflow edit on any branch.
- Actions settings: default workflow permissions read-only, "Actions can
  create/approve PRs" disabled.

## Review bots

Bugbot / Macroscope are configured in manual-trigger mode only; maintainers
invoke them explicitly. Never configure them to auto-run on every PR.

## When reviewing resident PRs

Treat workflow-file changes (`.github/workflows/`) with extra care — the
environment gate protects secrets, but a workflow edit could still waste
runner time or attempt privilege escalation. Also remember residents can push
to *any* non-main branch, so don't treat branch names as authorization.
