# Maintainer notes: repo security model

This repo is public but contributors (RL residents) are semi-trusted outside
collaborators. The setup enforces:

## Access

- Residents: outside collaborators with **triage** role. They can open
  issues/PRs (satisfies the interaction limit) but cannot push branches or merge.
- Maintainers: company members with **write/maintain**, in the
  `@PrimeIntellect-ai/residency-maintainers` team (referenced by CODEOWNERS).
- Interaction limit `collaborators_only` blocks everyone else from opening
  issues/PRs; it expires after 6 months, so `interaction-limits.yaml` renews it
  monthly (needs the `ADMIN_TOKEN` secret — a fine-grained PAT with
  Administration read/write on this repo).

## Branch protection (ruleset on `main`)

- Require PR, >=1 approval, dismiss stale approvals, require code owner review.
- Required status checks: `Ruff`, `Unit tests`.
- Block force pushes and deletions.

## CI trust boundaries

- `style.yaml` / `tests.yaml`: run on every PR (incl. forks). MUST never
  reference `secrets.*` — fork PRs execute contributor code.
- `integration-tests.yaml`: needs secrets. Gated by (a) the `safe-to-test`
  label, which only write-access users can add, and (b) the `integration`
  GitHub Environment, which should have **required reviewers** configured.
  `remove-safe-to-test-label.yaml` strips the label on every new push so
  approval never carries over to unseen code.
- Actions settings should be: fork PR workflows require approval (at least for
  first-time contributors), default workflow permissions read-only, "Actions
  can create/approve PRs" disabled.

## Review bots

Bugbot / Macroscope are installed in manual-trigger mode only; maintainers
invoke them explicitly. Never configure them to auto-run on every PR.

## When reviewing resident PRs

Treat workflow-file changes (`.github/workflows/`) with extra care — a PR that
edits `integration-tests.yaml` or `remove-safe-to-test-label.yaml` could try to
weaken the gates. `pull_request_target` workflows run the *base* branch's
workflow definition, which protects the label-gate itself, but review changes
to these files carefully before merging.
