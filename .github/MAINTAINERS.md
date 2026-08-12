# Maintainer notes

- The repo is **public**; anyone can open issues and fork PRs, matching the
  policy of `verifiers` and `prime-rl`.
- Residents are outside collaborators with **write** access; company maintainers
  are in `@PrimeIntellect-ai/residency-maintainers`.
- `main` requires a PR, one company CODEOWNER approval, Ruff, unit tests, and
  the required integration gate. There are no bypass actors.
- Secretless checks run on every PR update. Bugbot and Macroscope run
  automatically.
- Fork PR workflows use GitHub's default approval gate (first-time
  contributors require maintainer approval). `tests.yaml` is secretless and
  read-only, so this matches the exposure of `verifiers`/`prime-rl` PR CI.
- Environment changes wait for approval on the `integration` GitHub Environment.
  Approve only after reviewing the exact PR head. New pushes cancel stale runs
  and require fresh approval.
- Keep credentials on the `integration` Environment, never at repository
  level: with write-access contributors, repo-level secrets are readable by
  any workflow edit on any branch.
- Residents can push to any non-`main` branch; branch names are not an
  authorization boundary. Anything pushed to any branch is publicly visible.
