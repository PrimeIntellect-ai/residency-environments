# Maintainer notes

- Residents are outside collaborators with **write** access; company maintainers
  are in `@PrimeIntellect-ai/residency-maintainers`.
- `main` requires a PR, one company CODEOWNER approval, Ruff, unit tests, and
  the required integration workflow. There are no bypass actors.
- Secretless checks run on every PR update. Bugbot and Macroscope run
  automatically.
- Environment changes wait for approval on the `integration` GitHub Environment.
  Approve only after reviewing the exact PR head. New pushes cancel stale runs
  and require fresh approval.
- Keep integration credentials on the Environment, never at repository level.
- Residents can push to any non-`main` branch; branch names are not an
  authorization boundary.
