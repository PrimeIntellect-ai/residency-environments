# generators

Data generation and validation code for synthetically generated environments.

Each environment whose tasks are synthetically generated keeps two directories:

- `environments/<env-name>/` — the installable taskset.
- `generators/<env-name>/` — the code that generated and validated that data.

Generator code is not part of the environment package: it is kept here so the data's provenance stays reviewable and the generation can be re-run or extended later.
