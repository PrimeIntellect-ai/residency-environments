# PhySim support

The environment runs from a public Python base image and installs pinned public
packages during trusted setup. No local image build is required.

`smoke_runtime.py` runs the standard Verifiers evaluation CLI against a local
scripted provider. It performs real experiments, validates and submits a frozen
predictor, checks runtime isolation, and compares its grade with the bundle's
stored reference. Docker mode needs no paid services. Prime mode needs sandbox
credits but makes no paid model calls.

`check_published_install.py` installs a non-editable wheel in a separate virtual
environment, downloads all three current preparations anonymously, checks their
offline caches, runs a native experiment, and verifies each reference score:

```sh
uv build environments/physim --out-dir outputs/physim-dist
uv run --no-sync python scripts/physim/check_published_install.py \
  --wheel outputs/physim-dist/physim-0.13.0.dev0-py3-none-any.whl \
  --configs configs/physim --workdir outputs/physim-clean-install \
  --report outputs/physim-clean-install-report.json
```

The optional `docker/Dockerfile` bakes the scientific dependencies into an image
to reduce repeated installation time. Publish such an image to a registry before
using it on Prime, then set both `env.taskset.task.agent_image` and
`env.taskset.task.tools.predictor_runtime.image` to its pullable reference.
It contains no simulator, world data, or grader.
