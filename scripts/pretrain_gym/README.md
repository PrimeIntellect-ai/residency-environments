# pretrain-gym images

Images for [`pretrain-gym`](../../environments/pretrain_gym/README.md). Neither holds environment code or evaluation data.

- `Dockerfile.agent` → `pretrain-gym-agent:v1`: the agent's workspace, with Python data tools, CPU `torch` and `transformers`, and fastText `lid.176` in `/opt/models/`.
- `Dockerfile.scorer` → `pretrain-gym-scorer:v1`: Prime scorer VMs only (PyTorch 2.14, CUDA 12.6). Docker runs score on the host.
- `push_prime_images.sh <tag>`: builds both images in the Prime registry, which Prime VMs require.
- `hellaswag.py`: HellaSwag validation accuracy of weights exported by final scoring (`scoring.export_weights`).

```bash
docker build -f scripts/pretrain_gym/Dockerfile.agent -t pretrain-gym-agent:v1 scripts/pretrain_gym
```
