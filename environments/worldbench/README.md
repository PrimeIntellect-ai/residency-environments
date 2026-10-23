# WorldBench

WorldBench is a visually diverse multimodal reasoning benchmark with 2,000
image-based multiple-choice questions spanning living things, objects, scenes, the
digital world, academics, documents, and agents. See the
[paper](https://arxiv.org/abs/2606.06538) and
[official dataset](https://huggingface.co/datasets/zlab-princeton/WorldBench).

## Implementation

This environment provides WorldBench as a Verifiers taskset:

- It streams a pinned revision of the Hugging Face dataset.
- It implements the benchmark prompt, answer parser, and metadata.
- It scores exact-match accuracy and reports answer parsing.
- It preserves source images within a 10 MB encoded-payload and 8000-pixel
  dimension limit, resizing only when necessary.
- It supports direct evaluation with `NullHarness` and agent evaluation with harnesses
  such as Codex.

## Results

The following direct evaluations used all 2,000 examples, one rollout per example,
`NullHarness`, and a 32,768-token output cap:

### General results

| Model | Average accuracy | Living Things | Objects | Scenes | Digital World | Academics | Documents, Charts, and Tables | Agents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kimi K3 | 59.95% | 51.84% | 55.73% | 53.79% | 74.60% | 64.47% | 67.44% | 65.06% |
| Kimi-K2.5 | 53.60% | 47.76% | 51.58% | 50.57% | 58.47% | 53.07% | 59.88% | 63.25% |
| Qwen3.5-VL-35B-A3B | 48.30% | 41.22% | 46.64% | 49.43% | 50.00% | 51.32% | 50.58% | 51.81% |

### Paper comparison

| Model | Paper-reported | This environment | Difference |
| --- | ---: | ---: | ---: |
| Qwen3.5-VL-35B-A3B | 52.9% | 48.30% | -4.60 pp |
| Kimi-K2.5 | 52.5% | 53.60% | +1.10 pp |

#### Per-domain accuracy

| Domain | Qwen paper | Qwen this environment | Kimi paper | Kimi this environment |
| --- | ---: | ---: | ---: | ---: |
| Living Things | 48.6% | 41.22% | 49.0% | 47.76% |
| Objects | 52.6% | 46.64% | 52.0% | 51.58% |
| Scenes | 52.2% | 49.43% | 49.7% | 50.57% |
| Digital World | 50.4% | 50.00% | 56.9% | 58.47% |
| Academics | 57.0% | 51.32% | 51.3% | 53.07% |
| Documents, Charts, and Tables | 59.3% | 50.58% | 59.3% | 59.88% |
| Agents | 54.2% | 51.81% | 54.8% | 63.25% |
