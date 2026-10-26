# midog-mitosis results

What the reported evaluation measured, and how the dataset was checked. The task itself is described in the [environment README](../../environments/midog-mitosis/README.md).

## Final evaluation (October 8–9 and 19–20, 2026)

Five models in the same harness, Pi, on the 100 benchmark tasks, five attempts each. All 500 attempts per model were scored.

| Model               | pass@1 | pass@2 | pass@3 | pass@4 | pass@5 | Solved attempts | Mean F1 |
| ------------------- | ------ | ------ | ------ | ------ | ------ | --------------- | ------- |
| Gemini 4 Argon      | 32.4%  | 37.9%  | 40.4%  | 42.0%  | 43.0%  | 162 / 500       | 0.777   |
| Claude Opus 5.5     | 28.6%  | 33.7%  | 36.2%  | 37.4%  | 38.0%  | 143 / 500       | 0.740   |
| GPT-6 Astra         | 22.8%  | 28.2%  | 31.1%  | 33.2%  | 35.0%  | 114 / 500       | 0.721   |
| GLM 5.3 Flash       | 16.8%  | 25.6%  | 31.7%  | 36.6%  | 41.0%  | 84 / 500        | 0.666   |
| DeepSeek V4.1 Flash | 8.8%   | 14.3%  | 18.8%  | 22.6%  | 26.0%  | 44 / 500        | 0.552   |

pass@k is the chance that at least one of k attempts solves a task, estimated from the five. Mean F1 measures how close the listed points came to the true figures (1 is a perfect match); it is recorded, not rewarded. These 100 tasks are the whole benchmark, so the only noise is each model's randomness from one attempt to the next: the 95% intervals ([`report.py`](report.py)) are about ±2 to ±3 points for pass@1 and ±2 to ±5 for pass@5. Gemini 4 Argon's 3.8-point pass@1 lead over Opus is outside that noise (paired by task: +0.9 to +6.7); its pass@5 is 2 points above GLM's, within the noise, and 5 above Opus's. Opus's pass@1 is 5.8 points above GPT-6 Astra's, also outside it (+3.0 to +8.6). Opus's 12-point pass@1 lead over GLM is far outside that noise; GLM's 3-point pass@5 lead over Opus is within it (±5). GLM catches up because its attempts vary more: 38% of tasks have both passing and failing GLM attempts, against 19% for Opus and 23% for Argon and Astra. The best answer that ignores the image solves none of the 100 tasks.

### Protocol

- **Tasks:** the 100 benchmark tasks (`-n 100`): 55 slides, all seven tumor types, 97 single fields and 3 pairs.
- **Harness:** Pi 0.84.1 at `high` reasoning effort, up to 64,000 output tokens per reply and 150 turns ([`pi.toml`](../../configs/midog-mitosis/pi.toml)); verifiers 0.3.1; package 0.9.1. Each attempt ran in a fresh 8 GB Prime sandbox (1 CPU) holding only the prompt and `image.jpg`, with outbound traffic blocked except the model's API.
- **Relay:** requests went through [`jpeg_proxy.py`](jpeg_proxy.py), which re-encodes every image that isn't already JPEG as JPEG at quality 95. Pi resends every image the agent has opened on each turn, and Prime's API can reject requests with too much image data ("Downloaded image content cannot exceed 30MB"), though not consistently: it rejected Gemini above about 23.6 MB and Claude at 35 MB and failed 147 of DeepSeek's 300 attempts on October 7, yet accepted 35 MB from DeepSeek, GLM and OpenAI models on October 8. Through the relay, DeepSeek's largest request carried 17.5 MB of images instead of 68.4 MB (10,577 of its 30,048 logged requests were over 23.6 MB before compression), and Prime accepted every request of all three models. An attempt that still hits the limit counts as failed. For Gemini 4 Argon (largest request 10.1 MB), Prime's API failed 402 of 8,942 requests (4.5%) another way: inside an HTTP 200 response stream whose only event was a 504 error ("Request timed out. Please retry."), which Pi takes for an empty reply, not a provider error. The relay now retries such a request, after pauses of 5 to 60 seconds, up to six times (517 retries in all). GPT-6 Astra (largest request 6.6 MB) needed that for 1 of its 9,493 requests. The earlier runs had few replies of that kind (2 of GLM's 13,321 and 41 of DeepSeek's 33,988 replies carried no usage), and none ended an attempt.
- **Effect of the relay:** the same models ran on October 7 without it, three attempts each (Opus then with 2 GB sandboxes, which it never outgrew). Paired by task (`report.py --base`), pass@1 changed by +0.9 points (−2.4 to +4.3) for Opus, which never hit the limit, so compression costs it at most 2.4 points. GLM changed by +2.5 (−1.6 to +6.5) and DeepSeek by +2.8 (−0.3 to +5.9); without the relay, 14 of GLM's and 147 of DeepSeek's 300 attempts failed on the limit, and pass@1 was 27.7% (Opus), 14.3% (GLM) and 6.0% (DeepSeek).
- **Retries:** an attempt hit by an infrastructure failure (sandbox, network, provider) is rerun in a fresh sandbox, up to twice. None of the first 1,500 attempts needed it. Five of Gemini 4 Argon's did: Prime timed out the same request through all of the relay's retries three times in a row, and the attempt ended without an answer; each was rerun once and finished. One of GPT-6 Astra's did, for the same reason, and was rerun once.
- **Sandbox memory:** 8 GB, because Prime terminated 2 GB sandboxes holding many images without giving a reason (a controlled test lost 2 of 2 DeepSeek attempts in 2 GB and 0 of 2 in 8 GB).
- **Runs:** on Prime runner sandboxes, all with the same settings. Attempts 1–3 ran on October 8 (DeepSeek's on the first 20 tasks in a pilot earlier that day); attempts 4–5 ran that evening and finished at 00:10 UTC on October 9. Gemini 4 Argon ran on October 19, 17:07–20:18 UTC (the five reruns until 20:31), after two 10-attempt pilots that found the timeouts and were discarded. GPT-6 Astra ran from 22:00 UTC on October 19 to 00:39 on October 20 (the rerun until 00:51); its 10-attempt pilot, run at 20:49 with the same settings and without failures, counts as part of the run.

### Failure analysis

| Outcome of the attempt   | Gemini 4 Argon | Opus 5.5 | GPT-6 Astra | GLM 5.3 Flash | DeepSeek V4.1 Flash |
| ------------------------ | -------------: | -------: | ----------: | ------------: | ------------------: |
| Solved                   |            162 |      143 |         114 |            84 |                  44 |
| Too few points           |            268 |      334 |          85 |           209 |                 174 |
| Too many points          |             39 |       10 |         239 |           136 |                 185 |
| Right count, wrong cells |             31 |       13 |          62 |            70 |                  95 |
| No readable answer       |              0 |        0 |           0 |             1 |                   2 |
| Model calls per attempt  |           18.0 |      6.2 |        19.3 |          26.6 |                68.0 |
| Minutes per attempt      |            5.6 |      1.1 |         4.7 |           6.2 |                22.1 |

Opus mostly stops early: two thirds of its attempts list too few points. GLM and DeepSeek miss in both directions and make 4–11 times as many model calls, cropping and re-reading the image. Gemini 4 Argon sits between: it also lists too few points more often than too many, with three times Opus's model calls, and solves the most attempts. GPT-6 Astra errs the other way: nearly half its attempts (239) list too many points. DeepSeek's two attempts without an answer stopped at the 150-turn limit; GLM's one finished without a readable answer.

### Cost

Model calls at Prime list prices, with cached input at 10%: Opus about $89 ($0.18 per attempt), GLM $33 ($0.07), DeepSeek $77 ($0.15). The wallet paid about $282 in total, including sandboxes, the pilot and a DeepSeek runner lost mid-run ($46). Gemini 4 Argon: $182 ($0.36 per attempt) by the same formula; Prime billed $164 for the run's inference, and the run with its pilots, reruns and runners took about $181 from the wallet. GPT-6 Astra: $274 ($0.55 per attempt) by the formula; Prime billed about $305 for its inference, and the run with its pilot, rerun and runners took about $312 from the wallet.

### Isolation and audit

- Every attempt ran in its own sandbox: 500 distinct sandboxes per model.
- [`redteam.py`](redteam.py) `scan` found no expected answer, scoring text or source id in any tool output of the 2,500 attempts.
- The evaluated tasks are exactly the benchmark's 100 tasks in the pinned dataset revision. Traces are kept outside the repository.

## Evaluating another model

Run `configs/midog-mitosis/pi.toml` at `high` reasoning effort with `-n 100 -r 5` and the package's pinned dataset revision, through `jpeg_proxy.py`, then `report.py`. Report how many attempts were actually scored; don't estimate pass@k from fewer than k attempts per task, and don't count missing attempts as failures.

## Dataset checks

Package 0.9.1 pins dataset revision [`d9616cb2`](https://huggingface.co/datasets/tirandazdylan/midog-mitosis/tree/d9616cb2d5207d9b06943c0a263dbd4025228bea). It differs from the earlier `353bf98` only in `warmup/test`, which no longer repeats 27 fields shown in `tasks` pairs; every `tasks` task is byte-identical, so results on either revision compare directly. These checks validate the release independently of any model:

- [`validate_dataset.py`](../../generators/midog-mitosis/validate_dataset.py) rebuilds every task from the MIDOG++ labels without the builder's code; all checks passed. Slides split 315 / 15 / 173 (train / validation / test); training and validation tasks 357 / 64, warmups 543 / 37 / 292. Every `tasks` task has three or more figures, no field or figure is reused, no warmup shows a `tasks` field, and no fields overlap.
- Independent re-crops of six source slides matched (worst mean pixel difference 4.35/255; the bound is 6/255).
- Every task loads through the taskset; every expected answer scores 1, and missing or extra points score 0.
- Random points, even given the true count, passed 0 of 102,400 tries.
- [`mutation_test.py`](../../generators/midog-mitosis/mutation_test.py) planted nine faults, and the validator caught all of them; it also rejects revision `353bf98` for its 27 repeated fields.
- The published revision matches the validated build: every test task has the same id, order, image hash, question and answer, and every expected answer passes.
