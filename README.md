# Sudoku Vision Benchmark

The purpose of this benchmark is an intelligent test based on SIMPLE Rules where experience does help, but unlike chess there is no need to memorize anything but the rules and some common patterns.

[Read results of latest trial](/trials/README.md)

My _preliminary_ findings give us the following insights:

- We or I should pursue my hypothesis that vision-middleware is beneficial. Vision-middleware as in middleware that transcribes an image and gives a model an alt like `<img alt="{TRANSCRIBE IMAGE HERE}"/>` might be beneficial. I'll do my best to improve the smolagents harness to do this transcription for text-only models. Obviously the lack of vision capabilities is a chip on text-only models' shoulder, so let's see how useful a generalized transcription middleware that isn't tailored to this sudoku benchmark can improve average solve time. We're interested in reducing the average solve time because these models are cheap but their slowness (hypothesis is excessive input tokens caused by lack of vision capability), make them less competitive. They are basically making users choose whether a blind model is worth the affordability and our goal should be for the answer to be a resounding "yes".
- GPT-5.6-sol offers best performance for not the worst price. Parallel-run deepseek-pro is competitive.
- GLM-5.3 Flash is the best and only open-weights model to beat Sonnet-5 at the preliminary benchmark. Furthermore, it is significantly cheaper, with only DeepSeek V4 Flash being cheaper. This makes it a very good candidate for agentic tasks. A follow-up step is to setup dual-model in harnesses so that if the default model is a superior text-only model, and middleware doesn't exist, a prompt with an image is sent to the fallback vision-enabled model.
- ~~Meta's Muse Spark 1.2 open-weights is an important event, or even a vision enabled open-weights deepseek or glm model.~~
- Google is behind, maybe considerably so.
- How good are older proprietary models?

For additional context, GPT-5.2 (Dec 2025) and Opus 4.6 (Feb 2026) were the oldest models from OpenAI and Anthropic to be able to score 3/3 on the preliminary benchmark.

## Running

Note that there's a default timeout of 20 minutes, but based on results, we can probably reduce it to 15 minutes.

The preliminary runs are recipes in the [justfile](justfile):

```sh
just                 # list the recipes
just preliminary     # seed 42, n=3
just preliminary-v   # the same set with --vision-middleware --fresh
```

To re-test one model instead of the whole manifest, name it — a full OpenRouter
ID or any unique substring of one. It still inherits its category and
`expensive` flag from `models.toml`, so it is scheduled and recorded exactly as
it would be inside a full run:

```sh
just one glm-5.3-flash          # one model, same seeded set
just one-v glm-5.3-flash        # one model, through the vision middleware
just one qwen3.8-27b --fresh    # ignore the harness's saved rounds and redo them

uv run cli/run_benchmark.py --model glm-5.3-flash   # the same, without just
```

## Grading

1. Correctness
2. Time to solve
3. Cost effectiveness

### Timing a middleware run

With `--vision-middleware`, transcribing a puzzle happens before the harness
starts, so the work would otherwise fall outside the measured round and make
the middleware look like a free speedup. Each puzzle's transcription time is
therefore added to every model's round for that puzzle: `elapsed` means the
same thing on both sides of the comparison — what it cost to get that answer.
The round record keeps `harness_elapsed` and `transcription_elapsed` beside
it, so the adjustment can always be read back out.

A failed transcription is charged too. The run waited for it either way, and a
middleware that fails slowly is not free.

The transcriptions themselves are saved to `results/transcriptions/` — one
file per puzzle plus an `index.json` naming the transcriber and recording each
puzzle's char count and time — and the harness saves the text each round was
actually given under `transcriptions/` in that model's solutions dir. A
middleware run's timings only mean something next to what it said about the
image, so both are kept for auditing.

## Architecture
