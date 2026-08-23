# Sudoku Vision Benchmark

The purpose of this benchmark is an intelligent test based on SIMPLE Rules where experience does help, but unlike chess there is no need to memorize anything but the rules and some common patterns.

[Read results of latest trial](/trials/README.md)

My _preliminary_ findings give us the following insights:

- We or I should pursue my hypothesis that vision-middleware is beneficial. Vision-middleware as in middleware that transcribes an image and gives a model an alt like `<img alt="{TRANSCRIBE IMAGE HERE}"/>` might be beneficial. I'll do my best to improve the smolagents harness to do this transcription for text-only models. Obviously the lack of vision capabilities is a chip on text-only models' shoulder, so let's see how useful a generalized transcription middleware that isn't tailored to this sudoku benchmark can improve average solve time. We're interested in reducing the average solve time because these models are cheap but their slowness (hypothesis is excessive input tokens caused by lack of vision capability), make them less competitive. They are basically making users choose whether a blind model is worth the affordability and our goal should be for the answer to be a resounding "yes".
- GPT-5.6-sol offers best performance for not the worst price. Parallel-run deepseek-pro is competitive.
- Meta's Muse Spark 1.2 open-weights is an important event, or even a vision enabled open-weights deepseek or glm model.
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

## Grading

1. Correctness
2. Time to solve
3. Cost effectiveness

## Architecture
