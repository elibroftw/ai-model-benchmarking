# Trials

## Summary

After running the benchmark, even if it fails mid-way the following command reads the solutions dir.

`just summarize-md`

3 puzzles (easy 1, hard 1, medium 1)

| # | model | type | mw | score | avg s | cost | $/h | images | tok_in | tok_out |
|--:|---|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | `openai/gpt-5.6-terra` | V | -- | 3/3 | 66.2 | $0.77 | $13.95 | 3/3 | 260,255 | 20,740 |
| 2 | `openai/gpt-5.6-sol` | V | yes | 3/3 | 70.5s | $0.37 | $6.25 | 3/3 | 48,229 | 17,419 |
| 3 | `openai/gpt-5.6-terra-pro` | V | -- | 3/3 | 97.1 | $1.73 | $21.36 | 3/3 | 464,384 | 66,577 |
| 4 | `anthropic/claude-opus-5` | V | -- | 3/3 | 99.7 | $1.48 | $17.83 | 3/3 | 154,825 | 28,293 |
| 5 | `anthropic/claude-fable-5` | V | yes | 3/3 | 105.3s | $2.93 | $33.34 | 3/3 | 145,111 | 29,483 |
| 6 | `anthropic/claude-fable-5.1` | V | yes | 3/3 | 108.1s | $3.16 | $35.06 | 3/3 | 192,510 | 24,656 |
| 7 | `google/gemini-3.8-flash` | V | yes | 3/3 | 129.4s | $0.47 | $4.34 | 3/3 | 344,179 | 56,051 |
| 8 | `openai/gpt-5.6-sol-pro` | V | -- | 3/3 | 141.4 | $1.78 | $15.08 | 3/3 | 288,251 | 62,336 |
| 9 | `meta/muse-spark-1.2` | V | -- | 3/3 | 142.4s | $0.62 | $5.22 | 3/3 | 158,599 | 99,192 |
| 10 | `z-ai/glm-5.3-flash` | V | -- | 3/3 | 173.3s | $0.07 | $0.50 | 3/3 | 224,641 | 75,772 |
| 11 | `anthropic/claude-sonnet-5` | V | -- | 3/3 | 212.3 | $1.59 | $9.00 | 3/3 | 417,092 | 75,732 |
| 12 | `deepseek/deepseek-v4-pro-0813` | T | yes | 3/3 | 273.3s | $0.25 | $1.09 | 3/3 | 183,190 | 64,437 |
| 13 | `meta/muse-spark-1.3` | V | yes | 3/3 | 301.8s | $1.33 | $5.31 | 3/3 | 332,985 | 215,995 |
| 14 | `z-ai/glm-5.3` | T | -- | 3/3 | 404.7 | $1.86 | $5.51 | 3/3 | 957,568 | 117,955 |
| 15 | `x-ai/grok-4.6` | V | -- | 3/3 | 418.2 | $0.76 | $2.19 | 3/3 | 102,119 | 93,413 |
| 16 | `moonshotai/kimi-k3` | V | -- | 3/3 | 499.2 | $3.89 | $9.35 | 3/3 | 900,394 | 119,054 |
| 17 | `qwen/qwen3.8-max` | V | -- | 3/3 | 499.6 | $1.00 | $2.39 | 3/3 | 258,427 | 79,789 |
| 18 | `~deepseek/deepseek-v4-flash-latest` | T | -- | 3/3 | 527.4 | $0.13 | $0.30 | 3/3 | 1,396,193 | 218,938 |
| 19 | `qwen/qwen3.8-27b` | V | -- | 3/3* | 527.9s | $0.35 | $0.79 | 3/3 | 138,624 | 97,426 |
| 20 | `z-ai/glm-5.2` | T | -- | 3/3 | 583.3 | $1.23 | $2.52 | 3/3 | 3,172,546 | 152,419 |

NOTE: Non-discounted prices are used to calculate costs. As of 2026-08-26 this applies to gpt-5.6-sol and glm-5.3-flash.

TODO: if an older model has lost in both correctness and cost effectiveness respective to being closed/open-weights, it should not be tested ever again (don't use preliminary results to filter models). Based on preliminary results, here are the models I would remove:

- glm-5.3 (made obsolete by v4 pro)
- glm-5.2 (made obsolete by v4 flash)
- qwen3.8-max (made obsolete by grok 4.6)
- openai/gpt-5.6-sol-pro (made obsolete by 5.6-sol)
  - This is a great example of why preliminary results should not be used to filter models, we need to increase quantity + difficulty
- qwen3.8-27b, moonshotai/kimi-k3
  - I want to test unsloth's GGUF, so not yet

## WIP: Vision Trials

Vision improved not just text-only models like deepseek-v4-pro, but also vision-capable models like gpt-5.6-sol. It seems that transcription reduces the time spent thinking by models that have higher capabilities. In other words, excessive thinking is reduced when transcriptions are provided. We can see that the transcriptions also significantly lowered tokens, which also significantly lowered costs (more so for deepseek v4 pro than gpt 5.6 sol).

| # | model | type | mw | score | avg s | cost | $/h | images | tok_in | tok_out |
|--:|---|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|
| - | `openai/gpt-5.6-sol` | V | yes | 3/3 | 70.5s | $0.37 | $6.25 | 3/3 | 48,229 | 17,419 |
| - | `openai/gpt-5.6-sol` | V | -- | 3/3 | 97.9 | $0.70 | $8.55 | 3/3 | 119,367 | 22,039 |
| - | `deepseek/deepseek-v4-pro-0813` | T | yes | 3/3 | 273.3s | $0.25 | $1.09 | 3/3 | 183,190 | 64,437 |
| - | `deepseek/deepseek-v4-pro-0813` | T | -- | 3/3 | 310.2 | $0.52 | $2.03 | 3/3 | 608,433 | 62,039 |
| - | `anthropic/claude-fable-5` | V | yes | 3/3 | 105.3s | $2.93 | $33.34 | 3/3 | 145,111 | 29,483 |
| - | `anthropic/claude-fable-5` | V | -- | 3/3 | 131.4 | $3.83 | $34.97 | 3/3 | 195,142 | 37,557 |

Some models saw worse performance, which to me suggests that these models weren't honestly thinking in the first place.
Proper tracing is required to ensure these models did not cheat to begin with.

These findings suggest that text-only models were able to cut down the time spent on parsing puzzles but they are still hard blocked on reasoning, which won't improve just because they have a text-only input.

We're waiting on an open-source text-only model that actually has strong and efficient reasoning capabilities where a vision-middleware would be considered optimization.

## More Findings

- Muse spark improved on a second run of the preliminary results, going from 190seconds to 140 seconds (near openai and anthropic's frontier models).

## Models That Failed Preliminary Benchmark

The preliminary benchmark was the following command. The harness was run repeatedly with a seed of 42 and n-puzzles 3 until the harness wasn't the blocking factor regarding completion. Models that could not complete 3/3 are listed in models.toml. Models that were released less than 6 months ago and failed the preliminary benchmark are listed here.

```py
uv run cli/run_benchmark.py \
    --harness-cmd "uv run --project ./sudoku-agent-harness sudoku-agent-harness" \
    --harness-id smolagents \
    --n-puzzles 3 \
    --seed 42 \
    -v
```

- google/gemini-3.7-flash (August 13, 2026)
- qwen/qwen3.8-2.4t-a95b (August 12, 2026)
- meta/muse-glimmer-30b (August 10, 2026)
- thinkingmachines/inkling (July 30, 2026)
  - It kept trying to run code that was forbidden and irrelevant to the test. For example, it was opening images in the default program at one point. This is after I wanted to give the model a second try since I saw it was harness errors and not something else. After stripping the permissions to display images, it suddenly started replying impossible (0/3).
- inclusionai/ling-3.0-flash (July 23rd, 2026)
- thinkingmachines/inkling-small (July 15, 2026)
- minimax/minimax-m3 (June 1, 2026)
- xiaomi/mimo-v2.5 (Apr 22, 2026)
- xiaomi/mimo-v2.5-pro (Apr 22, 2026)
- moonshotai/kimi-k2.6 (April 20, 2026)
- google/gemma-4-31b-it (April 2, 2026)
