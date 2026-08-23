# Trials

## Summary

After running the benchmark, even if it fails mid-way the following command reads the solutions dir.

`uv run summarize_results.py --format markdown --hide-errors`

| # | model | type | score | avg s | cost | $/h | images | tok_in | tok_out |
|--:|---|:--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | `openai/gpt-5.6-sol` | V | 3/3 | 97.9 | $0.46 | $5.63 | 3/3 | 119,367 | 22,039 |
| 2 | `anthropic/claude-opus-5` | V | 3/3 | 99.7 | $1.48 | $17.83 | 3/3 | 154,825 | 28,293 |
| 3 | `anthropic/claude-fable-5` | V | 3/3 | 131.4 | $3.83 | $34.97 | 3/3 | 195,142 | 37,557 |
| 4 | `openai/gpt-5.6-sol-pro` | V | 3/3 | 141.4 | $1.20 | $10.18 | 3/3 | 288,251 | 62,336 |
| 5 | `meta/muse-spark-1.2` | V | 3/3 | 190.3 | $1.24 | $7.80 | 3/3 | 602,540 | 113,661 |
| 6 | `deepseek/deepseek-v4-pro-0813` | T | 3/3 | 310.2 | $0.52 | $2.03 | 3/3 | 608,433 | 62,039 |
| 7 | `z-ai/glm-5.3` | T | 3/3 | 404.7 | $1.86 | $5.51 | 3/3 | 957,568 | 117,955 |
| 8 | `x-ai/grok-4.6` | V | 3/3 | 418.2 | $0.76 | $2.19 | 3/3 | 102,119 | 93,413 |
| 9 | `moonshotai/kimi-k3` | V | 3/3 | 499.2 | $3.89 | $9.35 | 3/3 | 900,394 | 119,054 |
| 10 | `qwen/qwen3.8-max` | V | 3/3 | 499.6 | $1.00 | $2.39 | 3/3 | 258,427 | 79,789 |
| 11 | `~deepseek/deepseek-v4-flash-latest` | T | 3/3 | 527.4 | $0.13 | $0.30 | 3/3 | 1,396,193 | 218,938 |
| 12 | `z-ai/glm-5.2` | T | 3/3 | 583.3 | $1.23 | $2.52 | 3/3 | 3,172,546 | 152,419 |
| 13 | `qwen/qwen3.8-27b` | V | 3/3 | 584.2 | $0.41 | $0.85 | 3/3 | 197,418 | 111,643 |

## Models That Failed Preliminary Benchmark

The preliminary benchmark was the following command. The harness was run repeatedly with a seed of 42 and n-puzzles 3 until the harness wasn't the blocking factor regarding completion. Models that could not complete 3/3 are listed in models.toml. Models that were released less than 6 months ago and failed the preliminary benchmark are listed here.

```py
uv run run_benchmark.py \
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
