# Trials

## Summary

After running the benchmark, even if it fails mid-way the following command reads the solutions dir.

`uv run summarize_results.py --format markdown`

| # | model | type | score | cost | claimed | images | avg s | tok_in | tok_out | errors |
|--:|---|:--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | `meta/muse-spark-1.2` | V | 3/3 | $1.24 | 3/3 | 3/3 | 190.3 | 602,540 | 113,661 | 3 |
| 2 | `deepseek/deepseek-v4-pro-0813` | T | 3/3 | $0.52 | 3/3 | 3/3 | 310.2 | 608,433 | 62,039 | 3 |
| 3 | `~deepseek/deepseek-v4-flash-latest` | T | 3/3 | $0.13 | 3/3 | 3/3 | 527.4 | 1,396,193 | 218,938 | 3 |
| 4 | `z-ai/glm-5.2` | T | 3/3 | $1.23 | 3/3 | 3/3 | 583.3 | 3,172,546 | 152,419 | 3 |
| 5 | `qwen/qwen3.8-27b` | V | 3/3 | $0.41 | 3/3 | 3/3 | 584.2 | 197,418 | 111,643 | 3 |
| 6 | `meta/muse-glimmer-30b` | V | 2/3 | - | 2/3 | 2/3 | 621.5 | 207,958 | 95,960 | 3 |
| 7 | `xiaomi/mimo-v2.5-pro` | T | 2/3 | - | 2/3 | 2/3 | 804.8 | 681,115 | 136,864 | 3 |
| 8 | `google/gemma-4-31b-it` | V | 1/3 | - | 2/3 | 2/3 | 182.5 | 173,458 | 38,560 | 3 |
| 9 | `z-ai/glm-5.3` | T | 1/3 | - | 1/2 | 1/2 | 1329.2 | 653,466 | 52,896 | 2 |
| 10 | `thinkingmachines/inkling-small` | V | 0/3 | - | 0/3 | 0/3 | 87.2 | 246,737 | 49,483 | 3 |
| 11 | `inclusionai/ling-3.0-flash` | V | 0/3 | - | 0/3 | 0/3 | 145.8 | 8,814,615 | 9,811 | 3 |
| 12 | `thinkingmachines/inkling` | V | 0/3 | - | 0/3 | 0/3 | 222.4 | 177,917 | 159,105 | 3 |
