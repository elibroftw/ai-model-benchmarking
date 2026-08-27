# Vision Middleware POC

`just v-mw-bench` results

The first model that passes human review is `gemma-4-26b-a4b-it` even if it is a few seconds slower. By human review I mean that the first two qwen models literally hallucinated a space, and the smaller qwen model uses english for transcribing instead of similar to how our script outputs the puzzle to console (the models are unaware of the ASCII art format). It also happens this model is cheaper than Qwen3 VL 32B Instruct.

1. qwen/qwen3-vl-32b-instruct  median   5.80s   mean   5.62s   min   4.00s   max   7.59s   (10/10 usable) $0.104 / $0.416
2. qwen/qwen3-vl-235b-a22b-instruct  median   7.06s   mean   8.54s   min   2.89s   max  20.80s   (10/10 usable)
3. google/gemma-4-26b-a4b-it         median   9.06s   mean  10.42s   min   2.41s   max  33.26s   (10/10 usable) # $0.042 / $0.22
4. qwen/qwen3-vl-30b-a3b-instruct    median  13.11s   mean  22.15s   min   4.38s   max  65.95s   (10/10 usable)
5. google/gemma-4-31b-it  median   7.01s   mean   7.81s   min   4.56s   max  17.89s   (10/10 usable) $0.09 / $0.34
6. qwen/qwen3.8-27b       median  26.06s   mean  28.58s   min  11.64s   max  49.18s   (10/10 usable)
7. z-ai/glm-5.3-flash     median  16.72s   mean  18.23s   min  12.29s   max  29.51s   (9/10 usable)

## Failed Models

### Misleading Output

- qwen/qwen3-vl-8b-instruct: no spaces in the rows transcribed.

### Useless output

- google/gemma-3-4b-it: 429
- thinkingmachines/inkling-small
- thinkingmachines/inkling
- minimax/minimax-m3
