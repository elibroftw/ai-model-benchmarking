SEED := "42"
N := "3"

BENCH := "uv run cli/run_benchmark.py"

HARNESS := "uv run --project ./sudoku-agent-harness sudoku-agent-harness"
HARNESS_ID := "smolagents"

SUMMARIZE := "uv run cli/summarize_results.py"
TRIAL_TABLE := "uv run cli/trial_table.py"

# Show the available recipes.
default:
    @just --list

_preliminary *FLAGS:
    {{BENCH}} \
        --harness-cmd "{{HARNESS}}" \
        --harness-id {{HARNESS_ID}} \
        --seed {{SEED}} \
        --n-puzzles {{N}} \
        -v {{FLAGS}}

# Preliminary run: the small seeded set, straight to the models.
preliminary *FLAGS: (_preliminary " " + FLAGS)

# Preliminary run through the vision middleware (alt text for every puzzle).
preliminary-v *FLAGS: (_preliminary "--vision-middleware --fresh " + FLAGS)

# The same set for ONE model, by full id or unique substring: `just one glm-5.3-flash`
one MODEL *FLAGS: (_preliminary "--model " + MODEL + " --fresh" + FLAGS)

# One model through the vision middleware. e.g. `just one-v glm-5.3-flash`
one-v MODEL *FLAGS: (_preliminary "--model " + MODEL + " --vision-middleware --fresh " + FLAGS)

# test the vision-middleware by running the example file
v-mw-test *FLAGS:
    uv run vision-middleware/example.py {{FLAGS}}

# quick benchmark of open-weight vision-models
v-mw-bench:
    uv run vision-middleware/compare_vision_models.py

# Rebuild the leaderboard from results/, verifying every solution PNG locally.
summary *FLAGS:
    {{SUMMARIZE}} {{FLAGS}}

# The same report as a pasteable markdown table.
summary-md *FLAGS:
    {{SUMMARIZE}} --format markdown --hide-errors {{FLAGS}}

# The same report without the image-verification pass — faster, and correctness
# then rests on the run's own grader verdicts.
summary-quick *FLAGS:
    {{SUMMARIZE}} --no-verify-images {{FLAGS}}

# Summarize and (re)write results/leaderboard-<harness>.json.
leaderboard *FLAGS:
    {{SUMMARIZE}} --write-leaderboard {{FLAGS}}

# Markdown table for a trial file (default: trials/seed-42-n3-mixed.json).
trial-table *ARGS:
    {{TRIAL_TABLE}} {{ARGS}}
