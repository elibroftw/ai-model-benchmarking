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
preliminary: (_preliminary)

# Preliminary run through the vision middleware (alt text for every puzzle).
preliminary-v: (_preliminary "--vision-middleware --fresh")

# Rebuild the leaderboard from results/, verifying every solution PNG locally.
summarize *FLAGS:
    {{SUMMARIZE}} {{FLAGS}}

# The same report as a pasteable markdown table.
summarize-md *FLAGS:
    {{SUMMARIZE}} --format markdown --hide-errors {{FLAGS}}

# The same report without the image-verification pass — faster, and correctness
# then rests on the run's own grader verdicts.
summarize-quick *FLAGS:
    {{SUMMARIZE}} --no-verify-images {{FLAGS}}

# Summarize and (re)write results/leaderboard-<harness>.json.
leaderboard *FLAGS:
    {{SUMMARIZE}} --write-leaderboard {{FLAGS}}

# Markdown table for a trial file (default: trials/seed-42-n3-mixed.json).
trial-table *ARGS:
    {{TRIAL_TABLE}} {{ARGS}}
