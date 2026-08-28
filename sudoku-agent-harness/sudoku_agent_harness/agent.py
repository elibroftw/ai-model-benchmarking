"""smolagents-based agentic harness for file-in / file-out tasks.

The harness is deliberately task-agnostic: what the agent is asked to do
arrives as a task spec (`--task`), a JSON file naming the prompts, the input
and output filenames, and any files to install in the agent's working
directory. Nothing about any particular task lives here, so a different
harness can serve the same task by reading the same spec instead of
re-implementing it.

The harness is invoked ONCE per model with a directory of inputs and drives
the agent through them sequentially, using the SAME CodeAgent and the SAME
working directory across all rounds. `reset=False` on subsequent runs so
that:

- the model sees prior turns in its conversation (episodic memory),
- the Python interpreter state persists (variables, helper functions),
- files the agent writes to the working directory persist round-to-round.

The first round is framed as a warmup: it gets the spec's `first_round`
prompt, which is where a task puts its full rules and any encouragement to
build reusable infrastructure. Later rounds get the terser `next_round`
prompt so we don't repeatedly re-inflate the system context.

Per-round stats are emitted as JSONL to stdout so the caller can stream
progress in real time. All smolagents console output — including the live
token stream from `stream_outputs=True` — is routed to stderr so stdout
stays a clean, machine-readable JSONL channel.
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image


# Placeholders a task prompt may use; everything else is literal text.
# Substituted by plain replacement, never str.format, so a prompt containing
# braces (a JSON example, a dict literal) cannot break a run.
PROMPT_PLACEHOLDERS = (
    "round", "n_rounds", "input_filename", "output_filename", "transcription",
)

DEFAULT_INPUT_FILENAME = "input.png"
DEFAULT_OUTPUT_FILENAME = "output.png"
DEFAULT_INPUT_GLOB = "*.png"


def load_task(path) -> dict:
    """Read a task spec, filling in defaults for everything optional.

    Required: `prompts.first_round` and `prompts.next_round`, each with a
    `vision` and/or `text_only` variant. A task that only supplies one variant
    gets it used for both, which is right for a task whose wording doesn't
    depend on whether the model can see.
    """
    path = Path(path)
    try:
        spec = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"could not read task spec {path}: {e}") from e
    if not isinstance(spec, dict):
        raise ValueError(f"task spec {path} is not a JSON object")

    prompts = spec.get("prompts")
    if not isinstance(prompts, dict) or not prompts.get("first_round"):
        raise ValueError(
            f"task spec {path} needs prompts.first_round (and normally "
            f"prompts.next_round)"
        )
    for phase in ("first_round", "next_round"):
        variants = prompts.get(phase) or prompts["first_round"]
        if isinstance(variants, str):
            variants = {"vision": variants, "text_only": variants}
        if not variants.get("vision") and not variants.get("text_only"):
            raise ValueError(
                f"task spec {path}: prompts.{phase} has neither a vision nor a "
                f"text_only variant"
            )
        prompts[phase] = variants

    spec.setdefault("task", path.stem)
    spec.setdefault("input_filename", DEFAULT_INPUT_FILENAME)
    spec.setdefault("output_filename", DEFAULT_OUTPUT_FILENAME)
    spec.setdefault("input_glob", DEFAULT_INPUT_GLOB)
    spec.setdefault("assets", [])
    # Optional map of input filename → a text description of that file, which
    # a prompt can place with {transcription}. The harness neither produces nor
    # interprets these; it just hands each round the entry for its own input.
    if not isinstance(spec.get("transcriptions"), dict):
        spec["transcriptions"] = {}
    return spec


def build_prompt(spec: dict, *, round_number: int, n_rounds: int,
                 first: bool, with_images: bool,
                 transcription: str | None = None) -> str:
    """Pick the prompt for this round and fill in the generic placeholders."""
    variants = spec["prompts"]["first_round" if first else "next_round"]
    template = variants.get("vision" if with_images else "text_only") or (
        variants.get("text_only") or variants.get("vision")
    )
    values = {
        "round": str(round_number),
        "n_rounds": str(n_rounds),
        "input_filename": spec["input_filename"],
        "output_filename": spec["output_filename"],
        "transcription": transcription or "",
    }
    for key in PROMPT_PLACEHOLDERS:
        template = template.replace("{" + key + "}", values[key])
    return template


def default_models_config() -> Path | None:
    """The models.toml to read [custom_models] from when none was named.

    The manifest belongs to the project being benchmarked, not to the harness:
    the benchmarker keeps one models.toml at its repo root listing the models,
    their prices and any custom endpoints, and it invokes the harness from
    there. So the working directory is checked first, then the harness repo
    itself, which keeps a standalone `sudoku-agent-harness --model ...` working
    from inside its own checkout.

    Returns None when neither exists — custom endpoints are optional, and a
    model ID with no entry simply goes to OpenRouter.
    """
    here = Path.cwd().resolve()
    # The working directory, then upwards to the enclosing repository — so
    # running the harness from a subdirectory finds the project's manifest.
    # Bounded at the repo root so an unrelated models.toml further up the
    # filesystem can never be picked up.
    for directory in (here, *here.parents):
        candidate = directory / "models.toml"
        if candidate.exists():
            return candidate
        if (directory / ".git").exists():
            break
    fallback = Path(__file__).resolve().parent.parent / "models.toml"
    return fallback if fallback.exists() else None


def default_archive_dir() -> Path:
    """`archive/` at the harness repo root, regardless of the caller's cwd."""
    return Path(__file__).resolve().parent.parent / "archive"


STATE_FILENAME = ".harness_state.json"


def _strip_step_images(memory_step, agent=None) -> None:
    """Drop the duplicate input image smolagents pins to every action step.

    smolagents builds each `ActionStep` with `observations_images=images`,
    where `images` is the exact list handed to `agent.run()`. Nothing in a
    CodeAgent session ever replaces it (that hook exists for screenshot
    agents), so the round's input image is re-attached to the conversation
    once per step: by step 9 the request carries nine byte-identical copies
    of the same PNG and providers refuse it outright — DeepInfra caps a
    request at 8 images — and long before that the duplicates dominate the
    input tokens.

    The agent has already been shown this image in the round's task message,
    which stays in memory, so the per-step copies add nothing. Registered as
    a step callback, this clears each one as its step finishes, before the
    next model call reads memory. That holds a round to exactly one image no
    matter how many steps it takes.
    """
    if getattr(memory_step, "observations_images", None):
        memory_step.observations_images = None


def _drop_stale_images(agent) -> int:
    """Strip images from prior rounds out of the agent's memory.

    The session is deliberately persistent (`reset=False`), so every round's
    input image would otherwise stay in the conversation for the rest of the
    run, costing image tokens on every later request for a puzzle already
    answered. `_strip_step_images` caps a single round at one image; this
    keeps the count at one across rounds too.

    Only the pixels are dropped. Whatever the agent wrote about each input in
    its reasoning is untouched, so it keeps the session memory the benchmark
    is trying to measure.

    Call immediately before `agent.run()`: every image still in memory at
    that point belongs to an earlier round. Returns how many were dropped.
    """
    dropped = 0
    try:
        steps = agent.memory.steps
    except AttributeError:
        return 0
    for step in steps:
        for attr in ("task_images", "observations_images"):
            imgs = getattr(step, attr, None)
            if imgs:
                dropped += len(imgs)
                setattr(step, attr, None)
    return dropped


def _is_no_image_support_error(exc) -> bool:
    """True if a model rejected the request because it cannot accept images.

    OpenRouter answers 404 "No endpoints found that support image input".
    Other providers word it differently, so match on the substance rather
    than an exact string.
    """
    msg = str(exc).lower()
    return (
        "no endpoints found that support image input" in msg
        or ("image" in msg and "not support" in msg)
        or ("image" in msg and "unsupported" in msg)
    )


# How many times a round is re-attempted after a transient provider failure,
# and the base delay between attempts (grows with each retry).
TRANSIENT_RETRIES = 2
TRANSIENT_RETRY_DELAY = 5

# NOTE: do not pass `num_retries` to LiteLLMModel. litellm implements retries
# with tenacity, which is not a dependency here, so it fails the generation
# outright with "tenacity import failed" — every round of every model dies in
# milliseconds. The round-level retry above covers the same failures without
# needing it.


# Where the temperature actually lives is `[harness].temperature` in the
# calling project's models.toml (see `_config_temperature`): it is a property
# of the benchmark being run, not of this runner. This is only the fallback
# for a run with no manifest at all — pinning something beats inheriting a
# provider default that can change under the results.
DEFAULT_TEMPERATURE = 0.1

# Distinguishes "the caller said nothing" from "the caller said None", which
# means send no temperature at all. Only the first consults the manifest.
FROM_CONFIG = object()


def _config_temperature(config_path):
    """`[harness].temperature` from the models.toml, or the fallback.

    Accepts a number, or "none"/"default"/"provider" for send-nothing. A value
    that is neither is reported and ignored rather than raising: a typo in an
    optional setting should not cost a whole session.
    """
    import tomllib

    if config_path is None:
        return DEFAULT_TEMPERATURE
    try:
        data = tomllib.loads(Path(config_path).read_text())
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"[harness] could not read {config_path}: {e}", file=sys.stderr)
        return DEFAULT_TEMPERATURE
    section = data.get("harness")
    if not isinstance(section, dict) or "temperature" not in section:
        return DEFAULT_TEMPERATURE
    value = section["temperature"]
    if isinstance(value, str):
        if value.strip().lower() in ("none", "default", "provider"):
            return None
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    print(
        f"[harness] ignoring [harness].temperature={value!r} in {config_path}: "
        f"expected a number or \"none\"",
        file=sys.stderr,
    )
    return DEFAULT_TEMPERATURE


def _is_temperature_unsupported_error(exc) -> bool:
    """True if the model rejected the request because of `temperature`.

    Reasoning models commonly accept only their own default — OpenAI answers
    "Unsupported value: 'temperature' does not support 0.1 with this model.
    Only the default (1) value is supported." Matched on the substance, since
    every provider words it differently; a request that failed for some other
    reason must not lose the setting.
    """
    msg = str(exc).lower()
    if "temperature" not in msg:
        return False
    return any(
        phrase in msg
        for phrase in (
            "unsupported", "not support", "only the default", "not allowed",
            "cannot be used", "invalid", "unrecognized",
        )
    )


def _is_transient_provider_error(exc) -> bool:
    """True if a generation failed for a reason that may not recur.

    OpenRouter reports an upstream provider dropping out by injecting an error
    object into the SSE stream, which litellm surfaces as a
    `MidStreamFallbackError` wrapping "provider_unavailable" — nothing about
    the request is wrong, and the same prompt often succeeds moments later on
    another provider serving the same weights. Matched on the substance, so
    the wording of any one layer can change without silently disabling the
    retry.

    Deliberately narrow: a model that is genuinely unusable, a bad key, or a
    refusal must fail the round rather than be re-attempted for nothing.
    """
    msg = str(exc).lower()
    transient = (
        "provider_unavailable",
        "midstreamfallbackerror",
        "error injected into sse stream",
        "service unavailable",
        "temporarily unavailable",
        "overloaded",
        "internal server error",
        "bad gateway",
        "gateway timeout",
        "connection error",
        "timeout error",
    )
    return any(t in msg for t in transient)


def _preload_imaging(agent) -> None:
    """Pre-bind PIL into the agent's interpreter.

    smolagents' interpreter does not bind the root package for
    `import PIL.Image`, so the natural spelling raises "The variable `PIL` is
    not defined" — a failure mode that cost real rounds. Seeding the modules
    means both `PIL.Image.open(...)` and `from PIL import Image` work, and the
    agent need not spend a step importing at all. Best-effort.
    """
    try:
        import PIL, PIL.Image, PIL.ImageDraw, PIL.ImageFont, PIL.ImageOps  # noqa: F401
        agent.python_executor.send_variables({
            "PIL": PIL,
            "Image": PIL.Image,
            "ImageDraw": PIL.ImageDraw,
            "ImageFont": PIL.ImageFont,
        })
    except Exception as e:  # noqa: BLE001 - convenience only
        print(f"[harness] could not preload PIL: {e!r}", file=sys.stderr)


# Environment variables through which a child process can reach the desktop
# session of whoever is running the benchmark. `subprocess` is an authorized
# import, so the agent can invoke `xdg-open` — or a browser, or a file manager
# — directly, without going anywhere near PIL. Stripped for the duration of a
# session so such a command has nothing to attach to. Nothing the harness
# itself needs reads any of them: the model API is HTTPS, and the task's
# verifier is plain Python over pixels.
DESKTOP_ENV_VARS = (
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "DBUS_SESSION_BUS_ADDRESS",
    "XDG_RUNTIME_DIR",
    "XDG_CURRENT_DESKTOP",
    "XDG_SESSION_TYPE",
)


def _disarm_image_viewers() -> None:
    """Stop the agent from opening image windows on the operator's desktop.

    `Image` is preloaded into the interpreter and `im.show()` is a method
    call, so neither the import allowlist nor the interpreter's guard on
    un-registered builtins stands in its way. PIL's Linux viewer chain ends
    in `subprocess.Popen(["xdg-open", path])`, which hands the file to the
    desktop's default application; pointed at a directory, `xdg-open` opens a
    file manager. An agent that cannot quite make out a cell reaches for
    "let me look at it", and windows start appearing on the machine running
    the benchmark — which is not the machine the agent is reasoning about.

    A viewer the agent cannot see returns it nothing, so this empties PIL's
    viewer registry and makes `show()` a no-op that says what happened.
    Patching the class and the module (not a copy) covers every image object,
    including any created before this ran. Best-effort.
    """
    try:
        import PIL.Image, PIL.ImageShow

        PIL.ImageShow._viewers.clear()

        def _no_show(self, title=None, **options):
            print(
                "[harness] Image.show() is disabled — the harness runs "
                "headless, so there is no display for it to reach. Save the "
                "image to a file instead.",
                file=sys.stderr,
            )
            return None

        PIL.Image.Image.show = _no_show
        PIL.ImageShow.show = lambda image, title=None, **options: False
    except Exception as e:  # noqa: BLE001 - hardening is best-effort
        print(f"[harness] could not disarm image viewers: {e!r}", file=sys.stderr)


def _detach_from_desktop() -> dict:
    """Drop the desktop session out of the environment children inherit.

    Returns the previous values, for `_restore_env` to put back when the
    session ends. Covers the `subprocess` route that `_disarm_image_viewers`
    cannot: without a display, a session bus, or a runtime dir, `xdg-open`
    and friends have nothing to hand a file to.
    """
    saved: dict = {}
    for name in DESKTOP_ENV_VARS:
        if name in os.environ:
            saved[name] = os.environ.pop(name)
    # Anything honoring BROWSER (webbrowser, xdg-open's fallback) gets a
    # command that succeeds at doing nothing.
    saved["BROWSER"] = os.environ.get("BROWSER")
    os.environ["BROWSER"] = "/bin/false"
    return saved


def _restore_env(saved: dict) -> None:
    """Put back what `_detach_from_desktop` removed."""
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _allow_open(agent) -> None:
    """Make the builtin `open` callable inside the agent's interpreter.

    smolagents' interpreter exposes only an allowlist of builtins
    (`BASE_PYTHON_TOOLS`), and `open` is not on it. The obvious way to save a
    scratch file — `with open("notes.txt", "w") as f:` — therefore dies with
    "Forbidden function evaluation: 'open' is not among the explicitly allowed
    tools", and the agent spends a step rediscovering `pathlib`. Same class of
    paper cut as `_preload_imaging`.

    This grants nothing the session did not already have: `os`, `pathlib`,
    `shutil` and `subprocess` are all authorized imports, and a task that asks
    the agent to build reusable infrastructure is asking it to write files.

    Registered through `additional_functions`, not `send_variables`, for two
    reasons: `CodeAgent.run()` rebuilds the interpreter's tool table from that
    dict at the start of every round, so the registration survives the whole
    session; and a name merely present in interpreter state would still trip
    the separate guard against calling builtins that were never registered as
    tools. Best-effort.
    """
    try:
        executor = agent.python_executor
        executor.additional_functions["open"] = open
        # Effective immediately; run() would otherwise pick it up next round.
        executor.send_tools({**agent.tools, **agent.managed_agents})
    except Exception as e:  # noqa: BLE001 - convenience only
        print(f"[harness] could not authorize open(): {e!r}", file=sys.stderr)


def _install_assets(td: Path, assets) -> int:
    """Copy the task's files into the agent's working directory.

    Whatever a task wants on hand — a verifier it requires the agent to use, a
    reference document, a helper script. The harness neither knows nor cares
    what they are. Best-effort per file: a missing asset is reported and the
    session continues, since a round without the helper still produces a
    result worth recording.
    """
    installed = 0
    for asset in assets or []:
        src = Path(asset)
        try:
            shutil.copy(src, td / src.name)
            installed += 1
        except OSError as e:
            print(f"[harness] could not install asset {src}: {e}", file=sys.stderr)
    return installed


def _input_set_fingerprint(input_paths: list[Path]) -> str:
    """Content hash of the whole input set.

    Resuming is only safe against the identical set of inputs. Hashing the
    file contents (not just names) means regenerating the inputs invalidates
    the state even though the filenames are unchanged.
    """
    h = hashlib.sha256()
    for p in sorted(input_paths, key=lambda x: x.name):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _load_state(output_dir: Path, model_id: str, fingerprint: str) -> dict:
    """Return {item_name: round_record} for rounds already completed.

    Empty if there is no state, it belongs to another model, or the input set
    has changed since it was written.
    """
    state_path = output_dir / STATE_FILENAME
    if not state_path.exists():
        return {}
    try:
        state = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"[harness] ignoring unreadable state file: {e}", file=sys.stderr)
        return {}

    if _state_fingerprint(state) != fingerprint:
        print(
            "[harness] input set changed since last run - ignoring saved state "
            "and starting fresh.",
            file=sys.stderr,
        )
        return {}
    if state.get("model") != model_id:
        print(
            f"[harness] saved state is for {state.get('model')!r}, not "
            f"{model_id!r} - ignoring.",
            file=sys.stderr,
        )
        return {}

    done = {}
    for rec in state.get("rounds", []):
        name = _record_name(rec)
        # Only trust a record whose solution image is actually still there.
        if name and rec.get("success") and (output_dir / name).exists():
            done[name] = rec
    return done


def _state_fingerprint(state: dict):
    """Input-set hash from a state file, accepting the pre-rename key."""
    return state.get("input_set", state.get("puzzle_set"))


def _record_name(record: dict):
    """Input filename from a round record, accepting the pre-rename key."""
    return record.get("item_name") or record.get("puzzle_name")


def _save_state(output_dir: Path, model_id: str, fingerprint: str, rounds: list,
                task_name: str = "") -> None:
    """Persist completed rounds so an interrupted run can resume.

    Written atomically so a Ctrl+C mid-write can't leave a corrupt file.
    Best-effort: never raises.
    """
    try:
        state_path = output_dir / STATE_FILENAME
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "model": model_id,
                    "task": task_name,
                    "input_set": fingerprint,
                    "rounds": rounds,
                },
                indent=2,
            )
        )
        os.replace(tmp, state_path)
    except Exception as e:  # noqa: BLE001 - persistence is best-effort
        print(f"[harness] failed to save state: {e!r}", file=sys.stderr)


TRANSCRIPTIONS_SUBDIR = "transcriptions"


def _save_transcription(output_dir: Path, input_name: str, text: str):
    """Save the description this round's input came with, beside the outputs.

    The task spec's transcriptions arrive from whatever produced them — for
    this benchmark, a vision middleware — and are otherwise visible only
    inside the prompt of a session that has since ended. A run's timings and
    scores cannot be audited without them: whether the middleware helped, and
    whether it described the input correctly, is a question about this exact
    text. Written per round because a resumed session can mix rounds that got
    one with rounds that did not.

    Returns the path relative to `output_dir`, or None if it could not be
    written. Best-effort: never raises, since losing an audit copy must not
    fail the round that produced it.
    """
    try:
        dest_dir = output_dir / TRANSCRIPTIONS_SUBDIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{Path(input_name).stem}.txt"
        dest.write_text(text)
        return str(dest.relative_to(output_dir))
    except Exception as e:  # noqa: BLE001 - the audit copy is best-effort
        print(f"[harness] failed to save transcription for {input_name}: {e!r}",
              file=sys.stderr)
        return None


def _token_counts(agent) -> tuple[int, int]:
    """Session-cumulative (input, output) token counts from the agent's monitor.

    Returns (0, 0) if the monitor is unavailable or hasn't recorded anything,
    so token accounting can never break a run.
    """
    try:
        usage = agent.monitor.get_total_token_counts()
        return int(usage.input_tokens or 0), int(usage.output_tokens or 0)
    except Exception:  # noqa: BLE001 - metrics are best-effort
        return 0, 0


def _safe_model_name(model_id: str) -> str:
    """Turn `moonshotai/kimi-k3` into a single safe path segment."""
    return model_id.replace("/", "_").replace(":", "_")


def _archive_working_dir(td: Path, archive_dir: Path, model_id: str, rounds: list,
                         ignore_names=("input.png", "output.png")) -> Path | None:
    """Snapshot the agent's working directory for post-hoc inspection.

    Copies every file the agent left behind — its own scripts especially —
    plus a session.json of the round records. Any previous archive for this
    model is replaced so stale runs can't be mistaken for the current one.
    This is the audit trail for whatever the task forbids: the scripts are
    preserved so a caller can check how the answer was actually produced.

    Never raises: losing the archive must not fail an otherwise good session.
    """
    try:
        dest = Path(archive_dir).resolve() / _safe_model_name(model_id)
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # The round's input/output files are just scratch copies; the caller
        # already has both the inputs it supplied and the outputs it collected.
        shutil.copytree(
            td, dest,
            ignore=shutil.ignore_patterns(*ignore_names, "__pycache__"),
        )
        (dest / "session.json").write_text(
            json.dumps({"model": model_id, "rounds": rounds}, indent=2)
        )
        return dest
    except Exception as e:  # noqa: BLE001 - archiving is best-effort
        print(f"[harness] failed to archive working dir: {e!r}", file=sys.stderr)
        return None


class _RoundDeadline:
    """Per-round soft timeout, enforced between agent steps.

    smolagents' CodeAgent takes no `timeout` argument, so we register this as
    a step callback. It can only fire between steps — a single step that hangs
    inside a model call or subprocess will overrun it. The benchmarker's
    subprocess-level timeout is the hard backstop.
    """

    def __init__(self, seconds: int):
        self.seconds = seconds
        self.start: float | None = None

    def reset(self) -> None:
        self.start = time.perf_counter()

    def __call__(self, memory_step, agent=None) -> None:
        if self.start is None or self.seconds <= 0:
            return
        elapsed = time.perf_counter() - self.start
        if elapsed > self.seconds:
            raise TimeoutError(
                f"Round exceeded soft timeout of {self.seconds}s "
                f"(elapsed {elapsed:.1f}s)"
            )


def run(
    model_id: str,
    inputs_dir: Path,
    output_dir: Path,
    task: dict | str | Path,
    timeout: int,
    archive_dir: Path | None = None,
    fresh: bool = False,
    send_images: bool = True,
    models_config: Path | None = None,
    temperature=FROM_CONFIG,
) -> dict:
    """Run the agentic harness across a whole directory of task inputs.

    Args:
        model_id: OpenRouter model ID (e.g. "openai/gpt-4o"), or a key in the
            [custom_models] table of ``models_config``.
        inputs_dir: Directory of input files, one per round, taken in sorted
            order. Which files count is the task's `input_glob`.
        output_dir: Directory where each round's output is collected, under
            the same filename as its input.
        task: The task spec — a dict, or a path to the JSON file holding one.
            This is what makes the run a Sudoku benchmark, or anything else:
            the prompts, the filenames, and the files to install in the
            working directory. See `load_task`.
        timeout: Per-round soft timeout in seconds.
        archive_dir: Where to copy the agent's working directory when the
            session ends, under `<archive_dir>/<model>/`. Defaults to
            `archive/` at the harness repo root. Pass an explicit path to
            relocate it. This is the audit trail: every script the agent
            wrote is preserved, so a caller can check how the answers were
            actually produced against whatever its task forbids.
        fresh: Ignore any saved state and redo every round. By default a run
            resumes: rounds already completed for this model against this
            exact input set are skipped and their stored records replayed.
        send_images: Attach each round's input image to the message. Left on by
            default; if the model turns out to reject images the harness
            detects that on the first round and drops to text-only
            automatically, switching to the task's `text_only` prompts. Pass
            False to skip the failed attempt when you already know the model is
            text-only.
        temperature: Sampling temperature sent with every generation.
            Omitted (or ``FROM_CONFIG``) takes it from
            ``[harness].temperature`` in the models.toml below, which is where
            the benchmark pins it; None sends no temperature at all, leaving
            each provider's own default in force — which is what makes two
            runs incomparable, so prefer a number. A model that rejects an
            explicit temperature is detected on the round it happens and
            finishes the session without one.
        models_config: Path to a TOML file whose [custom_models] table maps
            model IDs to custom endpoint settings (provider, model_name,
            api_base, api_key). When ``model_id`` matches a key in that table
            the harness uses those settings to construct the LiteLLM model
            instead of the default OpenRouter path. Omit it to look for
            `models.toml` where `default_models_config` says — the caller's
            own manifest, then the harness's — which is how the benchmarker's
            top-level models.toml is picked up without being passed.

    Returns:
        dict with `success` (bool), `rounds` (list of per-round records),
        `n_solved` (int), `total_elapsed` (float), `error` (str, optional).
        Each round record carries `middleware` (whether the task spec supplied
        a transcription for that round's input) and `temperature` (what the
        round was actually generated at). Each round is also emitted as
        one JSONL line on stdout as it completes, and saved to the state file.
    """
    from rich.console import Console
    from smolagents import AgentLogger, CodeAgent, LiteLLMModel, LogLevel

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set (check .env file).")

    # Resolve custom model config before falling back to OpenRouter.
    if models_config is None:
        models_config = default_models_config()
    if temperature is FROM_CONFIG:
        temperature = _config_temperature(models_config)
    custom = _resolve_custom_model(model_id, models_config)
    if custom is not None:
        litellm_model_id = f"{custom['provider']}/{custom['model_name']}"
        litellm_api_key = custom["api_key"]
        litellm_api_base = custom["api_base"]
        print(
            f"[harness] custom model {model_id!r} -> "
            f"{custom['provider']}/{custom['model_name']} @ {custom['api_base']}",
            file=sys.stderr,
        )
    else:
        litellm_model_id = f"openrouter/{model_id}"
        litellm_api_key = api_key
        litellm_api_base = "https://openrouter.ai/api/v1"

    spec = task if isinstance(task, dict) else load_task(task)
    task_name = spec.get("task", "")
    input_name = spec["input_filename"]
    output_name = spec["output_filename"]

    inputs_dir = Path(inputs_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not inputs_dir.exists():
        raise FileNotFoundError(f"Inputs directory not found: {inputs_dir}")

    input_paths = sorted(inputs_dir.glob(spec["input_glob"]))
    if not input_paths:
        raise FileNotFoundError(
            f"No {spec['input_glob']} files found under {inputs_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = _input_set_fingerprint(input_paths)
    completed = {} if fresh else _load_state(output_dir, model_id, fingerprint)
    if completed:
        print(
            f"[harness] resuming: {len(completed)}/{len(input_paths)} round(s) "
            f"already done for {model_id}. Note the agent starts fresh, so it "
            f"cannot reuse infrastructure the interrupted run built.",
            file=sys.stderr,
        )

    model = LiteLLMModel(
        model_id=litellm_model_id,
        api_key=litellm_api_key,
        api_base=litellm_api_base,
        # Forwarded to the completion call. Omitted entirely when None, so the
        # request carries no `temperature` field rather than one the endpoint
        # has to interpret.
        **({} if temperature is None else {"temperature": temperature}),
    )
    if temperature is not None:
        print(f"[harness] temperature={temperature}", file=sys.stderr)
    deadline = _RoundDeadline(timeout)
    agent = CodeAgent(
        tools=[],
        model=model,
        additional_authorized_imports=[
            # Meta
            "importlib", # importlib.reload
            # Image manipulation
            "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont",
            "PIL.ImageOps", "PIL.ImageFilter",
            "numpy",
            # Filesystem / IO
            "os", "os.path", "io", "pathlib", "shutil", "glob", "tempfile", "posixpath",
            # Data handling
            "json", "base64", "re", "collections", "itertools", "math",
            # Shell / external tools
            "subprocess",
        ],
        max_steps=100,
        stream_outputs=True,
        # CodeAgent takes no `timeout` kwarg; the per-round budget is enforced
        # between steps by this callback instead.
        # `_strip_step_images` first, so the round's images are pruned even on
        # the step where the deadline fires (a raising callback stops the rest).
        step_callbacks=[_strip_step_images, deadline],
        # With stream_outputs=True, smolagents renders the live token stream
        # (and all its other logging) through its logger's rich Console, which
        # defaults to stdout. stdout is our JSONL stats channel, so send the
        # agent's terminal output to stderr instead.
        logger=AgentLogger(
            level=LogLevel.INFO,
            console=Console(file=sys.stderr, highlight=False),
        ),
    )

    # Hold onto the real stdout for JSONL records, then point sys.stdout at
    # stderr for the whole session. smolagents (and anything the agent's own
    # code prints) would otherwise corrupt the machine-readable stdout channel.
    _preload_imaging(agent)
    _allow_open(agent)
    _disarm_image_viewers()

    # May be flipped to False on the first round if the provider rejects
    # images; the agent is then told to read input.png itself.
    if not send_images:
        print(
            "[harness] text-only mode: agent must read input.png itself",
            file=sys.stderr,
        )

    jsonl_out = sys.stdout

    # ONE persistent working dir across all rounds.
    total_start = time.perf_counter()
    rounds = []
    interrupted = False
    # Rounds executed in THIS process, as opposed to replayed from state. The
    # first one gets the warmup prompt and resets the agent, because on resume
    # the agent has no memory of the interrupted run.
    executed = 0
    with tempfile.TemporaryDirectory(prefix="agent-harness-") as td:
        td = Path(td)
        cwd = os.getcwd()
        n_assets = _install_assets(td, spec.get("assets"))
        if n_assets:
            print(
                f"[harness] installed {n_assets} task asset(s) into the working dir",
                file=sys.stderr,
            )
        os.chdir(td)
        saved_env = _detach_from_desktop()
        sys.stdout = sys.stderr
        try:
            n_rounds = len(input_paths)
            for i, input_path in enumerate(input_paths):
                # Already done against this exact input set: replay the stored
                # record so the caller still sees every round.
                prior = completed.get(input_path.name)
                if prior is not None:
                    record = {**prior, "resumed": True}
                    rounds.append(record)
                    jsonl_out.write(json.dumps(record) + "\n")
                    jsonl_out.flush()
                    print(
                        f"[harness] round {i + 1}/{n_rounds}: "
                        f"{input_path.name} already done, skipping",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

                # Hand the agent this round's input under the task's name.
                input_dst = td / input_name
                output_src = td / output_name
                if output_src.exists():
                    output_src.unlink()
                shutil.copy(input_path, input_dst)

                item_id = _item_id_from_name(input_path)
                first_executed = executed == 0

                # A description of this round's input, if the task spec
                # carries one, for prompts that use {transcription}.
                transcription_text = spec["transcriptions"].get(input_path.name)
                # Kept on disk as well as in the prompt: see
                # `_save_transcription`.
                transcription_file = (
                    _save_transcription(
                        output_dir, input_path.name, transcription_text
                    )
                    if transcription_text is not None else None
                )

                def _build_prompt(with_images):
                    return build_prompt(
                        spec,
                        round_number=i + 1,
                        n_rounds=n_rounds,
                        first=first_executed,
                        with_images=with_images,
                        transcription=transcription_text,
                    )

                prompt = _build_prompt(send_images)

                img = Image.open(input_dst)
                deadline.reset()
                # The monitor only resets on round 1 (reset=True), so its
                # counts accumulate across the session; diff them per round.
                tokens_before = _token_counts(agent)
                start = time.perf_counter()
                error = None
                final_answer = None
                print(
                    f"\n===== Round {i + 1}/{n_rounds}: "
                    f"{input_path.name} =====",
                    file=sys.stderr,
                    flush=True,
                )
                # Prior rounds' input images would otherwise pile up in the
                # conversation, one per round, for the rest of the session.
                if send_images and not first_executed:
                    n_dropped = _drop_stale_images(agent)
                    if n_dropped:
                        print(
                            f"[harness] dropped {n_dropped} image(s) from earlier rounds",
                            file=sys.stderr,
                            flush=True,
                        )

                attempt = 0
                try:
                    for attempt in range(TRANSIENT_RETRIES + 1):
                        try:
                            final_answer = agent.run(
                                prompt,
                                # Reset only on the first round executed in
                                # this process; later rounds keep the session
                                # going.
                                reset=first_executed,
                                **({"images": [img]} if send_images else {}),
                            )
                            break
                        except Exception as e:
                            if (model.kwargs.get("temperature") is not None
                                    and _is_temperature_unsupported_error(e)):
                                # This model accepts only its own default. Drop
                                # the setting for the rest of the session and
                                # re-run the round; the alternative is failing
                                # every round of a model that works fine.
                                model.kwargs.pop("temperature", None)
                                print(
                                    f"[harness] model rejects "
                                    f"temperature={temperature}; continuing "
                                    f"with the provider's default for the rest "
                                    f"of this session",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                deadline.reset()
                                final_answer = agent.run(
                                    prompt,
                                    reset=first_executed,
                                    **({"images": [img]} if send_images else {}),
                                )
                                break
                            if send_images and _is_no_image_support_error(e):
                                # Text-only model: drop the attachment for the
                                # rest of the session and retry this round with
                                # the task's text-only prompt, which tells the
                                # agent to read the input file itself.
                                send_images = False
                                print(
                                    f"[harness] model cannot accept images; "
                                    f"switching to text-only mode (agent must "
                                    f"read {input_name} itself)",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                deadline.reset()
                                final_answer = agent.run(
                                    _build_prompt(False), reset=first_executed
                                )
                                break
                            # A provider dropping out mid-generation is not the
                            # model failing the task; re-attempting the round
                            # costs tokens but is the difference between a data
                            # point and a hole in the run.
                            if (attempt < TRANSIENT_RETRIES
                                    and _is_transient_provider_error(e)):
                                delay = TRANSIENT_RETRY_DELAY * (attempt + 1)
                                print(
                                    f"[harness] transient provider failure on "
                                    f"{input_path.name} "
                                    f"(attempt {attempt + 1}/"
                                    f"{TRANSIENT_RETRIES + 1}): "
                                    f"{type(e).__name__}: {e}\n"
                                    f"[harness] retrying this round in {delay}s",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                time.sleep(delay)
                                deadline.reset()
                                continue
                            raise
                except KeyboardInterrupt:
                    # Let everything already finished persist, then stop.
                    interrupted = True
                    print(
                        f"\n[harness] interrupted during {input_path.name}; "
                        f"{len(rounds)} completed round(s) saved. Re-run the "
                        f"same command to resume.",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                except Exception as e:  # noqa: BLE001 - one bad round must not kill the whole session
                    error = f"{type(e).__name__}: {e}"
                executed += 1
                elapsed = time.perf_counter() - start

                dest = output_dir / input_path.name
                success = output_src.exists() and error is None
                if success:
                    shutil.move(str(output_src), str(dest))

                in_after, out_after = _token_counts(agent)
                round_in = max(0, in_after - tokens_before[0])
                round_out = max(0, out_after - tokens_before[1])

                record = {
                    "item_id": item_id,
                    "item_name": input_path.name,
                    "round": i + 1,
                    # Whether this round ran with a transcription supplied by
                    # the caller's vision middleware — i.e. whether the prompt
                    # carried alt text for the input or the model was left to
                    # read the image itself. Recorded per round because a
                    # resumed session can replay rounds run the other way, and
                    # the two are not comparable. A round record without this
                    # key predates the field.
                    "middleware": transcription_text is not None,
                    # Where the text that round was given is saved, relative
                    # to the output dir, so the record and the transcription
                    # it ran on stay together.
                    "transcription_file": transcription_file,
                    "transcription_chars": (
                        len(transcription_text)
                        if transcription_text is not None else None
                    ),
                    "success": success,
                    # >1 when a transient provider failure cost this round an
                    # extra attempt; its time and tokens are all counted here.
                    "attempts": attempt + 1,
                    # What this round was actually generated at: null once a
                    # model that refuses the setting has fallen back to the
                    # provider's own default.
                    "temperature": model.kwargs.get("temperature"),
                    "elapsed": elapsed,
                    "input_tokens": round_in,
                    "output_tokens": round_out,
                    "total_tokens": round_in + round_out,
                    "final_answer": _stringify(final_answer),
                }
                if error is not None:
                    record["error"] = error
                elif not success:
                    record["error"] = (
                        f"Agent finished without producing {output_name}"
                    )

                rounds.append(record)
                # Streaming stats: one JSONL line per round as it completes.
                # Written to the REAL stdout, bypassing the stderr redirect.
                jsonl_out.write(json.dumps(record) + "\n")
                jsonl_out.flush()
                # Persist after every round so Ctrl+C at any point keeps all
                # the work done so far.
                _save_state(output_dir, model_id, fingerprint, rounds, task_name)
        except KeyboardInterrupt:
            # Ctrl+C outside agent.run() (e.g. while copying files).
            interrupted = True
            print(
                f"\n[harness] interrupted; {len(rounds)} completed round(s) "
                f"saved. Re-run the same command to resume.",
                file=sys.stderr,
            )
        finally:
            _save_state(output_dir, model_id, fingerprint, rounds, task_name)
            sys.stdout = jsonl_out
            _restore_env(saved_env)
            os.chdir(cwd)
            # Must happen before the TemporaryDirectory context exits and
            # deletes everything, and in `finally` so a crashed session is
            # still archived (those are the runs worth inspecting).
            archived_to = _archive_working_dir(
                td,
                archive_dir if archive_dir is not None else default_archive_dir(),
                model_id,
                rounds,
                ignore_names=(input_name, output_name),
            )
            if archived_to is not None:
                print(f"[harness] archived working dir -> {archived_to}", file=sys.stderr)

    return {
        # An interrupted session did not cover the input set, so it is not a
        # successful run even though the rounds it did finish are saved.
        "success": not interrupted,
        "interrupted": interrupted,
        "rounds": rounds,
        "n_solved": sum(1 for r in rounds if r["success"]),
        "n_resumed": sum(1 for r in rounds if r.get("resumed")),
        "total_elapsed": time.perf_counter() - total_start,
        # Only counts this process's work; replayed rounds contribute nothing.
        "total_input_tokens": sum(r.get("input_tokens", 0) for r in rounds if not r.get("resumed")),
        "total_output_tokens": sum(r.get("output_tokens", 0) for r in rounds if not r.get("resumed")),
    }


def _item_id_from_name(path: Path) -> int | None:
    """Extract the trailing number from e.g. `puzzle_042.png`.

    Lets a caller correlate a round with the input it supplied without the
    harness knowing what the inputs are. None if the name has no such suffix.
    """
    stem = path.stem  # e.g. "puzzle_042"
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def _resolve_custom_model(model_id: str, config_path) -> dict | None:
    """Look up `model_id` in a TOML config's [custom_models] table.

    Returns a dict with keys ``provider``, ``model_name``, ``api_base``,
    ``api_key`` when the model is found, or None when it is not — the
    caller should fall back to the default OpenRouter path.

    Raises ValueError when the config file exists but is malformed, so the
    user knows their explicit ``--models-config`` is broken.
    """
    import tomllib

    if config_path is None:
        return None
    config_path = Path(config_path)
    if not config_path.exists():
        return None
    try:
        data = tomllib.loads(config_path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"could not parse {config_path}: {e}") from e
    custom = data.get("custom_models") if isinstance(data, dict) else None
    if not isinstance(custom, dict):
        return None
    entry = custom.get(model_id)
    if isinstance(entry, dict):
        return {
            "provider": entry.get("provider", "openai"),
            "model_name": entry["model_name"],
            "api_base": entry["api_base"],
            "api_key": entry.get("api_key", "not-needed"),
        }
    return None


def _stringify(x):
    if x is None:
        return None
    s = str(x)
    return s if len(s) <= 2000 else s[:2000] + "...[truncated]"
