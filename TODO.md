# Roadmap

## Run Harness in Docker Container

## Vision MiddleWare

We want to innovate and invent vision/image middleware that can be added to any text-only models. How this would work is as follows.
Our program can accept TEXT and IMAGES. If the model selected does not support images, we first send our image to the best vision model (all open-weight), with the best prompt at transcribing images.

subtask: find a way to benchmark transcribing performance by prompt by creating a harness where a model turns a transcription into an image which gets graded by similarity by another vision model like gemini. The text model can create its own prompt and use the results of the grader to keep tweaking and improving the prompt. The grader will use a variety of input images.

subtask: generate ideas for images to be used for transcribing; the purpose of such a middleware is to enable browser-use, computer-use, understanding what's on the computer screen. e.g. if shown an image of a 9x9 captcha, the middleware should transcribe what is contained in each box rather than the details. Another example is using UI.

## Alternative agentic harnesses

The benchmarker talks to an external harness over a subprocess CLI contract
(`--model MODEL --task TASK.json --inputs-dir DIR --output-dir DIR`). Any
program matching that contract can plug in, and since the task itself — every
prompt, plus the verifier the agent is given — is passed in as `--task`, a new
harness has no Sudoku code to duplicate. The default implementation lives in the sibling
[sudoku-agent-harness](../sudoku-agent-harness/) repo and uses
[smolagents](https://github.com/huggingface/smolagents).

Other harnesses worth considering, especially for the delegation benchmark
below:

- **[OpenHands](https://github.com/All-Hands-AI/OpenHands)** — a much beefier
  autonomous agent (sandboxed shell, editor, browser). Overkill for Sudoku but
  a natural fit if we want to test agents that plan multi-step work, browse,
  or execute delegation across tools. It uses LiteLLM so OpenRouter models
  work out of the box — wrapping it as a `sudoku-agent-harness`-compatible
  CLI would just be a thin adapter.
- **[SWE-agent](https://github.com/princeton-nlp/SWE-agent)** — narrower focus
  on repo-editing tasks; probably the right shape for a coding-delegation
  variant.

## Delegation Benchmark (planned)

The Sudoku benchmark measures **direct** intelligence in the digital world — the
model itself does perception + reasoning + execution end-to-end. But that isn't
the only shape intelligence takes. Deciding when *not* to be the driver — when
to route a task to a specialist model, tool, or process — is its own skill, and
arguably the more load-bearing one for real deployments.

Concrete example: a user delegates a coding task (which needs no vision) to a
coding model rather than doing it themselves. The intelligent move is knowing
that's the right delegation, not personally writing the code.

**Goal:** design a companion benchmark that tests how well a model handles
"specific work" — i.e. given a task, does it correctly:

- Recognize which specialist / tool / model is the right fit,
- Hand it off cleanly with the right context,
- Verify and integrate the returned work,
- Avoid doing work itself that it should have delegated?

**Open questions to resolve before implementing:**

- What task pool distinguishes "should delegate" vs "should do myself"? (Mix of
  coding tasks, vision tasks, math tasks, retrieval tasks?)
- What does the delegation surface look like — a tool-use API where sub-models
  are exposed as callable tools?
- Grading axes: correctness of the final answer, correctness of the routing
  decision, cost-effectiveness of the routing (did it over-delegate or
  under-delegate?), latency.
