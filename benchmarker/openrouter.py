"""Minimal OpenRouter client used by the grader (LLM-as-grader for output images)."""
import base64

import httpx

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

GRADER_PROMPT = """You are looking at a rendered image of a 9x9 Sudoku grid.
Extract the digits into a 9x9 JSON matrix in row-major order.

- For every empty cell, use 0.
- For every filled cell, use the digit 1-9 that appears there.
- Do NOT solve the puzzle. Only report what you see.

Return ONLY a JSON object of the form:
{"grid": [[r1c1,...,r1c9], [r2...], ..., [r9c1,...,r9c9]]}
"""


async def extract_grid(client, image_bytes, grader_model, timeout=120):
    """Send an image to a cheap vision model and ask it to transcribe the 9x9 grid.

    Returns the raw response text; `grader.py` parses it into a matrix.
    """
    b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": grader_model,
        "messages": [
            {"role": "system", "content": GRADER_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this Sudoku grid."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ],
    }
    try:
        resp = await client.post(CHAT_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise RuntimeError(
                f"OpenRouter returned 404 for model '{grader_model}'. "
                f"This usually means the model ID is not recognised. "
                f"Check https://openrouter.ai/models for the correct ID. "
                f"(Raw: {e})"
            ) from e
        raise RuntimeError(
            f"OpenRouter HTTP {e.response.status_code} for model "
            f"'{grader_model}': {e}"
        ) from e
    data = resp.json()
    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"No choices in grader response: {data}")
    return data["choices"][0]["message"]["content"]
