"""Parse the grader-LLM's grid transcription and verify the Sudoku solution.

The benchmarker hands the harness's output PNG to a cheap vision model
(the "grader LLM"), which transcribes what it sees back into a 9x9 matrix.
This module parses that transcription and verifies the puzzle.
"""
import json
import re


def _is_9x9_ints(grid, low=0, high=9):
    if not isinstance(grid, list) or len(grid) != 9:
        return False
    for row in grid:
        if not isinstance(row, list) or len(row) != 9:
            return False
        for cell in row:
            if not isinstance(cell, int) or not low <= cell <= high:
                return False
    return True


def parse_grader_response(text):
    """Extract a 9x9 grid (0-9) from the grader LLM's response.

    Returns the matrix or None if no valid grid was found. `0` means the
    grader saw an empty cell there.
    """
    if not text:
        return None

    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "")

    for candidate in _iter_json_values(cleaned):
        if isinstance(candidate, dict):
            for key in ("grid", "solution", "matrix", "cells"):
                if key in candidate and _is_9x9_ints(candidate[key]):
                    return candidate[key]
        if isinstance(candidate, list) and _is_9x9_ints(candidate):
            return candidate

    # Fallback: 81 digits (0-9) in reading order.
    digits = re.findall(r"[0-9]", cleaned)
    if len(digits) >= 81:
        it = iter(digits[:81])
        return [[int(next(it)) for _ in range(9)] for _ in range(9)]

    return None


def _iter_json_values(text):
    """Yield JSON values found by scanning for balanced { } or [ ] blocks."""
    for opener, closer in [("{", "}"), ("[", "]")]:
        i = 0
        while i < len(text):
            if text[i] != opener:
                i += 1
                continue
            depth = 0
            for j in range(i, len(text)):
                if text[j] == opener:
                    depth += 1
                elif text[j] == closer:
                    depth -= 1
                    if depth == 0:
                        chunk = text[i:j + 1]
                        try:
                            yield json.loads(chunk)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            else:
                break


def verify(puzzle, solution):
    """Grade a proposed solution against the original puzzle clues.

    `solution` is a 9x9 matrix of ints 0-9 as transcribed by the grader LLM.
    A `0` in the solution means the model left that cell blank (or the grader
    couldn't read it); it counts as incomplete.
    """
    result = {
        "clues_preserved": True,
        "clues_violated": [],
        "sudoku_valid": True,
        "violations": [],
        "complete": True,
        "blank_cells": [],
    }

    if not _is_9x9_ints(solution, low=0, high=9):
        return {
            **result,
            "clues_preserved": False,
            "sudoku_valid": False,
            "complete": False,
            "correct": False,
            "error_type": "GRADER_PARSE_ERROR",
        }

    for r in range(9):
        for c in range(9):
            if solution[r][c] == 0:
                result["complete"] = False
                result["blank_cells"].append([r, c])
            elif puzzle[r][c] != 0 and solution[r][c] != puzzle[r][c]:
                result["clues_preserved"] = False
                result["clues_violated"].append([r, c])

    if result["complete"]:
        for r in range(9):
            if len(set(solution[r])) != 9:
                result["sudoku_valid"] = False
                result["violations"].append(f"row {r}")
        for c in range(9):
            col = [solution[r][c] for r in range(9)]
            if len(set(col)) != 9:
                result["sudoku_valid"] = False
                result["violations"].append(f"col {c}")
        for br in range(3):
            for bc in range(3):
                box = [solution[br * 3 + i][bc * 3 + j]
                       for i in range(3) for j in range(3)]
                if len(set(box)) != 9:
                    result["sudoku_valid"] = False
                    result["violations"].append(f"box {br}{bc}")
    else:
        result["sudoku_valid"] = False

    result["correct"] = (
        result["complete"] and result["clues_preserved"] and result["sudoku_valid"]
    )
    if result["correct"]:
        result["error_type"] = "COMPLETE"
    elif not result["complete"]:
        result["error_type"] = "INCOMPLETE"
    elif not result["clues_preserved"]:
        result["error_type"] = "CLUE_VIOLATION"
    else:
        result["error_type"] = "REASONING_ERROR"
    return result
