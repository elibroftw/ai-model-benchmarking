"""Sudoku puzzle generation with unique-solution guarantee."""
import random
from copy import deepcopy


def _find_empty(grid):
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                return r, c
    return None


def _is_valid(grid, r, c, num):
    for i in range(9):
        if grid[r][i] == num or grid[i][c] == num:
            return False
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            if grid[i][j] == num:
                return False
    return True


def _count_solutions(grid, limit=2):
    """Count solutions up to `limit`. Used to enforce uniqueness."""
    empty = _find_empty(grid)
    if empty is None:
        return 1
    r, c = empty
    count = 0
    for num in range(1, 10):
        if _is_valid(grid, r, c, num):
            grid[r][c] = num
            count += _count_solutions(grid, limit - count)
            grid[r][c] = 0
            if count >= limit:
                return count
    return count


def _fill_grid(grid):
    empty = _find_empty(grid)
    if empty is None:
        return True
    r, c = empty
    nums = list(range(1, 10))
    random.shuffle(nums)
    for num in nums:
        if _is_valid(grid, r, c, num):
            grid[r][c] = num
            if _fill_grid(grid):
                return True
            grid[r][c] = 0
    return False


def generate_puzzle(n_clues=32, max_attempts=200):
    """Generate a Sudoku with a unique solution and approximately `n_clues` givens.

    Returns (puzzle, solution) as 9x9 lists of ints (0 = blank in puzzle).
    Falls back to more clues than requested if uniqueness prevents further removal.
    """
    solution = [[0] * 9 for _ in range(9)]
    _fill_grid(solution)

    puzzle = deepcopy(solution)
    positions = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(positions)

    target_removals = 81 - n_clues
    removed = 0
    attempts = 0
    for r, c in positions:
        if removed >= target_removals or attempts >= max_attempts:
            break
        attempts += 1
        backup = puzzle[r][c]
        puzzle[r][c] = 0
        test = deepcopy(puzzle)
        if _count_solutions(test, limit=2) != 1:
            puzzle[r][c] = backup
        else:
            removed += 1

    return puzzle, solution
