"""Wrapper around Pascal Pons' perfect Connect 4 solver.

The solver is a separate C++ program in external/connect4. It reads positions on
stdin, one move sequence per line (columns 1-7), and prints the sequence back with
its score. Positive means the player to move wins, and the size of the number says
how quickly.

The test sets at blog.gamesolver.org give the score of each position but not of
each move, so working out whether a move was a mistake means solving the position
that follows it. That's what solve_many() is for - one process handles thousands
of positions, since starting it costs a second or so to load the 32MB opening book.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOLVER = ROOT / "external" / "connect4" / "c4solver.exe"
BOOK = ROOT / "external" / "connect4" / "7x6.book"

# Test set filename -> the label used in the plots. L1 is the opening, L3 the
# endgame, which is the opposite of what the names suggest.
TEST_SETS = {
    "Test_L1_R1": "Beginning - Easy",
    "Test_L1_R2": "Beginning - Medium",
    "Test_L1_R3": "Beginning - Hard",
    "Test_L2_R1": "Middle - Easy",
    "Test_L2_R2": "Middle - Medium",
    "Test_L3_R1": "End - Easy",
}

TEST_DIR = ROOT / "external" / "testsets"


def available() -> bool:
    """Whether the solver has been built and the opening book downloaded."""
    return SOLVER.exists() and BOOK.exists()


def solve_many(positions: list[str], timeout: int = 3600) -> list[int]:
    """Score a list of move sequences. Returns one score per position, in order.

    An empty string means the empty board, which the solver rejects, so callers
    should not pass one.
    """
    if not available():
        raise RuntimeError(
            f"solver not built. Expected {SOLVER} and {BOOK}. "
            "See external/README.md."
        )
    if not positions:
        return []

    result = subprocess.run(
        [str(SOLVER)],
        input="\n".join(positions) + "\n",
        capture_output=True,
        text=True,
        cwd=str(SOLVER.parent),
        timeout=timeout,
    )

    scores = []
    for line in result.stdout.splitlines():
        parts = line.split()
        # The first line is the opening-book banner, and any position the solver
        # rejects prints an error instead of a score.
        if len(parts) >= 2 and parts[0] in ("", *positions[:0]) or len(parts) < 2:
            continue
        try:
            scores.append(int(parts[1]))
        except ValueError:
            continue

    if len(scores) != len(positions):
        raise RuntimeError(
            f"solver returned {len(scores)} scores for {len(positions)} positions. "
            f"stderr: {result.stderr[:200]}"
        )
    return scores


def load_test_set(name: str) -> list[tuple[str, int]]:
    """Read a test file as (move sequence, score) pairs."""
    path = TEST_DIR / name
    tokens = path.read_text().split()
    return [(tokens[i], int(tokens[i + 1])) for i in range(0, len(tokens), 2)]


def legal_columns(moves: str) -> list[int]:
    """Columns (1-7) that still have room after playing `moves`."""
    heights = [0] * 7
    for ch in moves:
        heights[int(ch) - 1] += 1
    return [c + 1 for c in range(7) if heights[c] < 6]
