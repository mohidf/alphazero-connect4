"""Turning a board into the array the network reads.

Always from the point of view of whoever is about to move:

    plane 0  my pieces
    plane 1  their pieces

So the same situation encodes the same way whether R or Y is on the move, which
means one network can play both colours and every training example counts twice
as much. Using a fixed "R plane / Y plane" instead makes the network learn the
same thing twice.

numpy only here; the conversion to a tensor happens in network.py.
"""

import numpy as np

from connect4.board import Board, EMPTY, ROWS, COLS

PLANES = 2
INPUT_SHAPE = (PLANES, ROWS, COLS)
ACTION_SIZE = COLS


def encode(board: Board, player: str) -> np.ndarray:
    """Board as a (2, 6, 7) float32 array, seen from `player`.

    Takes the player explicitly because a board built from a grid has no history
    to read it from, and the caller knows whose turn it is anyway.
    """
    encoded = np.zeros(INPUT_SHAPE, dtype=np.float32)
    for row in range(ROWS):
        for col in range(COLS):
            cell = board.grid[row][col]
            if cell == player:
                encoded[0, row, col] = 1.0
            elif cell != EMPTY:
                encoded[1, row, col] = 1.0
    return encoded


def legal_move_mask(board: Board) -> np.ndarray:
    """1.0 for playable columns, 0.0 otherwise.

    The policy head always outputs seven numbers whether or not the columns are
    playable, so they have to be zeroed before anything is normalised.
    """
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    for col in board.available_moves():
        mask[col] = 1.0
    return mask


def mask_and_normalise(priors: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Drop illegal moves and rescale so the rest sum to 1.

    Priors have to be probabilities, not logits - logits are often negative and
    would quietly wreck PUCT's exploration term, so that's checked here.

    If every legal prior is zero we fall back to uniform. An untrained network
    outputs nonsense and shouldn't be able to crash the search.
    """
    if not mask.any():
        raise ValueError("cannot normalise priors: no legal moves in the mask")

    masked = np.where(mask > 0, priors, 0.0).astype(np.float32)

    if np.any(masked < 0):
        raise ValueError(
            "priors must be non-negative; got negative values "
            "(did you pass raw logits instead of softmax probabilities?)"
        )

    total = masked.sum()
    if total <= 0:
        return (mask / mask.sum()).astype(np.float32)

    return masked / total


def decode_to_string(encoded: np.ndarray) -> str:
    """Print an encoded position. X is the mover, O is the opponent - not R/Y,
    because the encoding has thrown away which colour is which."""
    rows = []
    for row in range(ROWS):
        line = ""
        for col in range(COLS):
            if encoded[0, row, col] == 1.0:
                line += "X"
            elif encoded[1, row, col] == 1.0:
                line += "O"
            else:
                line += "."
        rows.append(line)
    return "\n".join(rows) + "\n"
