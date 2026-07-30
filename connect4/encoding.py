"""Board -> network input encoding, in canonical (current-player) form.

The network sees every position from the perspective of the player about to move:

    plane 0  the mover's own pieces
    plane 1  the opponent's pieces

so a position is encoded identically whether it is R or Y to move. That is what
lets one network play both sides, and it means a training example is equally
useful no matter which colour produced it.

Encoding by fixed colour instead (an "R plane" and a "Y plane") forces the
network to learn two mirror-image copies of the same function, and halves the
value of every example. Getting this wrong does not raise — it just trains
slowly and plateaus low.

Everything here is numpy only; no torch. Conversion to a tensor happens at the
network boundary, so this module and its tests stay fast and dependency-free.
"""

import numpy as np

from connect4.board import Board, EMPTY, ROWS, COLS

# (planes, rows, cols) — channel-first, matching torch's Conv2d convention.
PLANES = 2
INPUT_SHAPE = (PLANES, ROWS, COLS)

# Policy target/output is one entry per column.
ACTION_SIZE = COLS


def encode(board: Board, player: str) -> np.ndarray:
    """Return `board` as a float32 array of shape INPUT_SHAPE, from `player`'s view.

    plane 0 is 1.0 where `player` has a piece, plane 1 is 1.0 where the opponent
    does, and 0.0 elsewhere. `player` is whoever is about to move.

    Note this takes the player explicitly rather than deriving it from
    board.last_player: a board built by Board(grid) has no history, and the
    caller always knows whose turn it is anyway.
    """
    # Initialize the encoding array
    encoded = np.zeros(INPUT_SHAPE, dtype=np.float32)
    # Fill in the planes based on the board's grid
    for row in range(ROWS):
        for col in range(COLS):
            cell = board.grid[row][col]
            if cell == player:
                encoded[0, row, col] = 1.0
            elif cell != EMPTY:
                encoded[1, row, col] = 1.0
    return encoded


def legal_move_mask(board: Board) -> np.ndarray:
    """Return a float32 mask of shape (ACTION_SIZE,): 1.0 for playable columns.

    Needed because the policy head emits seven logits regardless of whether the
    columns are playable. Illegal moves must be zeroed *before* renormalising,
    or probability leaks onto moves that cannot be made.
    """
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    for col in board.available_moves():
        mask[col] = 1.0
    return mask


def mask_and_normalise(priors: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out illegal moves in `priors` and renormalise to sum to 1.

    `priors` must be non-negative — i.e. probabilities, not raw logits. That is
    checked rather than assumed: feeding logits in produces negative "priors"
    that break PUCT's exploration term while the network trains normally, which
    is a miserable thing to track down.

    If every legal prior is 0 the result is uniform over the legal moves. An
    untrained or saturated network must not be able to crash the search — during
    early self-play the network is garbage by definition.
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
        # Assign uniform, don't add it — adding only happens to be correct when
        # `masked` is all zeros, and silently wrong otherwise.
        return (mask / mask.sum()).astype(np.float32)

    return masked / total


def decode_to_string(encoded: np.ndarray) -> str:
    """Render an encoded position back to text, for debugging.

    Use 'X' for the mover's pieces and 'O' for the opponent's — deliberately not
    R/Y, since the encoding has discarded which colour is which. If you find
    yourself wanting R/Y here, that's a sign the canonical form has leaked.
    """
    # Initialize the decoded string
    decoded = ""
    for row in range(ROWS):
        for col in range(COLS):
            if encoded[0, row, col] == 1.0:
                decoded += "X"
            elif encoded[1, row, col] == 1.0:
                decoded += "O"
            else:
                decoded += "."
        decoded += "\n"
    return decoded

