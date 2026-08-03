"""How good can this architecture get if the data isn't the problem?

Self-play on one laptop GPU tops out around 26% error against the solver, and the
open question is whether that's the network's ceiling or just a data shortage. This
answers it by removing the data constraint entirely: label positions with the
perfect solver, train on those directly, and measure with the same benchmark.

It is deliberately not AlphaZero - the training signal comes from the solver rather
than from the agent's own games. The point isn't to build a better player this way,
it's to find out what error rate the architecture can reach at all, so the self-play
number has something to be compared against.

    python supervised_ceiling.py --generate 150000     # ~30 min, solver-bound
    python supervised_ceiling.py --train
    python supervised_ceiling.py --evaluate

Two things make the result trustworthy, and both are easy to get wrong:

Positions are excluded by board state, not by move sequence. Different move orders
transpose into the same position, so filtering on the string alone would leak test
positions into training through the back door.

Targets are built exactly the way solver_benchmark.build_ground_truth builds them -
solve every child, negate, take the argmax set. That benchmark scores the *fastest*
win, which is stricter than merely preserving the result. Training against a looser
definition than the one being measured would understate the ceiling.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from connect4 import arena, solver
from connect4.board import Board, PLAYER_R
from connect4.encoding import ACTION_SIZE, encode
from connect4.mcts import other
from connect4.network import Connect4Net, count_parameters, default_device, load, save
from connect4.train import losses, make_optimizer
from solver_benchmark import agent_errors, board_from_moves, load_ground_truth, policy_errors

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "supervised.npz"
CHECKPOINT_DIR = ROOT / "checkpoints" / "supervised"

# Test positions run from 1 to 41 plies, so training positions cover the same range.
MIN_PLIES, MAX_PLIES = 1, 40
SOLVE_CHUNK = 40_000


# ---------------------------------------------------------------- dataset


def state_key(board: Board, player: str) -> tuple:
    """Identity of a position, independent of how it was reached."""
    return (tuple(cell for row in board.grid for cell in row), player)


def forbidden_states() -> set[tuple]:
    """Every board state appearing in the test sets, to keep them out of training."""
    forbidden = set()
    for name in solver.TEST_SETS:
        for moves, _ in solver.load_test_set(name):
            board, player = board_from_moves(moves)
            forbidden.add(state_key(board, player))
    return forbidden


def sample_positions(count: int, rng: np.random.Generator) -> list[str]:
    """Random non-terminal positions, spread evenly over game length.

    Random play rather than played-out games: it covers the position space far more
    broadly, which is what a ceiling measurement wants. It does mean the training
    distribution isn't the distribution a real game visits - noted in the writeup.
    """
    forbidden = forbidden_states()
    print(f"excluding {len(forbidden):,} test-set states", flush=True)

    seen: set[tuple] = set()
    positions: list[str] = []
    attempts = 0

    while len(positions) < count:
        attempts += 1
        plies = int(rng.integers(MIN_PLIES, MAX_PLIES + 1))
        board, player, moves = Board(), PLAYER_R, []

        for _ in range(plies):
            if board.is_terminal():
                break
            col = int(rng.choice(board.available_moves()))
            board.make_move(col, player)
            moves.append(col + 1)
            player = other(player)
        else:
            if board.is_terminal():
                continue
            key = state_key(board, player)
            if key in forbidden or key in seen:
                continue
            seen.add(key)
            positions.append("".join(str(c) for c in moves))

            if len(positions) % 25_000 == 0:
                print(f"  sampled {len(positions):,} / {count:,}", flush=True)

    print(f"  {len(positions):,} unique positions from {attempts:,} attempts", flush=True)
    return positions


def solve_in_chunks(children: list[str]) -> list[int]:
    """Score every child, a chunk at a time.

    One solver call per chunk rather than one giant one: the process reloads the
    32MB book each time, but a million lines of stdin in one go is asking for
    trouble and this way progress is visible.
    """
    scores: list[int] = []
    started = time.perf_counter()
    for start in range(0, len(children), SOLVE_CHUNK):
        chunk = children[start:start + SOLVE_CHUNK]
        scores.extend(solver.solve_many(chunk))
        done = len(scores)
        rate = done / max(1e-9, time.perf_counter() - started)
        print(f"  solved {done:,} / {len(children):,}  ({rate:,.0f}/s, "
              f"~{(len(children) - done) / max(1e-9, rate) / 60:.0f}m left)", flush=True)
    return scores


# Anything below every real score, so illegal columns never win an argmax.
UNPLAYABLE = -10_000


def ending_move_value(board: Board, player: str, col: int, stones: int) -> int | None:
    """Score of a move that ends the game, or None if the game goes on.

    The solver refuses any sequence that carries on past a win, so winning moves
    have to be scored here instead. Its convention is that a score counts the
    winner's remaining stones: 22 minus however many the winner ends up having
    played. A player with `stones` already on the board will have played
    stones//2 + 1 once this move lands.
    """
    child = board.copy()
    child.make_move(col, player)
    if child.winner() is not None:
        return 22 - (stones // 2 + 1)
    if not child.available_moves():
        return 0  # filled the board without a winner
    return None


def build_targets(positions: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Policy and value targets, same definition the benchmark uses."""
    move_values = np.full((len(positions), ACTION_SIZE), UNPLAYABLE, dtype=np.int32)
    children: list[str] = []
    index: list[tuple[int, int]] = []  # which (position, column) each child came from

    for position_id, moves in enumerate(positions):
        board, player = board_from_moves(moves)
        for col in solver.legal_columns(moves):
            ending = ending_move_value(board, player, col - 1, len(moves))
            if ending is None:
                children.append(moves + str(col))
                index.append((position_id, col - 1))
            else:
                move_values[position_id, col - 1] = ending

    print(f"solving {len(children):,} child positions "
          f"({len(index):,} of {int((move_values > UNPLAYABLE).sum()) + len(index):,} moves "
          f"need the solver)", flush=True)
    scores = solve_in_chunks(children)

    # A move's value is the child's score negated - the child is scored for the
    # opponent, who moves there.
    for (position_id, col), child_score in zip(index, scores):
        move_values[position_id, col] = -child_score

    best = move_values.max(axis=1)
    optimal = move_values == best[:, None]
    policy = (optimal / optimal.sum(axis=1, keepdims=True)).astype(np.float32)
    # Sign only, to match what the self-play value head learns: a game result, not
    # the solver's how-many-moves-until-it-ends scale.
    value = np.sign(best).astype(np.float32)

    return policy, value


def generate(count: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    positions = sample_positions(count, rng)
    policy, value = build_targets(positions)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_PATH,
        moves=np.array(positions),
        policy=policy,
        value=value,
    )
    wins = int((value > 0).sum())
    draws = int((value == 0).sum())
    print(f"\nwrote {DATA_PATH}  ({len(positions):,} positions)")
    print(f"  player to move wins {wins:,}  draws {draws:,}  loses {len(positions)-wins-draws:,}")


# ---------------------------------------------------------------- training


def encode_dataset(moves: np.ndarray) -> np.ndarray:
    states = np.zeros((len(moves), 2, 6, 7), dtype=np.float32)
    for i, sequence in enumerate(moves):
        board, player = board_from_moves(str(sequence))
        states[i] = encode(board, player)
    return states


def mirror(states: np.ndarray, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left-right flip. The board is symmetric, so this is free extra data."""
    return states[:, :, :, ::-1].copy(), policy[:, ::-1].copy()


def train_one(
    channels: int,
    blocks: int,
    data: dict,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    fraction: float = 1.0,
) -> Path:
    torch.manual_seed(seed)
    net = Connect4Net(channels=channels, blocks=blocks).to(device)
    name = f"{channels}ch_{blocks}blk" + ("" if fraction == 1.0 else f"_{fraction:g}x")
    print(f"\n=== {name}: {count_parameters(net):,} parameters ===", flush=True)

    states, policy, value = data["states"], data["policy"], data["value"]
    # The validation split is taken from the full set first, so every run is scored
    # on the same held-out positions no matter how much training data it gets.
    split = int(0.95 * len(states))
    train_slice = slice(0, int(split * fraction))
    val_slice = slice(split, len(states))
    print(f"  {train_slice.stop:,} training positions, "
          f"{len(states) - split:,} held out", flush=True)

    # Augmentation on the training half only, so validation stays a clean holdout.
    flipped_states, flipped_policy = mirror(states[train_slice], policy[train_slice])
    train_states = torch.from_numpy(np.concatenate([states[train_slice], flipped_states]))
    train_policy = torch.from_numpy(np.concatenate([policy[train_slice], flipped_policy]))
    train_value = torch.from_numpy(np.concatenate([value[train_slice], value[train_slice]]))

    val_states = torch.from_numpy(states[val_slice]).to(device)
    val_policy = torch.from_numpy(policy[val_slice]).to(device)
    val_value = torch.from_numpy(value[val_slice]).to(device)

    optimizer = make_optimizer(net, learning_rate=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.3)
    steps = len(train_states) // batch_size
    rng = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        net.train()
        order = rng.permutation(len(train_states))
        running = 0.0
        for step in range(steps):
            picks = order[step * batch_size:(step + 1) * batch_size]
            batch_states = train_states[picks].to(device, non_blocking=True)
            policy_logits, predicted = net(batch_states)
            policy_loss, value_loss = losses(
                policy_logits,
                predicted,
                train_policy[picks].to(device),
                train_value[picks].to(device),
            )
            loss = policy_loss + value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
        scheduler.step()

        net.eval()
        with torch.no_grad():
            logits, predicted = net(val_states)
            policy_loss, value_loss = losses(logits, predicted, val_policy, val_value)
            # Same question the benchmark asks: is the top move an optimal one?
            chosen = logits.argmax(dim=-1)
            correct = val_policy[torch.arange(len(chosen), device=device), chosen] > 0
            accuracy = 100.0 * float(correct.float().mean())
            value_sign = 100.0 * float((predicted.sign() == val_value.sign()).float().mean())

        print(f"  epoch {epoch:>2}/{epochs}  train {running/steps:.4f}  "
              f"val p {float(policy_loss):.4f} v {float(value_loss):.4f}  "
              f"top1 {accuracy:5.2f}%  value sign {value_sign:5.2f}%", flush=True)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    out = CHECKPOINT_DIR / f"{name}.pt"
    save(net.eval(), out)
    print(f"  wrote {out}", flush=True)
    return out


def train(epochs: int, batch_size: int, seed: int) -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"no dataset at {DATA_PATH}. Run --generate first.")

    raw = np.load(DATA_PATH)
    print(f"loaded {len(raw['moves']):,} positions, encoding", flush=True)
    data = {
        "states": encode_dataset(raw["moves"]),
        "policy": raw["policy"],
        "value": raw["value"],
    }
    device = default_device()
    # The last one is the data-scaling check: if a quarter of the positions gets
    # the same error as all of them, the limit is the architecture, not the data.
    for channels, blocks, fraction in ((64, 4, 1.0), (128, 8, 1.0), (128, 8, 0.25)):
        train_one(channels, blocks, data, device, epochs, batch_size, seed, fraction)


# ---------------------------------------------------------------- evaluation


def evaluate(search_limit: int, simulations: int) -> None:
    truth = load_ground_truth()
    device = default_device()

    candidates = {
        "self-play 64ch (overnight2)": ROOT / "checkpoints" / "overnight2" / "best.pt",
        "self-play 128ch (big)": ROOT / "checkpoints" / "big" / "best.pt",
        "supervised 64ch": CHECKPOINT_DIR / "64ch_4blk.pt",
        "supervised 128ch": CHECKPOINT_DIR / "128ch_8blk.pt",
        "supervised 128ch (1/4 data)": CHECKPOINT_DIR / "128ch_8blk_0.25x.pt",
    }
    nets = {name: load(path, device) for name, path in candidates.items() if path.exists()}
    missing = [name for name, path in candidates.items() if not path.exists()]
    if missing:
        print(f"skipping (no checkpoint): {', '.join(missing)}\n", flush=True)

    header = f"{'test set':22}" + "".join(f"{n:>28}" for n in nets)

    print("=== policy head only, full test sets (error %) ===", flush=True)
    print(header, flush=True)
    for name, label in solver.TEST_SETS.items():
        positions = list(truth[name])
        cells = "".join(
            f"{policy_errors(net, positions, truth[name]):>28.1f}" for net in nets.values()
        )
        print(f"{label:22}{cells}", flush=True)

    print(f"\n=== with PUCT search, {simulations} sims, "
          f"first {search_limit} positions per set (error %) ===", flush=True)
    print(header, flush=True)
    for name, label in solver.TEST_SETS.items():
        positions = list(truth[name])[:search_limit]
        cells = "".join(
            f"{agent_errors(arena.network_agent(net, simulations), positions, truth[name]):>28.1f}"
            for net in nets.values()
        )
        print(f"{label:22}{cells}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", type=int, metavar="N", help="build a dataset of N positions")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--search-limit", type=int, default=200)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.generate:
        generate(args.generate, args.seed)
    if args.train:
        train(args.epochs, args.batch_size, args.seed)
    if args.evaluate:
        evaluate(args.search_limit, args.simulations)
    if not (args.generate or args.train or args.evaluate):
        parser.error("nothing to do - pass --generate, --train or --evaluate")


if __name__ == "__main__":
    main()
