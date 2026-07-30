"""Tests for PUCT.

The evaluator is just a callable, so these use stubs and no network at all. That
separates "is the search right" from "is the network any good".

Sign convention being tested throughout: a node's Q is from the point of view of
whoever moves there, so it flips at every level going back up.
"""

import numpy as np
import pytest

from connect4.board import Board, COLS, ROWS, PLAYER_R, PLAYER_Y
from connect4.mcts import other
from connect4 import puct
from connect4.puct import (
    C_PUCT,
    Node,
    add_dirichlet_noise,
    backup,
    expand,
    network_evaluator,
    policy_target,
    puct_move,
    run_search,
    select_child,
    select_move,
    terminal_value,
    uniform_evaluator,
    visit_counts,
)
from tests.test_board import play, play_alternating, DRAW_SEQUENCE, R, Y
from tests.test_minimax import r_has_won, r_can_win_in_one, y_can_win_in_one


def full_board() -> Board:
    return play_alternating(DRAW_SEQUENCE)


def constant_evaluator(value: float):
    """Flat priors, fixed value - isolates the backup sign from policy effects."""
    def evaluate(board: Board, player: str) -> tuple[np.ndarray, float]:
        return np.full(COLS, 1.0 / COLS, dtype=np.float32), value
    return evaluate


def biased_evaluator(favourite: int, value: float = 0.0):
    """Puts most prior mass on one column."""
    def evaluate(board: Board, player: str) -> tuple[np.ndarray, float]:
        priors = np.full(COLS, 0.01, dtype=np.float32)
        priors[favourite] = 1.0
        return priors / priors.sum(), value
    return evaluate


def counting_evaluator(value: float = 0.0):
    """Flat priors, and a call counter, to check how often the net is consulted."""
    calls = []

    def evaluate(board: Board, player: str) -> tuple[np.ndarray, float]:
        calls.append((board.move_count, player))
        return np.full(COLS, 1.0 / COLS, dtype=np.float32), value

    return evaluate, calls


# --------------------------------------------------------------------------
# terminal values
# --------------------------------------------------------------------------

def test_terminal_value_of_a_draw_is_zero():
    assert terminal_value(full_board(), PLAYER_R) == 0.0


def test_terminal_value_is_minus_one_for_the_player_on_the_move():
    """A win is created by the move that just happened, so whoever is on the move at a won
    position is the player who lost."""
    board = r_has_won()
    assert board.winner() == PLAYER_R
    assert terminal_value(board, PLAYER_Y) == -1.0


def test_terminal_value_is_plus_one_for_the_winner():
    """Kept for correctness rather than reachability - no legal search path arrives at a
    won position with the winner still to move."""
    assert terminal_value(r_has_won(), PLAYER_R) == 1.0


# --------------------------------------------------------------------------
# node statistics
# --------------------------------------------------------------------------

def test_unvisited_node_has_zero_q():
    """Not infinity, unlike UCB1 - the prior term is what drives exploration."""
    assert Node(Board(), PLAYER_R).q == 0.0


def test_q_is_the_mean_value():
    node = Node(Board(), PLAYER_R)
    node.visits, node.value_sum = 4, 2.0
    assert node.q == 0.5


def test_puct_score_prefers_the_higher_prior_at_equal_visits():
    parent = Node(Board(), PLAYER_R)
    parent.visits = 16
    low = Node(Board(), PLAYER_Y, parent=parent, move=0, prior=0.1)
    high = Node(Board(), PLAYER_Y, parent=parent, move=1, prior=0.8)
    parent.children = {0: low, 1: high}

    assert high.puct_score() > low.puct_score()


def test_puct_score_prefers_the_less_visited_child_at_equal_prior():
    parent = Node(Board(), PLAYER_R)
    parent.visits = 30

    fewer = Node(Board(), PLAYER_Y, parent=parent, move=0, prior=0.5)
    fewer.visits = 5
    more = Node(Board(), PLAYER_Y, parent=parent, move=1, prior=0.5)
    more.visits = 20
    parent.children = {0: fewer, 1: more}

    assert fewer.q == more.q == 0.0   # no value signal, so only exploration differs
    assert fewer.puct_score() > more.puct_score()


def test_puct_score_flips_the_sign_of_the_childs_q():
    """A child that is winning for its own mover is bad for the parent."""
    parent = Node(Board(), PLAYER_R)
    parent.visits = 20

    good_for_opponent = Node(Board(), PLAYER_Y, parent=parent, move=0, prior=0.5)
    good_for_opponent.visits, good_for_opponent.value_sum = 10, 8.0

    bad_for_opponent = Node(Board(), PLAYER_Y, parent=parent, move=1, prior=0.5)
    bad_for_opponent.visits, bad_for_opponent.value_sum = 10, -8.0

    parent.children = {0: good_for_opponent, 1: bad_for_opponent}

    assert bad_for_opponent.puct_score() > good_for_opponent.puct_score()
    assert select_child(parent) is bad_for_opponent


# --------------------------------------------------------------------------
# expansion
# --------------------------------------------------------------------------

def test_expand_creates_every_legal_child_at_once():
    """Unlike vanilla MCTS, which attached one child per visit."""
    node = Node(Board(), PLAYER_R)
    expand(node, uniform_evaluator)

    assert sorted(node.children) == list(range(COLS))
    assert node.is_expanded


def test_expand_skips_illegal_columns():
    board = play_alternating([3] * ROWS)
    node = Node(board, PLAYER_R)
    expand(node, uniform_evaluator)

    assert 3 not in node.children
    assert sorted(node.children) == [c for c in range(COLS) if c != 3]


def test_expand_priors_are_masked_and_normalised():
    """Priors over the created children sum to 1, with nothing on illegal moves."""
    board = play_alternating([0] * ROWS)
    node = Node(board, PLAYER_R)
    expand(node, biased_evaluator(favourite=0))

    total = sum(child.prior for child in node.children.values())
    assert total == pytest.approx(1.0)
    assert 0 not in node.children


def test_expand_children_have_the_move_applied():
    node = Node(play([(3, R)]), PLAYER_Y)
    expand(node, uniform_evaluator)

    for col, child in node.children.items():
        assert child.move == col
        assert child.board.move_count == node.board.move_count + 1
        assert child.player_to_move == PLAYER_R
        assert child.parent is node


def test_expand_does_not_mutate_the_parent_board():
    node = Node(play([(3, R), (3, Y)]), PLAYER_R)
    grid_before = [row[:] for row in node.board.grid]

    expand(node, uniform_evaluator)

    assert node.board.grid == grid_before
    assert node.board.move_count == 2


def test_expand_returns_the_evaluator_value():
    node = Node(Board(), PLAYER_R)
    assert expand(node, constant_evaluator(0.42)) == pytest.approx(0.42)


# --------------------------------------------------------------------------
# backup
# --------------------------------------------------------------------------

def test_backup_increments_visits_along_the_path():
    root = Node(Board(), PLAYER_R)
    child = Node(Board(), PLAYER_Y, parent=root)
    grandchild = Node(Board(), PLAYER_R, parent=child)

    backup(grandchild, 1.0)

    assert root.visits == child.visits == grandchild.visits == 1


def test_backup_alternates_the_sign_up_the_tree():
    """The core sign convention."""
    root = Node(Board(), PLAYER_R)
    child = Node(Board(), PLAYER_Y, parent=root)
    grandchild = Node(Board(), PLAYER_R, parent=child)

    backup(grandchild, 1.0)

    assert grandchild.value_sum == 1.0
    assert child.value_sum == -1.0
    assert root.value_sum == 1.0


def test_backup_accumulates_across_calls():
    root = Node(Board(), PLAYER_R)
    child = Node(Board(), PLAYER_Y, parent=root)

    backup(child, 1.0)
    backup(child, 0.5)

    assert child.visits == 2
    assert child.value_sum == pytest.approx(1.5)
    assert root.value_sum == pytest.approx(-1.5)


# --------------------------------------------------------------------------
# search mechanics
# --------------------------------------------------------------------------

def test_root_sees_every_simulation():
    """Root expansion is not backed up, so visits == simulations."""
    for simulations in (1, 10, 64):
        root = run_search(Board(), PLAYER_R, uniform_evaluator, simulations)
        assert root.visits == simulations


def test_search_does_not_mutate_the_caller_board():
    board = play([(3, R), (3, Y)])
    grid_before = [row[:] for row in board.grid]

    run_search(board, PLAYER_R, uniform_evaluator, 32)

    assert board.grid == grid_before
    assert board.move_count == 2


def test_search_only_creates_legal_children():
    board = play_alternating([2] * ROWS)
    root = run_search(board, PLAYER_R, uniform_evaluator, 32)
    assert 2 not in root.children
    assert set(root.children) == set(board.available_moves())


def test_terminal_root_gains_no_children():
    """A finished game must not be expanded - the same contract vanilla MCTS needed, and
    for the same reason: playing on makes winner() report None."""
    root = run_search(r_has_won(), PLAYER_Y, uniform_evaluator, 20)

    assert root.children == {}
    assert root.visits == 20
    assert root.board.move_count == 7
    # Y is on the move at a position R has won, so every simulation scored -1.
    assert root.value_sum == pytest.approx(-20.0)


def test_search_on_a_full_board_returns_a_childless_root():
    root = run_search(full_board(), PLAYER_R, uniform_evaluator, 8)
    assert root.children == {}
    assert root.value_sum == 0.0


def test_one_evaluator_call_per_simulation_plus_the_root():
    """No rollouts: each simulation ends in exactly one evaluation, except those ending at
    a terminal position, which need none."""
    evaluate, calls = counting_evaluator()
    run_search(Board(), PLAYER_R, evaluate, simulations=25)

    # 25 leaf expansions + 1 root expansion; none of these end terminal at
    # this depth on an empty board.
    assert len(calls) == 26


def test_evaluator_is_asked_from_the_right_perspective():
    """Each node must be evaluated as its own mover, alternating down the tree."""
    evaluate, calls = counting_evaluator()
    run_search(Board(), PLAYER_R, evaluate, simulations=8)

    # Root is R to move with 0 pieces; depth-1 children are Y to move with 1.
    assert calls[0] == (0, PLAYER_R)
    assert all(player == PLAYER_Y for count, player in calls[1:] if count == 1)


# --------------------------------------------------------------------------
# search quality - exact even with a useless evaluator
# --------------------------------------------------------------------------

def test_finds_an_immediate_win_with_a_uniform_evaluator():
    """Terminal values inside the tree are exact, so the network's quality is irrelevant
    here."""
    assert puct_move(r_can_win_in_one(), PLAYER_R, uniform_evaluator, simulations=50) == 3


def test_blocks_an_immediate_loss_with_a_uniform_evaluator():
    assert puct_move(y_can_win_in_one(), PLAYER_R, uniform_evaluator, simulations=100) == 3


def test_a_winning_move_dominates_the_visit_counts():
    root = run_search(r_can_win_in_one(), PLAYER_R, uniform_evaluator, 100)
    counts = visit_counts(root)
    assert counts[3] > counts.sum() / 2


def test_priors_steer_the_search():
    """With no value signal at all, visits should follow the priors."""
    root = run_search(Board(), PLAYER_R, biased_evaluator(favourite=5), 100)
    counts = visit_counts(root)
    assert counts.argmax() == 5


def test_value_sign_propagates_to_the_root():
    """An evaluator that always says "the mover is winning" must make the root look
    losing*, because the root's children are the opponent's positions."""
    root = run_search(Board(), PLAYER_R, constant_evaluator(1.0), simulations=1)
    assert root.q == pytest.approx(-1.0)

    root = run_search(Board(), PLAYER_R, constant_evaluator(-1.0), simulations=1)
    assert root.q == pytest.approx(1.0)


def test_a_losing_evaluation_does_not_stop_the_search_finding_a_win():
    """Terminal values override the evaluator entirely."""
    assert puct_move(r_can_win_in_one(), PLAYER_R, constant_evaluator(-1.0), 60) == 3


# --------------------------------------------------------------------------
# visit counts, policy target, move selection
# --------------------------------------------------------------------------

def test_visit_counts_are_zero_on_illegal_moves():
    board = play_alternating([1] * ROWS)
    counts = visit_counts(run_search(board, PLAYER_R, uniform_evaluator, 32))
    assert counts[1] == 0.0
    assert counts.sum() == 32


def test_policy_target_is_a_distribution():
    root = run_search(Board(), PLAYER_R, uniform_evaluator, 64)
    target = policy_target(root)

    assert target.shape == (COLS,)
    assert target.sum() == pytest.approx(1.0)
    assert (target >= 0).all()


def test_policy_target_at_temperature_zero_is_one_hot():
    root = run_search(r_can_win_in_one(), PLAYER_R, uniform_evaluator, 64)
    target = policy_target(root, temperature=0)

    assert target.sum() == pytest.approx(1.0)
    assert target[3] == 1.0


def test_policy_target_matches_normalised_visits_at_temperature_one():
    root = run_search(Board(), PLAYER_R, uniform_evaluator, 64)
    counts = visit_counts(root)
    assert np.allclose(policy_target(root, 1.0), counts / counts.sum(), atol=1e-6)


def test_policy_target_rejects_an_unvisited_root():
    with pytest.raises(ValueError):
        policy_target(Node(Board(), PLAYER_R))


def test_select_move_at_temperature_zero_is_the_most_visited():
    root = run_search(r_can_win_in_one(), PLAYER_R, uniform_evaluator, 64)
    assert select_move(root, temperature=0) == int(visit_counts(root).argmax())


def test_select_move_with_temperature_only_picks_legal_moves():
    board = play_alternating([4] * ROWS)
    root = run_search(board, PLAYER_R, uniform_evaluator, 64)
    rng = np.random.default_rng(0)

    for _ in range(30):
        assert select_move(root, temperature=1.0, rng=rng) in board.available_moves()


def test_select_move_raises_on_a_childless_root():
    with pytest.raises(ValueError):
        select_move(run_search(full_board(), PLAYER_R, uniform_evaluator, 4))


def test_puct_move_on_a_full_board_returns_none():
    assert puct_move(full_board(), PLAYER_R, uniform_evaluator, 8) is None


# --------------------------------------------------------------------------
# Dirichlet noise
# --------------------------------------------------------------------------

def test_dirichlet_noise_keeps_priors_a_distribution():
    root = Node(Board(), PLAYER_R)
    expand(root, uniform_evaluator)
    add_dirichlet_noise(root, rng=np.random.default_rng(0))

    total = sum(child.prior for child in root.children.values())
    assert total == pytest.approx(1.0)
    assert all(child.prior > 0 for child in root.children.values())


def test_dirichlet_noise_changes_the_priors():
    root = Node(Board(), PLAYER_R)
    expand(root, uniform_evaluator)
    before = {col: child.prior for col, child in root.children.items()}

    add_dirichlet_noise(root, rng=np.random.default_rng(0))

    assert any(root.children[col].prior != before[col] for col in before)


def test_dirichlet_noise_only_touches_legal_columns():
    board = play_alternating([0] * ROWS)
    root = Node(board, PLAYER_R)
    expand(root, uniform_evaluator)

    add_dirichlet_noise(root, rng=np.random.default_rng(0))

    assert 0 not in root.children
    assert sum(c.prior for c in root.children.values()) == pytest.approx(1.0)


def test_noise_makes_self_play_vary():
    """Without noise the search is deterministic given a deterministic evaluator, so every
    self-play game would be identical and the training set would stop growing."""
    plain = [
        visit_counts(run_search(Board(), PLAYER_R, uniform_evaluator, 48))
        for _ in range(2)
    ]
    assert np.array_equal(plain[0], plain[1])

    noisy = [
        visit_counts(
            run_search(
                Board(), PLAYER_R, uniform_evaluator, 48,
                add_noise=True, rng=np.random.default_rng(seed),
            )
        )
        for seed in (0, 1)
    ]
    assert not np.array_equal(noisy[0], noisy[1])


# --------------------------------------------------------------------------
# integration with the real network
# --------------------------------------------------------------------------

def test_search_runs_with_an_untrained_network():
    """An untrained network must not be able to break the search - its priors are near-
    uniform and its values near-arbitrary, and that has to be survivable."""
    import torch

    from connect4.network import Connect4Net

    torch.manual_seed(0)
    net = Connect4Net().to(torch.device("cpu")).eval()

    root = run_search(Board(), PLAYER_R, network_evaluator(net), simulations=32)

    assert root.visits == 32
    assert set(root.children) == set(range(COLS))
    assert policy_target(root).sum() == pytest.approx(1.0)


def test_untrained_network_still_finds_a_forced_win():
    """Terminal values are exact, so tactics do not depend on training at all."""
    import torch

    from connect4.network import Connect4Net

    torch.manual_seed(0)
    net = Connect4Net().to(torch.device("cpu")).eval()

    assert puct_move(r_can_win_in_one(), PLAYER_R, network_evaluator(net), 60) == 3


# --------------------------------------------------------------------------
# Search - the resumable form used for batched self-play
# --------------------------------------------------------------------------

def drive(search, evaluator, simulations):
    """Run `search` to completion using `evaluator`, one leaf at a time."""
    while search.simulations_done < simulations:
        leaf = search.pending_leaf()
        if leaf is not None:
            priors, value = evaluator(leaf.board, leaf.player_to_move)
            search.resolve(priors, value)
    return search.root


@pytest.mark.parametrize("simulations", [1, 2, 5, 32, 64])
def test_search_matches_run_search_exactly(simulations):
    """The resumable form must be the same algorithm, not merely a similar one."""
    reference = run_search(Board(), PLAYER_R, uniform_evaluator, simulations)
    stepwise = drive(puct.Search(Board(), PLAYER_R), uniform_evaluator, simulations)

    assert stepwise.visits == reference.visits == simulations
    assert np.array_equal(visit_counts(stepwise), visit_counts(reference))
    assert stepwise.value_sum == pytest.approx(reference.value_sum)


def test_search_hands_out_the_root_first():
    """The root needs evaluating before anything can be selected."""
    search = puct.Search(Board(), PLAYER_R)
    assert search.pending_leaf() is search.root


def test_search_root_expansion_is_not_a_simulation():
    """Root expansion is setup."""
    search = puct.Search(Board(), PLAYER_R)
    leaf = search.pending_leaf()
    search.resolve(*uniform_evaluator(leaf.board, leaf.player_to_move))

    assert search.simulations_done == 0
    assert search.root.visits == 0
    assert search.root.is_expanded


def test_search_rejects_a_second_leaf_before_resolve():
    """At most one leaf outstanding - that invariant is why no virtual loss is needed when
    batching across games."""
    search = puct.Search(Board(), PLAYER_R)
    search.pending_leaf()
    with pytest.raises(RuntimeError):
        search.pending_leaf()


def test_search_rejects_resolve_with_nothing_outstanding():
    search = puct.Search(Board(), PLAYER_R)
    with pytest.raises(RuntimeError):
        search.resolve(np.full(COLS, 1.0 / COLS, dtype=np.float32), 0.0)


def test_search_returns_no_leaf_for_a_terminal_root():
    """A finished game needs no evaluation, and must gain no children."""
    search = puct.Search(r_has_won(), PLAYER_Y)
    assert search.finished
    assert search.pending_leaf() is None
    assert search.root.children == {}
    assert search.root.visits == 1
    assert search.root.value_sum == pytest.approx(-1.0)


def test_search_returns_no_leaf_when_a_simulation_ends_terminal():
    """Simulations that reach a decided position resolve without an evaluator call - that's
    why the batch is sometimes smaller than the game count."""
    search = puct.Search(r_can_win_in_one(), PLAYER_R)
    drive(search, uniform_evaluator, 60)

    leaves = [search.pending_leaf() for _ in range(10)]
    assert any(leaf is None for leaf in leaves)


def test_search_noise_requires_an_expanded_root():
    search = puct.Search(Board(), PLAYER_R)
    with pytest.raises(RuntimeError):
        search.add_noise()
