"""Tests for vanilla MCTS.

MCTS is stochastic, so tests come in two flavours:

  * **Deterministic** — structure and bookkeeping (UCB1, expansion, backprop
    perspective, board hygiene). These must always pass.
  * **Statistical** — playing strength. Seeded via random.seed() so a failure is
    reproducible, and written with margins wide enough that a correct
    implementation is not flaky. A tight margin here would cost more hours in
    false alarms than it ever caught bugs.

Strength expectations, so a correct implementation isn't mistaken for a broken
one: MCTS should crush a random player, find forced wins given enough
simulations, and stay roughly level with shallow alpha-beta while losing ground
as alpha-beta's depth rises.

Positions are built by playing moves, never by writing to `grid` directly —
a hand-written grid can hold physically impossible states (pieces floating with
empty cells beneath) and, worse, leaves last_move unset so winner() returns None.
"""

import random

import pytest

from connect4.board import Board, EMPTY, PLAYER_R, PLAYER_Y, ROWS, COLS
from connect4 import alphabeta as ab
from connect4 import mcts as mc
from connect4.mcts import Node, other, DRAW_REWARD, WIN_REWARD, LOSS_REWARD
from tests.test_board import play, play_alternating, DRAW_SEQUENCE, R, Y
from tests.test_minimax import r_can_win_in_one, y_can_win_in_one, r_has_won


def full_board() -> Board:
    """A legally-reachable full board with no winner."""
    return play_alternating(DRAW_SEQUENCE)


def play_game(agent_r, agent_y, first: str = PLAYER_R) -> str | None:
    """Play one full game and return the winner, or None for a draw."""
    board = Board()
    player = first
    while not board.is_terminal():
        agent = agent_r if player == PLAYER_R else agent_y
        board.make_move(agent(board, player), player)
        player = other(player)
    return board.winner()


def mcts_agent(simulations: int):
    def agent(board: Board, player: str) -> int:
        return mc.mcts_move(board, player, simulations)
    return agent


def random_agent(board: Board, player: str) -> int:  # noqa: ARG001 - agent signature
    return random.choice(board.available_moves())


def alphabeta_agent(depth: int):
    def agent(board: Board, player: str) -> int:
        col, _ = ab.best_move(board, depth, player == PLAYER_R)
        return col
    return agent


def score_match(agent, opponent, games: int) -> dict[str, int]:
    """Play `games` games, alternating who moves first. Counts are for `agent`.

    `agent` always plays R; only the first move alternates, so neither side keeps
    the first-move advantage across the match.
    """
    tally = {"win": 0, "loss": 0, "draw": 0}
    for i in range(games):
        first = PLAYER_R if i % 2 == 0 else PLAYER_Y
        winner = play_game(agent, opponent, first=first)
        if winner is None:
            tally["draw"] += 1
        elif winner == PLAYER_R:
            tally["win"] += 1
        else:
            tally["loss"] += 1
    return tally


# --------------------------------------------------------------------------
# Board.copy() — the prerequisite
# --------------------------------------------------------------------------

def test_copy_is_independent_of_the_original():
    """Writing to the copy must not touch the original.

    Board(list(grid)) shares the row lists and fails this. Every node in the
    tree would mutate every other node's position.
    """
    original = play([(3, R), (3, Y)])
    clone = original.copy()

    clone.make_move(0, PLAYER_R)

    assert original.grid[ROWS - 1][0] == EMPTY
    assert original.move_count == 2
    assert clone.move_count == 3


def test_copy_preserves_a_win():
    """A copy of a won board still reports its winner.

    The constructor sets last_move = None, and winner() only searches lines
    through last_move — so a copy that drops it reports no winner, is_terminal()
    goes False, and every rollout runs 42 moves and scores as a draw.
    """
    clone = r_has_won().copy()
    assert clone.winner() == PLAYER_R
    assert clone.is_terminal()


def test_copy_preserves_move_count_and_history():
    """move_count, last_move and last_player all survive the copy."""
    original = play([(3, R), (3, Y), (4, R)])
    clone = original.copy()

    assert clone.move_count == original.move_count == 3
    assert clone.last_move == original.last_move
    assert clone.last_player == original.last_player


def test_copy_of_empty_board_is_empty():
    clone = Board().copy()
    assert clone.move_count == 0
    assert clone.last_move is None
    assert clone.last_player is None
    assert clone.available_moves() == list(range(COLS))


# --------------------------------------------------------------------------
# Node bookkeeping
# --------------------------------------------------------------------------

def test_other_swaps_players():
    assert other(PLAYER_R) == PLAYER_Y
    assert other(PLAYER_Y) == PLAYER_R


def test_new_node_has_all_moves_untried():
    """untried_moves matches available_moves(), and is the node's own list —
    consuming it must not disturb the board."""
    board = Board()
    node = Node(board, PLAYER_R)

    assert sorted(node.untried_moves) == board.available_moves()

    node.untried_moves.pop()
    assert board.available_moves() == list(range(COLS))


def test_node_is_not_fully_expanded_until_all_moves_tried():
    node = Node(Board(), PLAYER_R)
    assert not node.is_fully_expanded()

    node.untried_moves.clear()
    assert node.is_fully_expanded()


def test_unvisited_child_scores_infinite_ucb1():
    """Otherwise selection revisits a known child instead of trying a new one,
    and whole branches never get explored."""
    parent = Node(Board(), PLAYER_R)
    parent.visits = 10
    child = Node(Board(), PLAYER_Y, parent=parent)
    parent.children.append(child)

    assert child.ucb1() == float("inf")


def test_ucb1_prefers_the_higher_win_rate_at_equal_visits():
    """Exploitation term behaves. Equal visits, different wins.

    parent.visits must be set: the exploration term takes log(parent.visits),
    and log(0) raises ValueError.
    """
    parent = Node(Board(), PLAYER_R)
    parent.visits = 20

    better = Node(Board(), PLAYER_Y, parent=parent)
    better.visits, better.wins = 10, 7.0
    worse = Node(Board(), PLAYER_Y, parent=parent)
    worse.visits, worse.wins = 10, 3.0
    parent.children.extend([better, worse])

    assert better.ucb1() > worse.ucb1()


def test_ucb1_prefers_the_less_visited_child_at_equal_win_rate():
    """Exploration term behaves. Identical win rate, different visit counts.

    Both children sit at 70%, so the exploitation term cancels exactly and only
    the exploration term can separate them. A sign error there is invisible in
    the test above but fails here.
    """
    parent = Node(Board(), PLAYER_R)
    parent.visits = 30

    fewer = Node(Board(), PLAYER_Y, parent=parent)
    fewer.visits, fewer.wins = 10, 7.0
    more = Node(Board(), PLAYER_Y, parent=parent)
    more.visits, more.wins = 20, 14.0
    parent.children.extend([fewer, more])

    assert fewer.wins / fewer.visits == more.wins / more.visits
    assert fewer.ucb1() > more.ucb1()


def test_best_child_picks_the_highest_ucb1():
    parent = Node(Board(), PLAYER_R)
    parent.visits = 20

    low = Node(Board(), PLAYER_Y, parent=parent, move=0)
    low.visits, low.wins = 10, 1.0
    high = Node(Board(), PLAYER_Y, parent=parent, move=1)
    high.visits, high.wins = 10, 9.0
    parent.children.extend([low, high])

    assert parent.best_child() is high


# --------------------------------------------------------------------------
# the four phases
# --------------------------------------------------------------------------

def test_expand_attaches_one_child_and_removes_the_move():
    node = Node(Board(), PLAYER_R)
    before = list(node.untried_moves)

    child = mc.expand(node)

    assert node.children == [child]
    assert len(node.untried_moves) == len(before) - 1
    assert child.move in before
    assert child.move not in node.untried_moves
    assert child.parent is node


def test_expand_applies_the_move_to_a_copy():
    """The child's board has one more piece; the parent's board is untouched."""
    node = Node(play([(3, R), (3, Y)]), PLAYER_R)
    parent_grid = [row[:] for row in node.board.grid]

    child = mc.expand(node)

    assert child.board.move_count == node.board.move_count + 1
    assert node.board.grid == parent_grid
    assert child.board is not node.board


def test_expand_flips_the_player_to_move():
    node = Node(Board(), PLAYER_R)
    child = mc.expand(node)
    assert child.player_to_move == PLAYER_Y
    assert child.board.last_player == PLAYER_R


def test_expand_rejects_a_fully_expanded_node():
    node = Node(Board(), PLAYER_R)
    node.untried_moves.clear()
    with pytest.raises(ValueError):
        mc.expand(node)


def test_select_stops_at_a_node_with_untried_moves():
    """A root that isn't fully expanded is returned as-is."""
    node = Node(Board(), PLAYER_R)
    assert mc.select(node) is node


def test_select_stops_at_a_terminal_node():
    """Descent must not run past the end of a game.

    A won board still has legal moves, so this node is NOT fully expanded —
    which is exactly why select() has to test the position, not the expansion
    state.
    """
    node = Node(r_has_won(), PLAYER_Y)
    assert node.board.is_terminal()
    assert not node.is_fully_expanded()
    assert mc.select(node) is node


def test_select_descends_through_a_fully_expanded_node():
    """With every child attached and visited, selection must move down a level."""
    root = Node(Board(), PLAYER_R)
    while not root.is_fully_expanded():
        child = mc.expand(root)
        child.visits, child.wins = 1, 0.5
    root.visits = COLS

    assert mc.select(root) in root.children


def test_rollout_returns_a_winner_or_none():
    """Result is PLAYER_R, PLAYER_Y, or None — never anything else."""
    random.seed(0)
    board = Board()
    for _ in range(50):
        assert mc.rollout(board, PLAYER_R) in (PLAYER_R, PLAYER_Y, None)


def test_rollout_does_not_mutate_the_board():
    """It plays on a copy. If this fails, the tree is being corrupted from
    underneath the search."""
    random.seed(0)
    board = play([(3, R), (3, Y)])
    grid_before = [row[:] for row in board.grid]

    mc.rollout(board, PLAYER_R)

    assert board.grid == grid_before
    assert board.move_count == 2


def test_rollout_from_a_won_position_returns_that_winner():
    """The rollout must see the game as already over and return immediately.

    If Board.copy() dropped last_move, is_terminal() would be False here, the
    rollout would play on to 42 moves, and the result would come back as a draw.
    """
    assert mc.rollout(r_has_won(), PLAYER_Y) == PLAYER_R


def test_rollout_from_a_full_board_returns_the_draw():
    assert mc.rollout(full_board(), PLAYER_R) is None


def test_backpropagate_increments_visits_along_the_path():
    """Every node from the leaf to the root gains exactly one visit."""
    root = Node(Board(), PLAYER_R)
    child = Node(Board(), PLAYER_Y, parent=root)
    grandchild = Node(Board(), PLAYER_R, parent=child)

    mc.backpropagate(grandchild, PLAYER_R)

    assert root.visits == child.visits == grandchild.visits == 1


def test_backpropagate_credits_the_player_who_moved_into_the_node():
    """The perspective test, and the one most likely to be wrong.

    root is R to move; child is Y to move, so child was reached by R's move. A
    win for R credits the child (R moved into it) and not the root (Y moved into
    the root's position, notionally). Crediting by player_to_move instead gives a
    search that reliably prefers losing moves.
    """
    root = Node(Board(), PLAYER_R)
    child = Node(Board(), PLAYER_Y, parent=root)
    root.children.append(child)

    mc.backpropagate(child, PLAYER_R)

    assert child.wins == WIN_REWARD
    assert root.wins == LOSS_REWARD


def test_backpropagate_credits_the_other_side_symmetrically():
    """Same shape, opposite winner — the mirror of the test above."""
    root = Node(Board(), PLAYER_R)
    child = Node(Board(), PLAYER_Y, parent=root)
    root.children.append(child)

    mc.backpropagate(child, PLAYER_Y)

    assert child.wins == LOSS_REWARD
    assert root.wins == WIN_REWARD


def test_backpropagate_splits_a_draw():
    """A draw awards DRAW_REWARD to every node on the path."""
    root = Node(Board(), PLAYER_R)
    child = Node(Board(), PLAYER_Y, parent=root)

    mc.backpropagate(child, None)

    assert child.wins == DRAW_REWARD
    assert root.wins == DRAW_REWARD


# --------------------------------------------------------------------------
# mcts_move / build_tree — structure
# --------------------------------------------------------------------------

def test_mcts_move_returns_a_legal_column():
    random.seed(0)
    board = play([(3, R), (3, Y)])
    for _ in range(20):
        assert mc.mcts_move(board, PLAYER_R, simulations=10) in board.available_moves()


def test_mcts_move_on_full_board_returns_none():
    assert mc.mcts_move(full_board(), PLAYER_R, simulations=10) is None


def test_mcts_move_does_not_mutate_the_caller_board():
    """The caller's board must be untouched — grid and move_count."""
    random.seed(0)
    board = play([(3, R), (3, Y)])
    grid_before = [row[:] for row in board.grid]

    mc.mcts_move(board, PLAYER_R, simulations=50)

    assert board.grid == grid_before
    assert board.move_count == 2


def test_root_sees_every_simulation():
    """Root visit count equals the simulation count.

    Catches an off-by-one in the loop and a backprop that stops before the root.
    """
    random.seed(0)
    for simulations in (1, 10, 50):
        root = mc.build_tree(Board(), PLAYER_R, simulations)
        assert root.visits == simulations


def test_never_expands_past_a_terminal_position():
    """A terminal root must never grow children, however many simulations run.

    This is the regression test for the is_fully_expanded()/is_terminal() mix-up:
    a won board still has legal moves, so guarding on expansion state would
    happily play on past the win and make winner() report None for a decided game.
    """
    random.seed(0)
    root = mc.build_tree(r_has_won(), PLAYER_Y, simulations=25)

    assert root.children == []
    assert root.visits == 25
    assert root.board.move_count == 7
    assert root.board.winner() == PLAYER_R

    # The root is Y to move, so it is credited from R's perspective — and R won.
    # A full WIN_REWARD per simulation proves every rollout returned R rather
    # than None: if Board.copy() dropped last_move, is_terminal() would be False
    # here, the rollouts would play on to 42 moves, and this would be 12.5.
    assert root.wins == 25 * WIN_REWARD


# --------------------------------------------------------------------------
# playing strength (stochastic — seeded, wide margins)
# --------------------------------------------------------------------------

def test_finds_an_immediate_win():
    """With enough simulations the winning column dominates the visit counts.

    Needs far more simulations than alpha-beta needs depth: random rollouts have
    to stumble into the win often enough for it to show. If this ever fails,
    raise the count before suspecting a bug.
    """
    random.seed(0)
    root = mc.build_tree(r_can_win_in_one(), PLAYER_R, simulations=2000)
    winning = max(root.children, key=lambda c: c.visits)

    assert winning.move == 3
    assert mc.mcts_move(r_can_win_in_one(), PLAYER_R, simulations=2000) == 3


def test_blocks_an_immediate_loss():
    """The only non-losing move should win the visit count."""
    random.seed(0)
    assert mc.mcts_move(y_can_win_in_one(), PLAYER_R, simulations=2000) == 3


def test_beats_a_random_player_over_many_games():
    """The headline checkpoint: MCTS must dominate uniform random play.

    Margin is deliberately loose — measured 30/30, so 80% leaves plenty of room
    for seed variation without letting a broken search through.
    """
    random.seed(0)
    result = score_match(mcts_agent(120), random_agent, games=20)
    assert result["win"] >= 16, result


@pytest.mark.slow
def test_more_simulations_plays_at_least_as_well():
    """A high-simulation agent should not lose a head-to-head to a
    low-simulation one. Asserted as 'not worse' rather than 'better' — with
    random rollouts the gain is real but noisy."""
    random.seed(0)
    result = score_match(mcts_agent(400), mcts_agent(25), games=8)
    assert result["win"] >= result["loss"], result


@pytest.mark.slow
def test_holds_its_own_against_shallow_alphabeta():
    """Documents the capability gap rather than asserting a win.

    Vanilla MCTS spends ~25,000 board operations per move to match a search
    visiting ~1,000 nodes: competitive only because it's handed far more work.
    That inefficiency is what a learned value head fixes in Phase 4 — one
    evaluation in place of a 42-move random rollout.
    """
    random.seed(0)
    result = score_match(mcts_agent(400), alphabeta_agent(4), games=6)
    assert result["win"] + result["draw"] >= 1, result
