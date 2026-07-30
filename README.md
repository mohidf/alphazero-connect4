# alphazero-connect4

An AlphaZero-style Connect 4 bot I built from scratch to learn how game AI works.

I did it in four stages, each one building on the last:

1. **Game engine** - the board, dropping pieces, checking for wins
2. **Minimax + alpha-beta** with a hand-written scoring function
3. **Monte Carlo Tree Search** with random rollouts
4. **Neural network + self-play** (the AlphaZero part)

Stage 2 turned out to be really useful later on, because it gave me a decent
opponent to test the neural net against.

## Setup

```bash
pip install -r requirements.txt
```

You need Python 3.13. PyTorch will use your GPU if you have one, otherwise CPU
(training is much slower on CPU). tkinter comes with Python, so the window needs
nothing extra beyond Pillow, which is in requirements.txt.

## Play against it

There's a window:

```bash
python -m connect4.gui
python -m connect4.gui --opponent alphabeta --depth 6
python -m connect4.gui --checkpoint checkpoints/big/best.pt --simulations 800
```

Hover over a column to see where the piece would land, click to drop it. **New
game** asks which colour you want (red goes first). When someone wins, the four
pieces that did it get outlined.

The board is drawn with Pillow rather than tkinter shapes, because tkinter's
circles come out jagged - there's no anti-aliasing on its canvas. The pieces are
rendered once at startup and pasted, so hovering stays smooth.

The bot thinks on a background thread. If it didn't, the window would freeze for
the whole search and Windows would grey it out as not responding.

### Or in the terminal

```bash
python -m connect4.play --opponent alphabeta --depth 6
python -m connect4.play --checkpoint checkpoints/best.pt --simulations 400
```

You're red and go first. Type a column number 0-6. This one prints what the bot
was thinking each move - how many times it looked at each column, and whether it
thinks it's winning:

```
search: 0:  0% 1:  3% 2: 70% 3:  2% 4: 22% 5:  3% 6:  0%   value -0.24
```

Both take `--second` to let the bot move first and `--opponent random`.

## Train it

```bash
python run_overnight.py --hours 10 --tag myrun
python run_overnight.py --hours 10 --tag big --channels 128 --blocks 8 --simulations 800
```

It plays games against itself, trains on them, then plays the new network against
the old one and only keeps it if it actually wins. Checkpoints and a log go in
`checkpoints/myrun/`.

To stop it early without losing anything:

```bash
# create a file called STOP in the run's folder
touch checkpoints/myrun/STOP
```

It finishes the current round and saves before exiting.

## Results

All of this is from one network trained for about 16 hours, 222 iterations at 400
simulations a move.

### Games won against fixed opponents

Both opponents stay the same the whole way through, so the only thing changing is
the network. 20 games a point, alternating who starts, with random openings so the
games aren't all identical. Draws count as neither a win nor a loss.

![win rate with search](docs/won_games_alphazero.png)

That's the full agent - network plus PUCT search. The policy head on its own,
with no search at all, is a lot weaker:

![win rate without search](docs/won_games_network_only.png)

### Mistakes against a perfect solver

This one measures moves rather than games. For each test position I ask a perfect
solver to score every legal move, which gives the set of moves that don't throw
the game away. A move that isn't in that set is an error, so one bad move shows up
even in a game the bot goes on to win.

The positions are the six standard test sets from
[blog.gamesolver.org](http://blog.gamesolver.org/solving-connect-four/02-test-protocol/),
1000 each. Blue is the policy head, orange is my depth-5 alpha-beta for
comparison.

![error rates](docs/error_rates.png)

Every panel comes down, and on Beginning-Hard the network reaches alpha-beta -
which is the set alpha-beta is worst at, so that makes sense.

### Where the strength actually comes from

Same network, 200 positions per set:

| test set | policy only | with search | alpha-beta d5 |
|---|---|---|---|
| Beginning - Easy | 44.5% | 33.5% | 9.5% |
| Beginning - Medium | 43.0% | 29.0% | 25.5% |
| Beginning - Hard | 36.0% | 28.0% | **42.5%** |
| Middle - Easy | 44.0% | 22.0% | 4.0% |
| Middle - Medium | 48.0% | 25.5% | 21.0% |
| End - Easy | 28.0% | 12.0% | 2.5% |

Search roughly halves the error rate everywhere. So most of the playing strength
is the search, not what the network has actually learned - the policy plateaued
around iteration 100 and stopped improving.

### Head to head

30 games each, random openings, draws worth half:

| opponent | score |
|---|---|
| untrained network | 0.93 |
| alpha-beta depth 2 | 0.83 |
| alpha-beta depth 4 | 0.67 |
| alpha-beta depth 6 | 0.78 |

So it beats the classical engine I wrote in stage 2. After the first 10 training
rounds it was only scoring 0.17 against depth 4.

Worth noting these two measurements disagree: it wins more games than depth-5
alpha-beta but makes more mistakes than it. Winning and playing well aren't quite
the same thing.

### Reproducing the plots

```bash
python benchmark_curve.py --dir checkpoints/myrun --points 16
python solver_benchmark.py --truth                      # once, ~40s
python solver_benchmark.py --dir checkpoints/myrun --points 16
```

The solver benchmark needs the C++ solver built and the test sets downloaded -
see [external/README.md](external/README.md).

## Tests

```bash
pytest -q
```

291 tests. A lot of them exist because I got something wrong and wanted to make
sure it stayed fixed, like the fact that a won Connect 4 board still has legal
moves, which broke my search in a way that was really hard to spot.

## How the code is laid out

```
connect4/
  board.py       the game itself
  evaluate.py    hand-written scoring function (stage 2)
  minimax.py     minimax search
  alphabeta.py   minimax + pruning + move ordering
  mcts.py        Monte Carlo Tree Search with random rollouts (stage 3)
  encoding.py    turns a board into numbers the network can read
  network.py     the neural network (policy + value)
  puct.py        MCTS guided by the network instead of random rollouts
  selfplay.py    generating training games
  train.py       the loss function and training loop
  arena.py       playing two bots against each other to compare them
  pipeline.py    the full loop: play -> train -> test -> repeat
  play.py        play against it in the terminal
  gui.py         play against it in a window
  solver.py      wrapper around the perfect solver, for benchmarking

run_overnight.py      long training run
benchmark_curve.py    win rate over training iterations
solver_benchmark.py   error rate against the perfect solver
```

## Things I learned the hard way

- Copying a 2D list in Python doesn't copy the inner lists. This bit me three
  separate times.
- A board where someone already won *still has legal moves*, so checking "are
  there moves left" is not the same as "is the game over".
- Two bots that always pick the same move will replay the exact same game every
  time. I was running 12-game matches that were really only 2 different games, so
  my results looked way better than they were.
- Training loss going down doesn't mean the bot got better. The only way to know
  is to make it play games.
- Winning games and making good moves are different things, and you need both
  measurements to know what's going on.
