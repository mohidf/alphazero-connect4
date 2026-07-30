# alphazero-connect4

An AlphaZero-style Connect 4 bot I built from scratch to learn how game AI works.

I did it in four stages, each one building on the last:

1. **Game engine** — the board, dropping pieces, checking for wins
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
(training is much slower on CPU).

## Play against it

```bash
python -m connect4.play --opponent alphabeta --depth 6
python -m connect4.play --checkpoint checkpoints/best.pt --simulations 400
```

You're red and go first. Type a column number 0-6. The bot prints what it was
thinking each move — how many times it looked at each column, and whether it
thinks it's winning:

```
search: 0:  0% 1:  3% 2: 70% 3:  2% 4: 22% 5:  3% 6:  0%   value -0.24
```

Other options: `--second` to let it move first, `--opponent random`, `--quiet` to
hide the search output.

## Train it

```bash
python run_overnight.py --hours 10 --tag myrun
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

## Does it work?

Yes. I tested the trained network against the alpha-beta search from stage 2, 30
games each with random openings so the games aren't all identical (score counts a
draw as half a win):

| opponent | score |
|---|---|
| untrained network | 0.93 |
| alpha-beta depth 2 | 0.83 |
| alpha-beta depth 4 | 0.67 |
| alpha-beta depth 6 | 0.78 |

So it beats the classical engine I wrote earlier. After the first 10 training
rounds it was only scoring 0.17 against depth 4, so most of the strength came from
letting it train overnight (about 16 hours total).

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
  play.py        play against it yourself
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

