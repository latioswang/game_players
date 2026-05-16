# game_players

Reinforcement-learning game players. The first implementation is a 2048 agent that learns with temporal-difference RL and an n-tuple value function.

## Run it

```bash
python -m game_players.cli train --episodes 1000 --eval-every 100
python -m game_players.cli eval --games 100 --show-board
```

The saved model is written to `models/2048-agent.pkl`.

Training resumes from `models/2048-agent.pkl` by default when that checkpoint exists:

```bash
python -m game_players.cli train --episodes 1000
```

To ignore an existing checkpoint and start over:

```bash
python -m game_players.cli train --fresh --episodes 1000
```

By default, training tries to use the best target available on the current computer:

- If PyTorch is installed and a GPU backend is available, it uses the PyTorch DQN agent.
- On Apple Silicon Macs, the GPU backend is Apple Metal/MPS.
- If no PyTorch GPU backend is available, it falls back to the CPU n-tuple agent.
- Codex's command sandbox can block PyTorch's MPS check and produce a misleading macOS-version error. Run training from a normal shell, or allow the command to run outside the sandbox, to use MPS.

Set up dependencies in the local venv:

```bash
python3 -m venv venv
venv/bin/python -m pip install -e ".[gpu,log,dev]"
```

You can force a specific agent:

```bash
python -m game_players.cli train --agent dqn --device auto
python -m game_players.cli train --agent ntuple
```

Runtime logs use Python `glog`, so training and evaluation lines look like:

```text
I0517 00:57:59.432106 80581 cli.py:140] using n-tuple agent on cpu: no PyTorch GPU backend is available
```

## Planning the 2048 RL Agent

The main decisions are:

1. **What is the environment?** 2048 is a Markov decision process: the state is the 4x4 board, the action is one of four swipes, the reward is the sum of merged tile values, and the environment adds a random `2` or `4` tile after each valid move.

2. **What should the agent learn?** This implementation learns the value of an *afterstate*: the board after the swipe/merge but before the random tile appears. That removes one source of randomness from action evaluation. The agent can compare actions by `merge_reward + value(afterstate)`.

3. **What function approximator is appropriate?** A full lookup table over all boards is impossible, and a neural network would add training complexity. N-tuple features are a strong middle ground for 2048: each feature looks at a small pattern such as a row, column, or 2x2 square, then stores a learned weight for the exact tile exponents in that pattern.

4. **How does it learn?** It uses TD(0). After choosing a move and seeing the next board, it updates the previous afterstate toward:

   ```text
   next_merge_reward + gamma * value(next_afterstate)
   ```

   At the end of a game, the final afterstate is updated toward `0` because no future reward remains.

5. **How does it explore?** During training, epsilon-greedy exploration occasionally picks a random legal move. Evaluation disables exploration and always chooses the highest-valued move.

6. **What is the reward?** The reward is the score gained by merges on each move, matching the official game score. This makes the learned objective easy to interpret.

7. **What is the first quality bar?** The code should run with only the Python standard library, have deterministic seeds for repeatable experiments, and include tests for the move mechanics because bad 2048 mechanics would invalidate all learning results.

## Files

- `game_players/game2048.py`: board representation and game rules.
- `game_players/ntuple_agent.py`: TD learner and policy.
- `game_players/cli.py`: training and evaluation commands.
- `tests/test_game2048.py`: core mechanics tests.
