# Project: OpenSpiel LLM Arena

## 0. Project Goal
The goal of this project is to evaluate the decision-making capabilities of Large Language Models (LLMs) by engaging them in simple games implemented using Google's OpenSpiel framework. The LLMs can play against:
1. A random bot.
2. Another LLM.
3. Themselves (self-play).

This project explores how LLMs interpret game states, make strategic decisions, and adapt their behavior through natural language prompts.

---

## 1. How to Run

### Prerequisites
1. **Python Environment**:
   - Python 3.7 or higher.
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Install OpenSpiel Framework**:
   - Clone and set up OpenSpiel from its official repository:
     ```bash
     git clone https://github.com/deepmind/open_spiel.git
     cd open_spiel
     ./install.sh
     ```

3. **Project Setup**:
   - Clone this repository:
     ```bash
     git clone <repository-url>
     cd <repository-folder>
     pip3 install -e .
     ```

---

### Running the Simulator

1. Use the main binary to run simulations:
   - To simulate a single game:
     ```bash
      # From a JSON FILE
      python3 scripts/simulation.py --config config.json

      # Run with default config (Tic-Tac-Toe, 4 rounds)
      python3 scripts/simulate.py

      # Human vs Random
     python3 scripts/simulate.py --override agents.0.type=human agents.1.type=random

      # Human vs LLM
      python3 scripts/simulate.py --override env_config.game_name=tic_tac_toe agents.0.type=human agents.0.model=None agents.1.type=llm agents.1.model=gpt2

      # Random Bot vs LLM (Connect Four)
      python3 scripts/simulate.py --override env_config.game_name=connect_four agents.0.type=random agents.0.model=None agents.1.type=llm agents.1.model=flan_t5_small

      # Self-Play Tournament (10 rounds)
      python3 scripts/simulate.py --override env_config.game_name=kuhn_poker agents.0.type=llm agents.0.model=gpt2 agents.1.type=llm agents.1.model=distilgpt2 num_episodes=10

      # Multi-Game Challenge
      python3 scripts/simulate.py --override env_config.game_name=tic_tac_toe num_episodes=5 && \
      python3 scripts/simulate.py --override env_config.game_name=prisoners_dilemma num_episodes=5 && \
      python3 scripts/simulate.py --override env_config.game_name=connect_four num_episodes=5

     ```

2. Command-line options:
   - `--config`: Specify a JSON configuration file or raw JSON string.
                 Example: python scripts/simulation.py --config config.json
   - `--override`: Allows to modify specific parts of the default configuration.
---

## 2. Directory Structure

### Packages
- **`games/`**: Game loaders.
- **`simulators/`**: Simulator logic for each game.
- **`utils/`**: Shared utility functions (e.g., prompt generation, LLM integration).

### Results
- **`results/`**: Stores the JSON files with simulation results.

### Binary
- **`scripts/run_simulation.py`**: The main script to run simulations. This is the entry point for the project.

### Tests
- **`tests/`**: Unit tests for utilities and simulators.

### Game Registry
- **`games_registry.py`**: Centralized file for managing all available games and their simulators.

---

## 3. Adding a New Game
To add a new game to the OpenSpiel LLM Arena, follow these steps:

### Step 1: Implement the Game Loader
1. Create a new entry for the game in the **`games/loaders_module`** script.
   - For example, to add **Matching Pennies**:
     - Create:
       ```python
       def get_matching_pennies_game():
           from open_spiel.python.games import matching_pennies
           return matching_pennies.MatchingPenniesGame()
       ```

### Step 2: Implement the Game Simulator
2. Create a corresponding simulator file in the **`simulators/`** folder.
   - The simulator should inherit from the base `GameSimulator` class.

### Step 3: Register the Game
3. Add the new game to the **`games_registry.py`** file.
   - Example:
     ```python
     from games.matching_pennies import get_matching_pennies_game
     from simulators.matching_pennies_simulator import MatchingPenniesSimulator

     GAMES_REGISTRY["matching_pennies"] = {
         "loader": get_matching_pennies_game,
         "simulator": MatchingPenniesSimulator,
         "display_name": "Matching Pennies",
     }
     ```
---

## 4. Adding a New Agent
1. Implement the Agent Class
* Create a new file in `agents/`, e.g., `agents/rl_agent.py`.
* Ensure it inherits from `BaseAgent`.
* Implement `compute_action()`.

2. Register the Agent
Modify `agent_registry.py`.
Example:
```python
from agents.rl_agent import RLAgent
register_agent("rl", RLAgent)
```

3. Use the RL Agent in Simulation.
Example:
```python
python3 scripts/simulate.py --override agents.0.type=rl agents.0.model=my_trained_rl_model
```

## 5. Contribution Guidelines

### Steps to Contribute:
1. Fork this repository.
2. Create a feature branch.
3. Follow the directory structure and coding style outlined in this README.
4. Add appropriate unit tests for your contribution.
5. Submit a pull request with a detailed explanation of your changes.

---

## 6. Example Output

### Game: Tic-Tac-Toe
```
Current state of Tic-Tac-Toe:
x.o
...
...
LLM chooses action: 4
...
Final state of Tic-Tac-Toe:
x.o
..x
.o.
Scores: {'LLM_1': 1.0, 'Random_Bot': -1.0}
```

### Game: Rock-Paper-Scissors
```
Final state of Rock-Paper-Scissors:
Terminal? true
History: 0, 1
Returns: -1,1
Scores: {'google/flan-t5-small': -1.0, 'gpt2': 1.0}
Results saved to results/rock_paper_scissors_results.json
```

### Game: Connect Four
```
Current state of Connect Four:
.....
.....
...o.
..x..
.....
.....
...
Final state of Connect Four:
x wins!
Scores: {'google/flan-t5-small': 1.0, 'Random_Bot': -1.0}
```
