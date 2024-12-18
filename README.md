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
     python3 scripts/run_simulation.py --games tic_tac_toe
     ```
   - To simulate multiple games:
     ```bash
     python3 scripts/run_simulation.py --games tic_tac_toe rps prisoners_dilemma connect_four kuhn_poker matching_pennies_3p
     ```

2. Command-line options:
   - `--games`: Specify one or more games to simulate.

---

## 2. Directory Structure

### Packages
- **`games/`**: Game-specific logic (e.g., rules for Tic-Tac-Toe, Connect Four, Matrix Games, etc.).
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

## 3. Games Available

### List of Supported Games
1. **Tic-Tac-Toe**:
   - A classic 3x3 grid game where players aim to align three symbols horizontally, vertically, or diagonally.

2. **Python Iterated Prisoner’s Dilemma**:
   - A repeated strategy game where players choose between cooperation and defection to maximize rewards over multiple rounds.

3. **Rock-Paper-Scissors**:
   - A simultaneous-move game where rock beats scissors, scissors beats paper, and paper beats rock.

4. **Connect Four**:
   - A two-player game where players drop colored discs into a 7x6 grid, aiming to connect four of their own discs in a line.

5. **Matrix Rock-Paper-Scissors**:
   - A variant of RPS represented as a payoff matrix.

6. **Matrix Prisoner’s Dilemma**:
   - A matrix version of Prisoner’s Dilemma, emphasizing strategic interactions between players.

7. **Kuhn Poker**:
   - A simplified poker game with limited betting rounds and a small deck.

8. **Matching Pennies (3-player)**:
   - A three-player game where each player aims to match or mismatch the choices of others to achieve a payoff.

---

## 4. Adding a New Game
To add a new game to the OpenSpiel LLM Arena, follow these steps:

### Step 1: Implement the Game Loader
1. Create a new Python file for the game in the **`games/`** folder.
   - For example, to add **Matching Pennies**:
     - Create `games/matching_pennies.py`:
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

### Step 4: Run the New Game
4. Use the `run_simulation.py` script to test the new game. Example:
   ```bash
   python3 scripts/run_simulation.py --games matching_pennies
   ```

---

## 5. Features

- **LLM Integration**:
  - Uses Hugging Face Transformers to incorporate open-source LLMs (e.g., GPT-2, FLAN-T5).
  - Converts game states into natural language prompts for LLM decision-making.

- **Flexible Opponent Options**:
  - LLMs can play against random bots, other LLMs, or themselves.

- **Extensible Framework**:
  - Supports adding new OpenSpiel games by defining custom simulators and integrating them into the pipeline.

- **Results Logging**:
  - Saves simulation results in JSON format inside the `results/` folder for easy analysis and review.

---

## 6. Contribution Guidelines

### Steps to Contribute:
1. Fork this repository.
2. Create a feature branch.
3. Follow the directory structure and coding style outlined in this README.
4. Add appropriate unit tests for your contribution.
5. Submit a pull request with a detailed explanation of your changes.

---

## 7. Example Output

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
