# Project: Open Spiel LLM Arena

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
   - Install the required dependencies using pip:
     ```bash
     pip install transformers open_spiel
     ```

2. **Install OpenSpiel Framework**:
   - Clone and set up OpenSpiel from its official repository:
     ```bash
     git clone https://github.com/deepmind/open_spiel.git
     cd open_spiel
     ./install.sh
     ```

### Running the Simulator
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Run the main script:
   ```bash
   python game_simulator.py
   ```

3. Follow the prompts:
   - Select the games to simulate by entering their numbers.
   - Choose the opponent type for each game:
     1. Random Bot.
     2. Another LLM.
     3. Self-Play.

4. Observe the game states, LLM decisions, and final results printed to the console.

---

## 2. Games Available (for now)

### List of Supported Games
1. **Tic-Tac-Toe**:
   - A classic 3x3 grid game where players aim to align three symbols horizontally, vertically, or diagonally.

2. **Python Iterated Prisoner’s Dilemma**:
   - A repeated strategy game where players choose between cooperation and defection to maximize rewards over multiple rounds.

3. **Rock-Paper-Scissors**:
   - A simultaneous-move game where rock beats scissors, scissors beats paper, and paper beats rock.

4. **Matching Pennies**:
   - A simple two-player game where Player 1 wins if both players’ choices match, and Player 2 wins if they differ.

---

## 3. Features
- **LLM Integration**:
  - Uses Hugging Face Transformers to incorporate open-source LLMs (e.g., GPT-2, FLAN-T5).
  - Converts game states into natural language prompts for LLM decision-making.

- **Flexible Opponent Options**:
  - LLMs can play against random bots, other LLMs, or themselves.

- **Extensible Framework**:
  - Supports adding new OpenSpiel games by defining custom simulators and integrating them into the pipeline.

---

## 4. Example Output

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

### Game: Matching Pennies
```
Current state of Matching Pennies:
p0: Heads
p1: Tails
Final state of Matching Pennies:
p0: Heads
p1: Tails
Scores: {'google/flan-t5-small': 1.0, 'gpt2': -1.0}
```

---

## 5. Contribution Guidelines
Contributions are welcome! To contribute:
1. Fork this repository.
2. Create a feature branch.
3. Submit a pull request with a detailed explanation of your changes.

---


