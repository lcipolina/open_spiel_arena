#!/usr/bin/env python3
"""
train.py

Place-holder script for RL training.
It might set up a training loop, or rely on an RL library.
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description="Train an RL agent on an OpenSpiel game.")
    parser.add_argument("--game", type=str, default="tic_tac_toe",
                        help="Which OpenSpiel game to load.")
    parser.add_argument("--episodes", type=int, default=10000,
                        help="Number of training episodes.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed.")
    # Add hyperparameters if needed (learning rate, discount, etc.)

    args = parser.parse_args()

    # 1. Load game, build environment
    # 2. Create your RL agent/policy
    # 3. for episode in range(args.episodes):
    #       reset environment
    #       while not done:
    #           step with agent
    #           collect transitions
    #       update agent



if __name__ == "__main__":
    main()
