# Hangman Solver using HMM and Reinforcement Learning

## Project By
- **Keerti Kallugadde** - PES1UG23AM142
- **Kashish K S** – PES1UG23AM141  
- **Khushi Kogganur** – PES1UG23AM145  
- **Khushi Dev** – PES1UG23AM144  


This project implements an intelligent Hangman-solving agent using:

- **Hidden Markov Model (HMM)** – to estimate the probability of letters at specific positions in a word.
- **Q-Learning (Reinforcement Learning)** – to learn the best letter-guessing strategy based on rewards.
- **Evaluation and visualization** of the agent’s performance.

---

## Project Structure

├── corpus.txt # Training word dataset

├── test.txt # Testing word dataset

├── models/

│ ├── hmm_model.pkl # Saved HMM letter-position model

│ ├── rl_agent.pkl # Trained RL model

├── HMM_Training.py # Builds and saves the HMM model

├── RL_Agent.py # Trains the Q-learning agent

├── Evaluation.py # Evaluates the model performance

├── training_plots.png # Reward, win rate, and epsilon graphs

└── README.md


##  How It Works

### 1. HMM Training (`HMM_Training.py`)
- Loads and preprocesses words (lowercasing, removing duplicates and non-alphabetic words).
- Calculates probability of each letter at each position for different word lengths.
- Saves the trained HMM model to `models/hmm_model.pkl`.

### 2. RL Training (`RL_Agent.py`)
- Implements Q-learning.
- State = (current word pattern, guessed letters).
- Action = guessing a new letter.
- The score for each action combines:  
  **Q-value + HMM letter probability**

#### Reward Function:
| Event             | Reward |
|-------------------|--------|
| Correct guess     | +12 (+4 for each extra position revealed) |
| Wrong guess       | -10    |
| Repeated guess    | -4     |
| Word guessed      | +80    |
| Word lost         | -150   |

- Model is saved as `models/rl_agent.pkl`.
- Generates training graphs (`training_plots.png`).

### 3. Evaluation (`Evaluation.py`)
- Loads both models.
- Tests on:
  - 2000 words from `test.txt`
  - 2000 random words from `corpus.txt`
- Reports accuracy, wrong guesses, repeated guesses, and final score.

---

## Results

| Dataset       | Total Words | Successful Guesses | Accuracy |
|---------------|-------------|---------------------|----------|
| Test Words    | 2000        | 437                 | 21.85%   |
| Corpus Words  | 2000        | 449                 | 22.45%   |
| **Overall**   | 4000        | 886                 | **22.15%** |

- **Final Score:** 1,661,285  
- **Average Wrong Guesses per Game:** ~5–6  
- **Repeated Guesses:** 0  

---

## Training Visualizations (training_plots.png)

- Reward per episode  
- Win rate over time  
- Epsilon (exploration vs. exploitation) decay  

---

## How to Run

### Step 1: Train HMM Model
bash
python HMM_Training.py

Step 2: Train RL Agent
python RL_Agent.py

Step 3: Evaluate the Model
python Evaluation.py

 Requirements

Install necessary libraries using:

pip install numpy matplotlib

 Future Improvements
 | Improvement              | Description                                         |
| ------------------------ | --------------------------------------------------- |
| Candidate word filtering | Guess letters only from possible word set           |
| Deep Q-Learning (DQN)    | Replace Q-table with neural network                 |
| N-Gram or LSTM models    | Better alternative to HMM for letter prediction     |
| Larger corpus dataset    | Better prediction accuracy                          |
| Optimized reward shaping | Improve learning efficiency                         |
| Improved epsilon decay   | Better balance between exploration and exploitation |


