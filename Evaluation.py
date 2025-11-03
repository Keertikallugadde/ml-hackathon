import pickle
import random
import numpy as np
from collections import defaultdict, Counter

def default_dict_factory():
    return defaultdict(Counter)

def default_dict_dict_factory():
    return defaultdict(dict)

alphabet = list("abcdefghijklmnopqrstuvwxyz")

class QLearningAgent:
    def __init__(self, alpha=0.15, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995):
        self.Q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_state(self, pattern, guessed):
        return (pattern, "".join(sorted(guessed)))

    def choose_action(self, state, pattern, guessed):
        available = [a for a in alphabet if a not in guessed]
        if not available:
            return None
        L = len(pattern)
        hmm_probs = {a: 0 for a in available}
        if L in position_probs:
            for pos, ch in enumerate(pattern):
                if ch == "_":
                    for a in available:
                        hmm_probs[a] += position_probs[L][pos].get(a, 0)

        total_hmm = sum(hmm_probs.values()) + 1e-9
        hmm_probs = {a: hmm_probs[a]/total_hmm for a in available}

        if random.random() < self.epsilon:
            return random.choice(available)

        scores = {a: self.Q[(state, a)] + 10*hmm_probs.get(a, 0) for a in available}
        return max(scores, key=scores.get)

    def update(self, state, action, reward, next_state):
        max_next = max([self.Q[(next_state, a)] for a in alphabet], default=0)
        self.Q[(state, action)] += self.alpha * (reward + self.gamma * max_next - self.Q[(state, action)])

# Load models
with open('models/hmm_model.pkl', 'rb') as f:
    hmm_model = pickle.load(f)

position_probs = hmm_model['position_probs']

with open('models/rl_agent.pkl', 'rb') as f:
    rl_agent = pickle.load(f)

# Set epsilon to 0 for evaluation (no exploration, only exploitation)
rl_agent.epsilon = 0.0

with open('corpus.txt', 'r') as f:
    corpus_words = [w.strip().lower() for w in f.read().splitlines() if w.strip()]

with open('test.txt', 'r') as f:
    test_words = [w.strip().lower() for w in f.read().splitlines() if w.strip()]

print(f"Loaded corpus words: {len(corpus_words)}")
print(f"Loaded test words: {len(test_words)}")
print(f"RL Agent Q-table size: {len(rl_agent.Q)}")
print()

class HangmanEnv:
    def __init__(self, words, max_lives=6):
        self.words = words
        self.max_lives = max_lives

    def reset(self):
        self.word = random.choice(self.words)
        self.guessed = set()
        self.lives = self.max_lives
        self.pattern = "_" * len(self.word)
        return self.pattern

    def step(self, letter):
        if letter in self.guessed:
            return self.pattern, 0, False, None
        self.guessed.add(letter)
        if letter in self.word:
            new_pattern = list(self.pattern)
            for i, ch in enumerate(self.word):
                if ch == letter:
                    new_pattern[i] = letter
            self.pattern = "".join(new_pattern)
        else:
            self.lives -= 1
        if "_" not in self.pattern:
            return self.pattern, 0, True, True
        if self.lives <= 0:
            return self.pattern, 0, True, False
        return self.pattern, 0, False, None

print("Evaluating RL Agent on TEST words...")
env = HangmanEnv(test_words)
games_test = len(test_words)
success_count_test = wrong_guesses_test = repeated_guesses_test = 0

for _ in range(games_test):
    pattern = env.reset()
    guessed = set()
    while True:
        state = rl_agent.get_state(pattern, guessed)
        action = rl_agent.choose_action(state, pattern, guessed)
        
        if action is None or action in guessed:
            # Fallback if no action available
            available = [a for a in alphabet if a not in guessed]
            if not available:
                break
            action = available[0]
        
        next_pattern, _, done, success = env.step(action)
        
        if action in guessed:
            repeated_guesses_test += 1
        elif action not in env.word:
            wrong_guesses_test += 1
            
        guessed.add(action)
        pattern = next_pattern
        
        if done:
            if success:
                success_count_test += 1
            break

print("Evaluating RL Agent on CORPUS words...")
corpus_sample = corpus_words[:2000]
env = HangmanEnv(corpus_sample)
games_corpus = len(corpus_sample)
success_count_corpus = wrong_guesses_corpus = repeated_guesses_corpus = 0

for _ in range(games_corpus):
    pattern = env.reset()
    guessed = set()
    while True:
        state = rl_agent.get_state(pattern, guessed)
        action = rl_agent.choose_action(state, pattern, guessed)
        
        if action is None or action in guessed:
            # Fallback if no action available
            available = [a for a in alphabet if a not in guessed]
            if not available:
                break
            action = available[0]
        
        next_pattern, _, done, success = env.step(action)
        
        if action in guessed:
            repeated_guesses_corpus += 1
        elif action not in env.word:
            wrong_guesses_corpus += 1
            
        guessed.add(action)
        pattern = next_pattern
        
        if done:
            if success:
                success_count_corpus += 1
            break

total_success = success_count_test + success_count_corpus
total_games = games_test + games_corpus
total_wrong = wrong_guesses_test + wrong_guesses_corpus
total_repeated = repeated_guesses_test + repeated_guesses_corpus
avg_wrong_per_game = total_wrong / total_games
avg_repeated_per_game = total_repeated / total_games

final_score = (total_success * 2000) - (total_wrong * 5) - (total_repeated * 2)
success_rate = total_success / total_games

print("\n" + "=" * 60)
print("RL Agent Evaluation Complete!")
print("=" * 60)

print(f"\nTEST Words:")
print(f"  Total Games: {games_test}")
print(f"  Wins: {success_count_test}")
print(f"  Losses: {games_test - success_count_test}")
print(f"  Success Rate: {success_count_test/games_test:.2%} ({success_count_test}/{games_test})")
print(f"  Wrong Guesses: {wrong_guesses_test}")
print(f"  Repeated Guesses: {repeated_guesses_test}")

print(f"\nCORPUS Words:")
print(f"  Total Games: {games_corpus}")
print(f"  Wins: {success_count_corpus}")
print(f"  Losses: {games_corpus - success_count_corpus}")
print(f"  Success Rate: {success_count_corpus/games_corpus:.2%} ({success_count_corpus}/{games_corpus})")
print(f"  Wrong Guesses: {wrong_guesses_corpus}")
print(f"  Repeated Guesses: {repeated_guesses_corpus}")

print(f"\nOVERALL:")
print(f"  Total Games Played: {total_games}")
print(f"  Total Wins: {total_success}")
print(f"  Total Losses: {total_games - total_success}")
print(f"  Success Rate: {success_rate:.2%} ({total_success}/{total_games})")
print(f"  Total Wrong Guesses: {total_wrong}")
print(f"  Total Repeated Guesses: {total_repeated}")
print(f"  Avg Wrong Guesses per Game: {avg_wrong_per_game:.2f}")
print(f"  Avg Repeated Guesses per Game: {avg_repeated_per_game:.2f}")
print(f"  Final Score: {final_score}")
print("=" * 60)