"""
Interactive Hangman Game - Play against the AI or let AI play
Uses trained HMM model from hmm_model.pkl
"""

import pickle
import os
import random
from collections import Counter

# Load HMM model
def load_hmm():
    if not os.path.exists('hmm_model.pkl'):
        print("ERROR: hmm_model.pkl not found!")
        print("Please run part2_rl_agent.ipynb first to train the model.")
        return None
    
    with open('hmm_model.pkl', 'rb') as f:
        hmm = pickle.load(f)
    
    # Convert legacy dict to adapter if needed
    if isinstance(hmm, dict):
        print("Converting legacy HMM format...")
        letter_freq = hmm.get('letter_freq', None)
        if letter_freq is None:
            # Build from corpus
            with open('data/corpus.txt', 'r') as f:
                corpus = [w.strip().lower() for w in f.readlines()]
            letter_freq = Counter()
            for word in corpus:
                letter_freq.update(list(word))
        
        class HMMAdapter:
            def __init__(self, lf):
                self.letter_freq = Counter(lf)
            
            def predict_letter_probabilities(self, masked_word, guessed_letters):
                total = float(sum(self.letter_freq.values()) or 1.0)
                probs = {l: (self.letter_freq.get(l, 0) / total) for l in 'abcdefghijklmnopqrstuvwxyz'}
                for g in guessed_letters:
                    probs[g] = 0.0
                return probs
            
            def get_best_guess(self, masked_word, guessed_letters):
                probs = self.predict_letter_probabilities(masked_word, guessed_letters)
                valid = {l: p for l, p in probs.items() if l not in guessed_letters}
                if not valid:
                    return random.choice([c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in guessed_letters])
                return max(valid.items(), key=lambda x: x[1])[0]
        
        hmm = HMMAdapter(letter_freq)
    
    return hmm

# Load word list
def load_words():
    if os.path.exists('data/test.txt'):
        with open('data/test.txt', 'r') as f:
            words = [w.strip().lower() for w in f.readlines()]
    elif os.path.exists('data/corpus.txt'):
        with open('data/corpus.txt', 'r') as f:
            words = [w.strip().lower() for w in f.readlines()]
    else:
        words = ['python', 'hangman', 'machine', 'learning', 'algorithm']
    return words

# Display hangman
def display_hangman(wrong_guesses):
    stages = [
        """
           ------
           |    |
           |
           |
           |
           |
        --------
        """,
        """
           ------
           |    |
           |    O
           |
           |
           |
        --------
        """,
        """
           ------
           |    |
           |    O
           |    |
           |
           |
        --------
        """,
        """
           ------
           |    |
           |    O
           |   /|
           |
           |
        --------
        """,
        """
           ------
           |    |
           |    O
           |   /|\\
           |
           |
        --------
        """,
        """
           ------
           |    |
           |    O
           |   /|\\
           |   /
           |
        --------
        """,
        """
           ------
           |    |
           |    O
           |   /|\\
           |   / \\
           |
        --------
        """
    ]
    return stages[min(wrong_guesses, 6)]

# Play as human against computer
def play_human_mode(hmm, words):
    word = random.choice(words)
    guessed_letters = set()
    wrong_guesses = 0
    max_wrong = 6
    
    print("\n" + "="*50)
    print("🎮 HANGMAN GAME - You vs AI Word")
    print("="*50)
    print(f"Word length: {len(word)} letters")
    print("Guess the word letter by letter!\n")
    
    while wrong_guesses < max_wrong:
        # Show current state
        masked = ''.join([c if c in guessed_letters else '_' for c in word])
        print(display_hangman(wrong_guesses))
        print(f"\nWord: {' '.join(masked)}")
        print(f"Wrong guesses: {wrong_guesses}/{max_wrong}")
        print(f"Guessed letters: {' '.join(sorted(guessed_letters)) if guessed_letters else 'none'}")
        
        # Check win
        if '_' not in masked:
            print("\n🎉 YOU WIN! 🎉")
            print(f"The word was: {word}")
            return True
        
        # Get guess
        guess = input("\nGuess a letter: ").lower().strip()
        
        if len(guess) != 1 or not guess.isalpha():
            print("❌ Please enter a single letter!")
            continue
        
        if guess in guessed_letters:
            print("❌ You already guessed that letter!")
            continue
        
        guessed_letters.add(guess)
        
        if guess in word:
            print(f"✓ Good guess! '{guess}' is in the word!")
        else:
            print(f"✗ Sorry, '{guess}' is not in the word.")
            wrong_guesses += 1
    
    # Lost
    print(display_hangman(wrong_guesses))
    print(f"\n💀 GAME OVER! The word was: {word}")
    return False

# Watch AI play
def watch_ai_play(hmm, words, num_games=5):
    print("\n" + "="*50)
    print("🤖 AI PLAYING HANGMAN")
    print("="*50)
    
    wins = 0
    total_wrong = 0
    
    for game_num in range(num_games):
        word = random.choice(words)
        guessed_letters = set()
        wrong_guesses = 0
        max_wrong = 6
        
        print(f"\n--- Game {game_num + 1}/{num_games} ---")
        print(f"Secret word: {'*' * len(word)} ({len(word)} letters)")
        
        while wrong_guesses < max_wrong:
            masked = ''.join([c if c in guessed_letters else '_' for c in word])
            
            if '_' not in masked:
                print(f"✓ AI WIN! Word: {word}")
                wins += 1
                break
            
            # AI makes guess
            guess = hmm.get_best_guess(masked, guessed_letters)
            guessed_letters.add(guess)
            
            if guess in word:
                print(f"  AI guessed '{guess}' ✓ → {' '.join(masked)}")
            else:
                wrong_guesses += 1
                print(f"  AI guessed '{guess}' ✗ (Wrong: {wrong_guesses}/{max_wrong})")
        
        if wrong_guesses >= max_wrong:
            print(f"✗ AI LOST! Word was: {word}")
        
        total_wrong += wrong_guesses
    
    print("\n" + "="*50)
    print("AI PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Games: {num_games}")
    print(f"Wins: {wins} ({100*wins/num_games:.1f}%)")
    print(f"Losses: {num_games - wins}")
    print(f"Avg wrong guesses: {total_wrong/num_games:.2f}")
    print("="*50)

# AI vs AI battle
def ai_battle(hmm, words):
    print("\n" + "="*50)
    print("⚔️  AI vs RANDOM GUESSER")
    print("="*50)
    
    num_games = 20
    hmm_wins = 0
    random_wins = 0
    
    for game_num in range(num_games):
        word = random.choice(words)
        
        # HMM AI plays
        guessed = set()
        wrong = 0
        while wrong < 6:
            masked = ''.join([c if c in guessed else '_' for c in word])
            if '_' not in masked:
                hmm_wins += 1
                break
            guess = hmm.get_best_guess(masked, guessed)
            guessed.add(guess)
            if guess not in word:
                wrong += 1
        
        # Random AI plays
        guessed = set()
        wrong = 0
        alphabet = list('abcdefghijklmnopqrstuvwxyz')
        random.shuffle(alphabet)
        while wrong < 6:
            masked = ''.join([c if c in guessed else '_' for c in word])
            if '_' not in masked:
                random_wins += 1
                break
            guess = [c for c in alphabet if c not in guessed][0]
            guessed.add(guess)
            if guess not in word:
                wrong += 1
    
    print(f"\nResults after {num_games} games:")
    print(f"  HMM AI wins: {hmm_wins} ({100*hmm_wins/num_games:.1f}%)")
    print(f"  Random AI wins: {random_wins} ({100*random_wins/num_games:.1f}%)")
    print("="*50)

# Main menu
def main():
    print("\n" + "🎲 "*20)
    print(" "*15 + "HANGMAN GAME")
    print("🎲 "*20)
    
    # Load resources
    print("\nLoading AI model...")
    hmm = load_hmm()
    if hmm is None:
        return
    
    print("Loading word list...")
    words = load_words()
    print(f"✓ Loaded {len(words)} words\n")
    
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. Play Hangman (You vs AI)")
        print("2. Watch AI Play (5 games)")
        print("3. Watch AI Play (20 games)")
        print("4. AI Battle (HMM vs Random)")
        print("5. Quick Demo (1 AI game)")
        print("6. Exit")
        print("="*50)
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            play_human_mode(hmm, words)
        elif choice == '2':
            watch_ai_play(hmm, words, num_games=5)
        elif choice == '3':
            watch_ai_play(hmm, words, num_games=20)
        elif choice == '4':
            ai_battle(hmm, words)
        elif choice == '5':
            watch_ai_play(hmm, words, num_games=1)
        elif choice == '6':
            print("\n👋 Thanks for playing!")
            break
        else:
            print("❌ Invalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()
