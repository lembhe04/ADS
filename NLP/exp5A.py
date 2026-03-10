# Experiment 5.A - Stemming in English using NLTK
import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

# Download required NLTK data
nltk.download('punkt')

# Create stemmer objects
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

# Get input words from user
words = input("Enter words separated by spaces: ").strip().split()

print("\n=== Stemming Results ===")
print("{:<15} {:<15} {:<15} {:<15}".format("Word", "Porter", "Lancaster", "Snowball"))
print("-"*60)

for word in words:
    print("{:<15} {:<15} {:<15} {:<15}".format(
        word,
        porter.stem(word),
        lancaster.stem(word),
        snowball.stem(word)
    ))
