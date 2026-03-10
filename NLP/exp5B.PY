# Experiment 5.B - Hindi Stemming using Stanza (works on Python 3.11)
import stanza

# Download Hindi model (only the first time)
stanza.download('hi')

# Create NLP pipeline for Hindi
nlp = stanza.Pipeline('hi')

# Take user input
text = input("Enter Hindi words separated by spaces: ").strip()

# Process the text
doc = nlp(text)

print("\n=== Hindi Stemming (Lemmatization) Results ===")
for sent in doc.sentences:
    for word in sent.words:
        print(f"{word.text} → {word.lemma}")
