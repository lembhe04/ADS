import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download WordNet data if not already available
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def fst_parse(word):
    original = word.strip()
    if not original:
        return None, None

    lower_word = original.lower()

    # Lemmatize as a noun
    lemma = lemmatizer.lemmatize(lower_word, pos=wordnet.NOUN)

    # If lemma is different from the word → plural detected
    if lemma != lower_word:
        # Find the suffix part (plural ending)
        suffix = lower_word[len(lemma):]
        if not suffix:  # Safety
            suffix = "S"
        # Format outputs
        intermediate = f"{lemma.upper()} ^ {suffix.upper()} #"
        final_output = f"{' '.join(lemma.upper())} +N +PL"
        return intermediate, final_output

    # If lemma same as word → singular
    intermediate = f"{lemma.upper()} #"
    final_output = f"{' '.join(lemma.upper())} +N +SG"
    return intermediate, final_output

# ---------------- Main Program ----------------
if __name__ == "__main__":
    word = input("Enter a noun: ").strip()
    inter, out = fst_parse(word)
    if inter and out:
        print(f"INTERMEDIATE OUTPUT - {inter}")
        print(f"OUTPUT - {out}")
    else:
        print("Invalid input.")
