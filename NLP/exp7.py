import nltk
import spacy
import stanza
import pandas as pd

# Download resources for NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Load spaCy model
nlp_spacy = spacy.load("en_core_web_sm")

# Download StanfordNLP resources (first time only)
stanza.download("en")
nlp_stanford = stanza.Pipeline("en", processors="tokenize,pos")

# Input sentences
sentences = [
    "Book that flight.",
    "Hand me that book."
]

# Results container
rows = []

for sent in sentences:
    # NLTK
    tokens_nltk = nltk.word_tokenize(sent)
    tagged_nltk = nltk.pos_tag(tokens_nltk)

    # spaCy
    doc_spacy = nlp_spacy(sent)
    tagged_spacy = [(token.text, token.pos_) for token in doc_spacy]

    # StanfordNLP
    doc_stanford = nlp_stanford(sent)
    tagged_stanford = [(word.text, word.upos) for sent_out in doc_stanford.sentences for word in sent_out.words]

    # Make rows word by word
    max_len = max(len(tagged_nltk), len(tagged_spacy), len(tagged_stanford))
    for i in range(max_len):
        word_nltk = f"{tagged_nltk[i][0]}/{tagged_nltk[i][1]}" if i < len(tagged_nltk) else ""
        word_spacy = f"{tagged_spacy[i][0]}/{tagged_spacy[i][1]}" if i < len(tagged_spacy) else ""
        word_stanford = f"{tagged_stanford[i][0]}/{tagged_stanford[i][1]}" if i < len(tagged_stanford) else ""
        rows.append({
            "Sentence": sent,
            "NLTK": word_nltk,
            "spaCy": word_spacy,
            "StanfordNLP": word_stanford
        })
    rows.append({"Sentence": "----", "NLTK": "----", "spaCy": "----", "StanfordNLP": "----"})  # separator

# Create DataFrame
df = pd.DataFrame(rows)
print(df.to_string(index=False))
