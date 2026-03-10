import nltk
from nltk.util import ngrams
from collections import defaultdict, Counter

nltk.download('punkt')
nltk.download('punkt_tab')

corpus = [
    "I like  to study AI subject.",
    "Leena like the subject of NLP.",
    "NLP is subdomain of AI.",
    "Is NLP taught in AI syllabus ?"
]

sentences = [nltk.word_tokenize(sentence.lower()) for sentence in corpus]

def build_ngram_model(sentences, n):
    model = defaultdict(Counter)
    for sentence in sentences:
        for gram in ngrams(sentence, n):
            context = gram[:-1]
            word = gram[-1]
            model[context][word] += 1
    return model

models_cache = {}

def get_model(n):
    if n not in models_cache:
        models_cache[n] = build_ngram_model(sentences, n)
    return models_cache[n]

def next_word_probabilities(context, n):
    if n < 2:
        raise ValueError("n must be >= 2")
    tokens = nltk.word_tokenize(context.lower())
    context_tokens = tuple(tokens[-(n-1):])
    model = get_model(n)
    counts = model.get(context_tokens, Counter())
    total = sum(counts.values())
    if total == 0:
        return {"No prediction": 0}
    return {w: round(c/total, 3) for w, c in counts.items()}

print("Bigram Probabilities:", next_word_probabilities("Does Leena like", n=2))
print("Trigram Probabilities:", next_word_probabilities("Leena like the", n=3))
print("4-gram Probabilities:", next_word_probabilities("Is NLP taught in ai", n=4))
