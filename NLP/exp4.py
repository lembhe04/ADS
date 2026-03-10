# Finite State Transducer Simulation for Noun Inflection Parsing

# Irregular nouns dictionary (only unavoidable exceptions)
irregular_plurals = {
    "geese": "goose",
    "mice": "mouse",
    "men": "man",
    "women": "woman",
    "children": "child",
    "oxen": "ox"
}

def morphological_parser(word):
    original_word = word  # Keep original for display
    word = word.lower()
    
    # Check irregular forms first
    if word in irregular_plurals:
        root = irregular_plurals[word]
        return f"Intermediate Output: {root} +N +PL\nFinal Output: {root.upper()} +N +PL"
    elif word in irregular_plurals.values():
        return f"Intermediate Output: {word} +N +SG\nFinal Output: {word.upper()} +N +SG"
    
    # Rule: words ending in "ies" → change to "y" (city → cities)
    if word.endswith("ies") and len(word) > 3:
        root = word[:-3] + "y"
        return f"Intermediate Output: {root} +N +PL\nFinal Output: {root.upper()} +N +PL"
    
    # Rule: words ending in "ves" → change to "f" or "fe"
    if word.endswith("ves") and len(word) > 3:
        # Common fe-ending words
        fe_words = ["knife", "wife", "life"]
        possible_root_fe = word[:-3] + "fe"
        if possible_root_fe in fe_words:
            root = possible_root_fe
        else:
            root = word[:-3] + "f"
        return f"Intermediate Output: {root} +N +PL\nFinal Output: {root.upper()} +N +PL"
    
    # Rule: words ending in "es" after x, o, ch, sh → remove "es"
    if word.endswith("es") and (word[-3] in ["x", "o"] or word[-4:-2] in ["ch", "sh"]):
        root = word[:-2]
        return f"Intermediate Output: {root} +N +PL\nFinal Output: {root.upper()} +N +PL"
    
    # Rule: regular plurals ending in "s" → remove "s"
    if word.endswith("s") and len(word) > 1:
        root = word[:-1]
        return f"Intermediate Output: {root} +N +PL\nFinal Output: {root.upper()} +N +PL"
    
    # Otherwise, treat as singular noun
    return f"Intermediate Output: {word} +N +SG\nFinal Output: {word.upper()} +N +SG"

# -------- Main Program --------
if __name__ == "__main__":
    user_input = input("Enter a noun: ").strip()
    print(morphological_parser(user_input))
