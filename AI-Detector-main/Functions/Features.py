from nltk import word_tokenize
from collections import Counter

def avg_word_length(text):
    """
    calculates the average word length in a given text.
    """
    words = word_tokenize(text)
    return sum(len(word) for word in words) / len(words)



def calculate_type_token_ratio(text):
    """
    calculate the type-token ratio (TTR) for lexical richness.
    """
    tokens = word_tokenize(text)
    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens) if tokens else 0


def count_hapax_legomena(text):
    """
    calculate number of words that are used just one time.
    """
    words = word_tokenize(text)

    word_counts = Counter(words)

    hapax_legomena = [word for word, count in word_counts.items() if count == 1]

    return len(hapax_legomena)


def calculate_repetition_rate(text):
    """
    calculate the repetition rate in text.
    """
    words = word_tokenize(text)

    word_counts = Counter(words)

    repeated_words = sum(1 for count in word_counts.values() if count > 1)

    total_words = len(words)

    if total_words == 0:
        return 0

    repetition_rate = repeated_words / total_words

    return repetition_rate
