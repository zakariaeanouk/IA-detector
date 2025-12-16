from nltk import word_tokenize, sent_tokenize, word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from stop_words import get_stop_words
from nltk.corpus import stopwords
from collections import Counter
from textblob import TextBlob
from contractions import fix
from nltk import ngrams
import pandas as pd
import string
import nltk
import re


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')

"""# cleaning functions"""

def remove_html_tags(text):
  """
    removes html tags.
  """
  cleaned_text = re.sub(r'<.*?>', ' ', text)
  return cleaned_text

def remove_urls(text):
    """
      removes urls.
    """
    cleaned_text = re.sub(r'\b(?:http|https|www)\S+', ' ', text, flags=re.IGNORECASE)
    return cleaned_text

def remove_emails(text):
    """
      removes emails.
    """
    cleaned_text = re.sub(r'\S+@\S+', ' ', text)
    return cleaned_text


def remove_mentions(text):
    """
      removes mentions.
    """
    cleaned_text = re.sub(r'@\w+', ' ', text)
    return cleaned_text

def remove_hashtags(text):
    """
      removes hashtags.
    """
    cleaned_text = re.sub(r'#\w+', ' ', text)
    return cleaned_text

def remove_none_english_characters(text):
    """
      removes none english characters & emojis & ponctuation.
    """
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return cleaned_text


def noise_cleaning(text):
  """
    performs noise cleaning on text.
  """
  cleaned_text = remove_html_tags(text)
  cleaned_text = remove_urls(cleaned_text)
  cleaned_text = remove_emails(cleaned_text)
  cleaned_text = remove_mentions(cleaned_text)
  cleaned_text = remove_hashtags(cleaned_text)
  cleaned_text = remove_none_english_characters(cleaned_text)

  return cleaned_text

def expand_contractions(text):
    """
      expands contractions in the text (don't -> do not)
    """
    return fix(text)


def standardize_ordinal(text):
    """
      standardize ordinal numbers in text.
      from 12th -> 12, 1st -> 1 , etc...
    """
    cleaned_text = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', text)
    return cleaned_text


def standardization(text):
  """
    performs standardization on text.
  """
  cleaned_text = standardize_ordinal(text)
  cleaned_text = expand_contractions(cleaned_text)

  return cleaned_text

def remove_numbers(text):
  """
    removes numbers.
  """
  cleaned_text = re.sub(r'\d+', ' ', text)
  return cleaned_text



def remove_newline(text):
  """
    removes newline characters.
  """
  cleaned_text = re.sub(r'\n+', ' ', text)
  return cleaned_text



def remove_duplicated_spaces(text):
  """
    removes duplicated spaces.
  """
  cleaned_text = re.sub(r'\s+',' ', text)
  cleaned_text = cleaned_text.strip()

  return cleaned_text


def lower_case(text):
  """
    converts text to lower case.
  """
  cleaned_text = text.lower()
  return cleaned_text


def remove_punctuation(text):
    """
    Removes punctuation except periods and replaces consecutive periods with a single period.
    """
    translator = str.maketrans('', '', string.punctuation.replace('.', ''))
    cleaned_text = text.translate(translator)

    cleaned_text = re.sub(r'\.{2,}', '', cleaned_text)
    return cleaned_text



def remove_time_formats(text):
  """
    removes time formats am and pm only.
  """
  cleaned_text = re.sub(r'\s(am|pm)\b', '', text)
  return cleaned_text



def remove_alone_chars(text):
  """
    removes single characters.
  """
  cleaned_text = re.sub(r'\b\w\b', '', text)
  return cleaned_text



def basic_cleaning(text):
  """
    performs basic cleaning on text.
  """

  cleaned_text = remove_numbers(text)
  cleaned_text = remove_newline(cleaned_text)
  cleaned_text = lower_case(cleaned_text)
  cleaned_text = remove_time_formats(cleaned_text)
  cleaned_text = remove_alone_chars(cleaned_text)
  cleaned_text = remove_punctuation(cleaned_text)
  cleaned_text = remove_duplicated_spaces(cleaned_text)


  return cleaned_text

def tokenize(text):
  """
    tokenizes text.
  """
  tokens = word_tokenize(text)
  return tokens


stop_words = list(get_stop_words('en'))
nltk_words = list(stopwords.words('english'))

all_stopwords = stop_words + nltk_words
all_stopwords= set(all_stopwords)

def remove_stop_words(tokens):
  """
    removes stop words from tokens.
  """
  cleaned_tokens = [token for token in tokens if token not in all_stopwords]
  return cleaned_tokens


def lemmatize_text(tokens):
  """
    lemmatizes tokens.
  """
  lemmatizer = WordNetLemmatizer()
  lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
  cleaned_text = ' '.join(lemmatized_tokens)
  return cleaned_text


def text_tok_stop_lem(text):
  """
    performs tokenization, stop word removal and lemmatization on text.
  """
  tokens = tokenize(text)
  cleaned_tokens = remove_stop_words(tokens)
  cleaned_text = lemmatize_text(cleaned_tokens)

  return cleaned_text

def perform_all_cleaning(text):
  """
    performs all cleaning on text.
  """
  cleaned = noise_cleaning(text)
  cleaned = standardization(cleaned)
  cleaned = basic_cleaning(cleaned)
  cleaned = text_tok_stop_lem(cleaned)

  return cleaned

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

import pickle


with open('logistic_regression_model.pkl', 'rb') as mo:
    model = pickle.load(mo)

with open('model_vectorizer.pkl', 'rb') as ve:
    vectorizer = pickle.load(ve)

with open('scaler.pkl', 'rb') as sc:
    scaler = pickle.load(sc)

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text2text-generation", model="google/flan-t5-large")

initial_text = "Write a story about a robot who discovers a hidden talent for painting."
generated_text = pipe(initial_text, max_length=400)[0]['generated_text']
print(generated_text)



import numpy as np




text = generated_text

cleaned_text = perform_all_cleaning(text)

avg_length = avg_word_length(cleaned_text)
ttr = calculate_type_token_ratio(cleaned_text)
hapax_count = count_hapax_legomena(cleaned_text)
repetition_rate = calculate_repetition_rate(cleaned_text)


linguistic_features = np.array([[repetition_rate, avg_length, ttr, hapax_count]])

feature_names = ['repetition_rate', 'avg_word_length', 'type_token_ratio', 'hapax_legomena']

vectorized_text = vectorizer.transform([cleaned_text]).toarray()
df_vectorized_text = pd.DataFrame(vectorized_text, columns=vectorizer.get_feature_names_out())



final_features = np.concatenate([df_vectorized_text, linguistic_features], axis=1)

final_features_df = pd.DataFrame(final_features, columns=list(df_vectorized_text.columns) + feature_names)

normalized_features = scaler.transform(final_features_df)

normalized_features_df = pd.DataFrame(normalized_features, columns=final_features_df.columns)

out = model.predict(normalized_features_df)

proba = model.predict_proba(normalized_features_df)

print(f"Predicted label: {'Human' if out[0] == 0 else  'Ai'}")

print(f"Probabilities : {proba[0]}")

import numpy as np


## testing with attention is all you need's abstract

text = """
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
"""

cleaned_text = perform_all_cleaning(text)

avg_length = avg_word_length(cleaned_text)
ttr = calculate_type_token_ratio(cleaned_text)
hapax_count = count_hapax_legomena(cleaned_text)
repetition_rate = calculate_repetition_rate(cleaned_text)


linguistic_features = np.array([[repetition_rate, avg_length, ttr, hapax_count]])

feature_names = ['repetition_rate', 'avg_word_length', 'type_token_ratio', 'hapax_legomena']

vectorized_text = vectorizer.transform([cleaned_text]).toarray()
df_vectorized_text = pd.DataFrame(vectorized_text, columns=vectorizer.get_feature_names_out())



final_features = np.concatenate([df_vectorized_text, linguistic_features], axis=1)

final_features_df = pd.DataFrame(final_features, columns=list(df_vectorized_text.columns) + feature_names)

normalized_features = scaler.transform(final_features_df)

normalized_features_df = pd.DataFrame(normalized_features, columns=final_features_df.columns)

out = model.predict(normalized_features_df)

proba = model.predict_proba(normalized_features_df)

print(f"Predicted label: {'Human' if out[0] == 0 else  'Ai'}")
print(f"Probabilities : {proba[0]}")