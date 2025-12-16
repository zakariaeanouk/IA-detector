from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from stop_words import get_stop_words
from nltk.corpus import stopwords
from contractions import fix
import string
import nltk
import re

nltk.data.path.append('./nltk_data')

stop_words = list(get_stop_words('en'))
nltk_words = list(stopwords.words('english'))

all_stopwords = stop_words + nltk_words
all_stopwords= set(all_stopwords)


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


