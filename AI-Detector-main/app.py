from flask import Flask, render_template, request, redirect, url_for
from Functions.Cleaning import *
from Functions.Features import *
import pandas as pd
import numpy as np

import pickle

app = Flask(__name__)

def init():
    global model, vectorizer, scaler
    with open('./Model/logistic_regression_model.pkl', 'rb') as m:
        model = pickle.load(m)
    with open('./Model/model_vectorizer.pkl', 'rb') as v:
        vectorizer = pickle.load(v)
    with open('./Model/scaler.pkl', 'rb') as s:
        scaler = pickle.load(s)

@app.route('/')
def index():
    init()
    prediction = request.args.get('prediction', '')
    proba = request.args.get('proba', 0)
    text = request.args.get('text', '')
    
    return render_template('index.html', prediction=prediction, proba=proba, text=text)



@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']
    
    cleaned_text = perform_all_cleaning(text)

    avg_length = avg_word_length(text)
    ttr = calculate_type_token_ratio(text)
    hapax_count = count_hapax_legomena(text)
    repetition_rate = calculate_repetition_rate(text)


    linguistic_features = np.array([[repetition_rate, avg_length, ttr, hapax_count]])

    feature_names = ['repetition_rate', 'avg_word_length', 'type_token_ratio', 'hapax_legomena']

    vectorized_text = vectorizer.transform([cleaned_text]).toarray()

    df_vectorized_text = pd.DataFrame(vectorized_text, columns=vectorizer.get_feature_names_out())

    final_features = np.concatenate([df_vectorized_text, linguistic_features], axis=1)

    final_features_df = pd.DataFrame(final_features, columns=list(df_vectorized_text.columns) + feature_names)

    normalized_features = scaler.transform(final_features_df)

    normalized_features_df = pd.DataFrame(normalized_features, columns=final_features_df.columns) 

    out = model.predict(normalized_features_df)
    prediction = 'Human' if out[0] == 0 else 'AI'

    proba = model.predict_proba(normalized_features_df)
    proba = proba[0][0] if out[0] == 0 else proba[0][1]
    proba = round(proba * 100, 2)

    return redirect(url_for('index', prediction=prediction, proba=proba, text=text))



if __name__ == '__main__':
    app.run(debug=True)
