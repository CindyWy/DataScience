#flask API, Swagger UI
from importlib.resources import path
from unicodedata import name
from flask import Flask, jsonify, request #import objects from the Flask model
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

import pandas as pd
import sklearn
import re 
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sqlite3 as sql
import nltk
nltk.download('stopwords')
import json
import keras
import tensorflow.keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from collections import defaultdict
# from sklearn.model_selection import train_test_split
# from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, SimpleRNN, Activation, Dropout, BatchNormalization
# from tensorflow.keras import optimizers
# from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras import backend as K
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.corpus import stopwords
from keras.models import load_model

app = Flask(__name__) #define app using Flask

app.json_encoder = LazyJSONEncoder

swagger_template = dict(
    info = {
        'title': LazyString(lambda:'API Indonesia Tweet Sentiment Analysis with Neural Network & LSTM for Challange Platinum'),
        'version': LazyString(lambda:'1.0.0'),
        'description': LazyString(lambda:'API Documentation for POST API : File and Text (source code by Cindy)')
        }, host = LazyString(lambda: request.host)
    )

swagger_config = {
        "headers":[],
        "specs":[
            {
            "endpoint":'docs',
            "route":'/docs.json'
            }
        ],
        "static_url_path":"/flasgger_static",
        "swagger_ui":True,
        "specs_route":"/docs/"
    }

swagger = Swagger(app, template=swagger_template, config=swagger_config)

#===================================================================================
#Connect to database cleansing

conn1 = sql.connect('dictionary_db.db')
df3 = pd.read_sql_query(''' select * from kamusalay_tb;''', conn1)

#===================================================================================
#Predict Function LSTM - Text
#===================================================================================
def frame_text(string):
#Normalize and lower string
    string = string.lower()
    string = string.strip()
    string = re.sub(r'[^a-zA-z0-9]', ' ', string)
    string = re.sub(r'([^\s\w]|_)+', ' ', string)

#Normalize based on kamus Alay
    df3_dict = df3.set_index('alay',drop=True).to_dict()['normal']
    pattern = r"\b(" + "|".join(re.escape(k) for k in df3_dict) + r")\b"
    string = re.sub(pattern, lambda m: df3_dict[m.group(0)], string)

#Stopword removal based on Indonesia NLTK
    remove_stopword = lambda x : ' '.join(word for word in x.split() if word not in nltk.corpus.stopwords.words('indonesian'))
    string = remove_stopword(string)
      
#Apply LSTM model to predict
    file=open("tokenizer_lstm.pickle",'rb')
    tokenizer= pickle.load(file)

    sequences_string = tokenizer.texts_to_sequences(string)
    padded_string = pad_sequences(sequences_string)

    sentiment = ["negative", "neutral", "positive"]

    model= load_model('model_lstm.h5')
    prediction = model.predict(padded_string)
    get_sentiment=sentiment[np.argmax(prediction[0])]

    return get_sentiment

#===================================================================================
#Predict Function LSTM - Upload
def frame_upload(df):

#Normalize and lower string
    def cleansing(sent):
        string = str(sent).lower()
        string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
        string = re.sub(r'([^\s\w]|_)+', ' ', string)
        return string

    df['text'] = df.text.apply(cleansing)
    
#Normalize based on kamus Alay
    df3_dict = df3.set_index('alay',drop=True).to_dict()['normal']
    pattern = r"\b(" + "|".join(re.escape(k) for k in df3_dict) + r")\b"
    df['text'] = df['text'].str.replace(pattern, lambda m: df3_dict[m.group(0)], regex=True)

#Stopword removal based on Indonesia NLTK
    def remove_stopwords (text) :
        text =' '.join(word for word in text.split() if word not in nltk.corpus.stopwords.words('indonesian'))
        return text
    
    df['text'] = df['text'].apply(lambda x: remove_stopwords(x))
      
#Apply LSTM model to predict
    def predict (string):
        file=open("tokenizer_lstm.pickle",'rb')
        tokenizer= pickle.load(file)

        sequences_string = tokenizer.texts_to_sequences(string)
        padded_string = pad_sequences(sequences_string)

        sentiment = ["negative", "neutral", "positive"]

        model= load_model('model_lstm.h5')
        prediction = model.predict(padded_string)
        get_sentiment=sentiment[np.argmax(prediction[0])]

        return get_sentiment
    
    df['label'] = df['text'].apply(lambda x: predict(x))
    
    return df

#===================================================================================
#Predict Function NN - Text
#===================================================================================
def frame_nn_text(string):
#Normalize and lower string
    string = string.lower()
    string = string.strip()
    string = re.sub(r'[^a-zA-z0-9]', ' ', string)
    string = re.sub(r'([^\s\w]|_)+', ' ', string)

#Normalize based on kamus Alay
    df3_dict = df3.set_index('alay',drop=True).to_dict()['normal']
    pattern = r"\b(" + "|".join(re.escape(k) for k in df3_dict) + r")\b"
    string = re.sub(pattern, lambda m: df3_dict[m.group(0)], string)
     
#Apply NN model to predict
    file=open("feature_nn.p",'rb')
    count_vect = pickle.load(file)
    text = count_vect.transform([string])

    file=open("model_nn.p",'rb')
    model = pickle.load(file)

    get_sentiment = model.predict(text)[0]

    return get_sentiment

#===================================================================================
#Predict Function NN - Upload
def frame_nn_upload(df):

#Normalize and lower string
    def cleansing(sent):
        string = str(sent).lower()
        string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
        string = re.sub(r'([^\s\w]|_)+', ' ', string)
        return string

    df['text'] = df.text.apply(cleansing)
    
#Normalize based on kamus Alay
    df3_dict = df3.set_index('alay',drop=True).to_dict()['normal']
    pattern = r"\b(" + "|".join(re.escape(k) for k in df3_dict) + r")\b"
    df['text'] = df['text'].str.replace(pattern, lambda m: df3_dict[m.group(0)], regex=True)
     
#Apply NN model to predict
    def predict (string):

        # Feature Extraction
        file=open("feature_nn.p",'rb')
        count_vect = pickle.load(file)
        text = count_vect.transform([string])

        file=open("model_nn.p",'rb')
        model = pickle.load(file)

        get_sentiment = model.predict(text)[0]

        return get_sentiment
    
    df['label'] = df['text'].apply(lambda x: predict(x))
    
    return df

#===================================================================================
#POST API with LSTM MODEL
#===================================================================================

@swag_from("docs/post_uploadOne_lstm.yml", methods=['POST'])
@app.route('/uploader_LSTM', methods=['POST'])
def uploadOne_lstm():
    input_file = request.files['file']

    df = pd.read_csv(input_file, sep=';')
    df1 = frame_upload(df)
    
    conn4 = sql.connect('datatweet_db.db')
    df1.to_sql('tweet_tb', conn4, index=False, if_exists='append')
    conn4.commit()
    conn4.close()

    data = df1.to_dict(orient='index')

    return jsonify(data)

#===================================================================================

@swag_from("docs/post_addOne_lstm.yml", methods=['POST'])
@app.route('/submit_LSTM', methods=['POST'])
def addOne_lstm():

    original_text = request.form.get('text')

    get_sentiment = frame_text(original_text)
    
    conn2 = sql.connect('datatweet_db.db')
    cursor2 = conn2.cursor()
    query = "INSERT INTO tweet_tb (text, label) VALUES (?, ?)"
    values = (original_text, get_sentiment)
    cursor2.execute(query, values)
    conn2.commit()
    conn2.close()    
      
    json_response = {
        'status_code' : 200,
        'description' : "Result sentiment analysis using LSTM",
        'data' : {
            'text' : original_text,
            'sentiment' : get_sentiment
        }
    }

    response_data = jsonify(json_response)
    return response_data

#===================================================================================
#POST API with NN MODEL
#===================================================================================

@swag_from("docs/post_uploadOne_nn.yml", methods=['POST'])
@app.route('/uploader_NN', methods=['POST'])
def uploadOne_nn():
    input_file = request.files['file']

    df = pd.read_csv(input_file, sep=';')
    df1 = frame_nn_upload(df)
    
    conn4 = sql.connect('datatweet_db.db')
    df1.to_sql('tweet_tb', conn4, index=False, if_exists='append')
    conn4.commit()
    conn4.close()

    data = df1.to_dict(orient='index')

    return jsonify(data)

#===================================================================================

@swag_from("docs/post_addOne_nn.yml", methods=['POST'])
@app.route('/submit_NN', methods=['POST'])
def addOne_nn():

    original_text = request.form.get('text')
    get_sentiment = frame_nn_text(original_text)
    
    conn2 = sql.connect('datatweet_db.db')
    cursor2 = conn2.cursor()
    query = "INSERT INTO tweet_tb (text, label) VALUES (?, ?)"
    values = (original_text, get_sentiment)
    cursor2.execute(query, values)
    conn2.commit()
    conn2.close()    
      
    json_response = {
        'status_code' : 200,
        'description' : "Result sentiment analysis using NN",
        'data' : {
            'text' : original_text,
            'sentiment' : get_sentiment
        }
    }

    response_data = jsonify(json_response)
    return response_data


#===================================================================================

if __name__ == '__main__':
	app.run(debug=True, port=8080) #run app on port 8080 in debug mode