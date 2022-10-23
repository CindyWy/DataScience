#flask API, Swagger UI
from importlib.resources import path
from unicodedata import name
from flask import Flask, jsonify, request #import objects from the Flask model

from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
import pandas as pd
import sqlite3 as sql
import re

app = Flask(__name__) #define app using Flask

app.json_encoder = LazyJSONEncoder

swagger_template = dict(
    info = {
        'title': LazyString(lambda:'API Tweeter Cleansing with Swagger for Challange Gold'),
        'version': LazyString(lambda:'1.0.0'),
        'description': LazyString(lambda:'API Documentation for GET, POST, PUT, DELETE Method (Source code by Cindy)')
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
#Connect to all database

conn1 = sql.connect('dictionary_db.db')
conn2 = sql.connect('data_db.db')

#===================================================================================
#Call table Abusive, Kamus Alay, dan data which contain old_tweet

df1 = pd.read_sql_query(''' select old_tweet from tweeter_file;''', conn2)

#Panggil Abusive & Kamus Alay
df2 = pd.read_sql_query(''' select * from abusive_tb;''', conn1)
df2['label'] = 'CENSORED'

df3 = pd.read_sql_query(''' select * from kamusalay_tb;''', conn1)

#===================================================================================
#Cleansing Function which normalize the alay words and then CENSORED the abusive words

df1['id'] = range(0,len(df1))
df1['id'] = df1['id'].astype('int')
df1.index = df1['id']

def frame(df1):
    df1_process = df1
    df1_process['new_tweet'] = df1_process['old_tweet'].str.lower()
    df1_process['new_tweet'] = df1_process['new_tweet'].str.strip()
    df1_process['new_tweet'] = df1_process['new_tweet'].replace(r'([^\s\w]|_)+', '', regex=True)

#Normalize based on kamus Alay
    df3_dict = df3.set_index('alay',drop=True).to_dict()['normal']
    pattern = r"\b(" + "|".join(re.escape(k) for k in df3_dict) + r")\b"
    df1_process['new_tweet'] = df1_process['new_tweet'].str.replace(pattern, lambda m: df3_dict[m.group(0)], regex=True)

#Censored based on abusive
    df2_dict = df2.set_index('ABUSIVE',drop=True).to_dict()['label']
    pattern = r"\b(" + "|".join(re.escape(k) for k in df2_dict) + r")\b"
    df1_process['new_tweet'] = df1_process['new_tweet'].str.replace(pattern, lambda m: df2_dict[m.group(0)], regex=True)

    json = df1_process.to_dict(orient='index')
    return json

#===================================================================================

@swag_from("docs/get_hello.yml", methods=['GET'])
@app.route('/', methods=['GET'])
def test():
    return jsonify({'message' : 'It works!'})

#===================================================================================

@swag_from("docs/get_returnAll.yml", methods=['GET'])
@app.route('/lang', methods=['GET'])
def returnAll():
    json = frame(df1)
    return jsonify(json)

#===================================================================================

@swag_from("docs/get_returnOne.yml", methods=['GET'])
@app.route('/lang/<id>', methods=['GET'])
def returnOne(id):
    json = frame(df1)
    
    id = int(id)
    json = json[id]

    return jsonify(json)

#===================================================================================

@swag_from("docs/post_addOne.yml", methods=['POST'])
@app.route('/lang', methods=['POST'])
def addOne():
    old_tweet = {'old_tweet': request.json['old_tweet']}
    df1.loc[len(df1) + 1] = [old_tweet['old_tweet'], max(df1['id'])+1]
    df1.index = df1['id']
    
    json = frame(df1)

    id = max(df1.index)
    json = json[id]

    return jsonify(json)

#====================================================================================

@swag_from("docs/put_editOne.yml", methods=['PUT'])
@app.route('/lang/<id>', methods=['PUT'])
def editOne(id):
    old_tweet = {'old_tweet': request.json['old_tweet']}
    id = int(id)

    if id in df1['id'].tolist():
        df1.loc[id] = [old_tweet['old_tweet'], id]

        json = frame(df1)

        json = json[id]

        return jsonify(json)
    else :
        return 'input ulang'

#===================================================================================

@swag_from("docs/delete_removeOne.yml", methods=['DELETE'])
@app.route('/lang/<id>', methods=['DELETE'])
def removeOne(id):
    global df1
    id = int(id)
    
    df1 = df1.drop(id)
    
    json = frame(df1)
    return jsonify(json)

#===================================================================================

@swag_from("docs/post_uploadOne.yml", methods=['POST'])
@app.route('/uploader', methods=['POST'])
def uploadOne():
    test_file = request.files['file']

    try:
        df = pd.read_csv(test_file)
        json = frame(df)
        return jsonify(json)
    except ValueError:
      return 'Please only input file with extension csv, with one column and named the header : old_tweet'

#===================================================================================

if __name__ == '__main__':
	app.run(debug=True, port=8080) #run app on port 8080 in debug mode