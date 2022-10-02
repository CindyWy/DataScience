#flask API, Swagger UI
import re
from unicodedata import name
from flask import Flask, jsonify, request #import objects from the Flask model

from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

app = Flask(__name__) #define app using Flask

app.json_encoder = LazyJSONEncoder

swagger_template = dict(
    info = {
        'title': LazyString(lambda:'RESTFUL API Documentation with Swagger for Quiz_3'),
        'version': LazyString(lambda:'1.0.0'),
        'description': LazyString(lambda:'API Documentation for GET, POST, PUT, DELETE Method (API source code modify from Pretty Printed Github)')
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

languages = [{'name' : 'JavaScript'}, {'name' : 'Python'}, {'name' : 'Ruby'}, {'name' : 'Go'}]

#===================================================================================

@swag_from("docs/get_hello.yml", methods=['GET'])
@app.route('/', methods=['GET'])
def test():
    json_response = { 
		'message' : 'It works!'
        }

    response_data = jsonify(json_response)
    return response_data

#===================================================================================

@swag_from("docs/get_returnAll.yml", methods=['GET'])
@app.route('/lang', methods=['GET'])
def returnAll():
    json_response = { 
		'languages' : languages
        }

    response_data = jsonify(json_response)
    return response_data

#===================================================================================

@swag_from("docs/get_returnOne.yml", methods=['GET'])
@app.route('/lang/<string:name>', methods=['GET'])
def returnOne(name):
	#language = request.form.get('name')
	langs = [language for language in languages if language['name'] == name]
	json_response = { 
		'language' : langs[0]
        }

	response_data = jsonify(json_response)
	return response_data

#===================================================================================

@swag_from("docs/post_addOne.yml", methods=['POST'])
@app.route('/lang', methods=['POST'])
def addOne():
    text = request.form.get('language')

    language = {'name' : text}
    languages.append(language)
    
    json_response = { 
	 	'languages' : languages
         }

    response_data = jsonify(json_response)
    return response_data

#====================================================================================

@swag_from("docs/put_editOne.yml", methods=['PUT'])
@app.route('/lang/<string:name>', methods=['PUT'])
def editOne(name):
	#name = request.form.get('name')
	name2 = request.form.get('name2')

	langs = [language for language in languages if language['name'] == name]
	langs[0]['name'] = name2
	
	json_response = { 
		'language' : langs[0]
        }

	response_data = jsonify(json_response)
	return response_data

#===================================================================================

@swag_from("docs/delete_removeOne.yml", methods=['DELETE'])
@app.route('/lang/<string:name>', methods=['DELETE'])
def removeOne(name):
	
    langs = [language for language in languages if language['name'] == name]

    languages.remove(langs[0])

    json_response = { 
		'languages' : languages
        }

    response_data = jsonify(json_response)
    return response_data

#===================================================================================

if __name__ == '__main__':
	app.run(debug=True, port=8080) #run app on port 8080 in debug mode