from flask import Flask, render_template, request
from elasticsearch import Elasticsearch
from model import features, x_train, get_image_vector
import json
import os

app = Flask(__name__)

# Initialize Elasticsearch client
es = Elasticsearch()

# Vector dimension for image feature vectors
VECTOR_DIMENSION = 2048

# Create index and mappings
if not es.indices.exists(index='images'):
    mapping = {
        "mappings": {
            "properties": {
                "filename": {"type": "keyword"},
                "vector": {"type": "dense_vector", "dims": VECTOR_DIMENSION}
            }
        }
    }
    es.indices.create(index='images', body=mapping)

    # Index all images in the vector database
    for i in range(len(x_train)):
        doc = {
            'filename': f'image_{i}.png',
            'vector': features[i].tolist()
        }
        es.index(index='images', id=i, body=doc)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Save uploaded image to disk
    file = request.files['file']
    filename = file.filename
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    # Get vector representation of the uploaded image
    vector = get_image_vector(file)

    # Render the search page with query image
    return render_template('search.html', filename=filename, vector=json.dumps(vector.tolist()))

@app.route('/search', methods=['POST'])
def search():
    # Get query image vector representation
    query_vector = json.loads(request.form['vector'])

    # Search for similar images in the vector database
    search_vector = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.queryVector, 'vector') + 1.0",
                "params": {"queryVector": query_vector}
            }
        }
    }
    search_query = {
        "query": {
            "function_score": {
                "query": {"match_all": {}},
                "boost": "5",
                "functions": [search_vector]
            }
        },
        "size": 9,
        "_source": ["filename"]
    }
    search_results = es.search(index='images', body=search_query)['hits']['hits']

    # Extract image IDs and scores from search results
    results = []
    for result in search_results:
        image_id = result['_id']
        score = result['_score']
        results.append((image_id, score))

    # Render the search results page
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
