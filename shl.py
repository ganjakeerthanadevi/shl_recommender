import requests
from bs4 import BeautifulSoup
url = "https://www.shl.com/solutions/products/product-catalog/"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


assessments = pd.DataFrame({
    'name': ['Java Test', 'Python Test', 'Cognitive Test'],
    'description': [
        'Measures Java programming skills',
        'Evaluates Python coding ability', 
        'Assesses problem-solving skills'
    ],
    'duration': [30, 45, 60]
})


assessment_descriptions = assessments['description'].tolist()


model = SentenceTransformer('all-MiniLM-L6-v2')
assessment_embeddings = model.encode(assessment_descriptions)


def recommend_assessments(query, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, assessment_embeddings)
    top_indices = similarities.argsort()[0][-top_k:][::-1]
    return assessments.iloc[top_indices]


print(recommend_assessments("Need a test for programming skills"))
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
assessment_embeddings = model.encode(assessment_descriptions)
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



def recommend_assessments(query, top_k=10):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, assessment_embeddings)
    top_indices = similarities.argsort()[0][-top_k:][::-1]
    return assessments.iloc[top_indices]
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
# 1. Initialize Flask app (corrected)
app = Flask(__name__)  # Capital F in Flask and proper __name_

# 2. Sample assessment data
assessments = pd.DataFrame({
    'name': ['Java Test', 'Python Test', 'Cognitive Test'],
    'description': ['Measures Java skills', 'Evaluates Python', 'Problem-solving'],
    'url': ['https://test1', 'https://test2', 'https://test3'],
    'duration': [30, 45, 60]
})

# 3. Load model and create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
assessment_embeddings = model.encode(assessments['description'].tolist())

# 4. Recommendation function (fixed)
def recommend_assessments(query, top_k=3):
    query_embedding = model.encode([query])  # Fixed from model_encode
    similarities = cosine_similarity(query_embedding, assessment_embeddings)  # Fixed spelling
    top_indices = similarities.argsort()[0][-top_k:][::-1]  # Fixed indices
    return assessments.iloc[top_indices]  # Fixed from loc to iloc

# 5. Flask routes
@app.route('/')
def home():
    return "SHL Assessment Recommender API - Use /health or /recommend endpoints"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})  # Fixed curly braces

@app.route('/recommend', methods=['POST'])  # Fixed methods syntax
def recommend():
    data = request.get_json()
    query = data.get('query')
    recommendations = recommend_assessments(query)  # Call the function
    return jsonify({
        'recommended_assessments': recommendations.to_dict('records')
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)  # Added explicit port

import streamlit as st

st.title("SHL Assessment Recommender")
query = st.text_input("Enter job description or query:")
if query:
    recommendations = recommend_assessments(query)
    st.table(recommendations)
def recall_at_k(true_positives, recommended, k):
    tp = len(set(true_positives) & set(recommended[:k]))
    return tp / len(true_positives)
    
def map_at_k(all_true_positives, all_recommended, k):
    ap_sum = 0
    for tp, rec in zip(all_true_positives, all_recommended):
        precision_sum = 0
        relevant_count = 0
        for i in range(min(k, len(rec))):  # Fixed: Changed ; to :
            if rec[i] in tp:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        ap_sum += precision_sum / min(len(tp), k)  # Note indentation
    return ap_sum / len(all_true_positives)  # Final return
