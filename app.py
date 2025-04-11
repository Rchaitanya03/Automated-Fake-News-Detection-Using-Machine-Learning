import pickle
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer

# Load the trained Multinomial Naive Bayes model and vectorizer
with open('nb_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('count_vectorizer.pkl', 'rb') as f:
    count_vectorizer = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input from the user
        text_input = request.form['text_input']
        
        # Clean the input text
        cleaned_text = clean_text(text_input)
        
        # Vectorize the cleaned text using the saved vectorizer
        vectorized_text = count_vectorizer.transform([cleaned_text])
        
        # Predict using the Naive Bayes model
        prediction_prob = nb_model.predict_proba(vectorized_text)[:, 1]
        prediction = nb_model.predict(vectorized_text)[0]
        
        # Display result
        result = "True News" if prediction == 1 else "Fake News"
        return render_template('result.html', prediction=result, probability=prediction_prob[0])

def clean_text(text):
    """Clean the input text."""
    import re
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

if __name__ == '__main__':
    app.run(debug=True)
