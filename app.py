from flask import Flask, request, render_template
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download NLTK resources (stopwords)
nltk.download('stopwords')

app = Flask(__name__)

# Load the model and vectorizer from disk
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Initialize the Porter Stemmer
port_stem = PorterStemmer()

# Function to perform text preprocessing (stemming and stopwords removal)
def preprocess_text(content):
    # Remove non-alphabetical characters
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    # Convert to lowercase
    stemmed_content = stemmed_content.lower()
    # Tokenize into words
    stemmed_content = stemmed_content.split()
    # Apply stemming and remove stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in set(stopwords.words('english'))]
    # Join the words back into a single string
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the news content from the form
    content = request.form['content']
    
    # Preprocess the content
    preprocessed_content = preprocess_text(content)
    
    # Vectorize the preprocessed content
    vectorized_input_data = vectorizer.transform([preprocessed_content])
    
    # Make prediction using the loaded model
    prediction = model.predict(vectorized_input_data)

    # Determine the prediction result
    if prediction[0] == 0:
        result = 'The news is Real'
        image_path = 'img/real_news.jpg'  # Path to image for real news
    else:
        result = 'The news is Fake'
        image_path = 'img/fake_news.jpg'  # Path to image for fake news

    return render_template('analysis.html', prediction_text=result, image_path=image_path)

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

if __name__ == "__main__":
    app.run(debug=True)
