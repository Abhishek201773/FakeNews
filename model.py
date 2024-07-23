import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import nltk

# Download NLTK resources
nltk.download('stopwords')

# Load the dataset
try:
    news_dataset = pd.read_csv('train.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("The file 'train.csv' was not found. Please check the directory and file name.")
    import sys
    sys.exit()

# Fill missing values
news_dataset = news_dataset.fillna('')

# Create a new column 'content' by combining 'author' and 'title'
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Define feature and label
X = news_dataset['content']
Y = news_dataset['label']

# Initialize the Porter Stemmer
port_stem = PorterStemmer()

# Define a function for stemming
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming
news_dataset['content'] = news_dataset['content'].apply(stemming)

# Convert textual data to feature vectors
vectorizer = TfidfVectorizer()
vectorizer.fit(news_dataset['content'])

X = vectorizer.transform(news_dataset['content'])

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data: ', test_data_accuracy)

# Save the model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model and vectorizer saved to 'model.pkl' and 'vectorizer.pkl'.")
