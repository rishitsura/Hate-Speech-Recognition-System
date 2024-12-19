import streamlit as st
import numpy as np
import pandas as pd
import nltk
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords

# Ensure NLTK resources are available
nltk.download('stopwords')

# Load and preprocess data
data = pd.read_csv('twitter.csv')
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
data = data[["tweet", "labels"]]

stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

data["tweet"] = data["tweet"].apply(clean)

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

# Streamlit app
st.title('Hate Speech Detection')

st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

user_input = st.text_input('Enter a Text: (Example: I hate you,  I love you, Shut the fuck up!)')

if st.button('Predict'):
    if user_input:
        data = cv.transform([user_input]).toarray()
        output = clf.predict(data)
        st.write(f'Prediction: {output[0]}')
    else:
        st.write("Please enter text to classify.")