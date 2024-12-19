
# Hate Speech Recognition System

This project implements a Hate Speech Recognition System using a Decision Tree Classifier. The model is trained on a dataset of tweets to classify them into three categories:

- Hate Speech
- Offensive Language
- No Hate and Offensive Language

The project also includes a web interface using Streamlit for user interaction.

---

## Dataset

The dataset used in this project is a CSV file named `twitter.csv`. It contains tweets labeled as:
- `0`: Hate Speech
- `1`: Offensive Language
- `2`: No Hate and Offensive Language

The `labels` column is derived by mapping these values to the corresponding categories.

---

## Project Setup

### Dependencies

Ensure you have the following libraries installed:
- `numpy`
- `pandas`
- `sklearn`
- `nltk`
- `streamlit`

If not installed, you can use the following commands to install them:
```bash
pip install numpy pandas scikit-learn nltk streamlit
```

### Data Preparation

Ensure the `twitter.csv` file is placed in your project directory for easy access, or modify the data loading path accordingly.

---

## Preprocessing

The following preprocessing steps are applied to the text data:
1. Lowercasing the text.
2. Removing URLs, HTML tags, punctuation, and numerical values.
3. Removing stopwords using NLTK.
4. Stemming the words using NLTK's Snowball Stemmer.

---

## Model Training

The pipeline includes:
- Vectorizing the text data using `CountVectorizer`.
- Splitting the dataset into training and testing sets.
- Training a `DecisionTreeClassifier` on the training data.

### Metrics

The accuracy of the model is calculated using `accuracy_score` from `sklearn`.

---

## Interactive Interface

A Streamlit interface is created for real-time predictions. Users can input a tweet, and the model will classify it into one of the three categories.

### Launching the Interface

The interface can be launched with the following command:
```python
streamlit run app.py
```

---

## Usage

Before using the Hate Speech Detection system, you must first get the project files onto your local machine. You can either download the repository as a ZIP file or use git to clone it directly:

```bash
git clone https://github.com/yourgithubusername/hate-speech-recognition.git
```

Once you have the project files:

1. Navigate to the project directory:
```bash
cd hate-speech-recognition
```

2. Install the required dependencies:
```bash
pip install numpy pandas scikit-learn nltk streamlit
```

3. Launch the Streamlit application to interact with the model:
```bash
streamlit run app.py
```

Navigate to `localhost:8501` in your web browser to use the interface.

Enter a tweet in the Streamlit interface to check if it contains hate speech.

---

## Example

### Input
```plaintext
"This is an example tweet!"
```

### Output
```plaintext
"No Hate and Offensive Language"
```

---

## File Structure
```
app.py                          # Python script for the Streamlit application
twitter.csv                     # Dataset file
requirements.txt                # File listing the necessary Python libraries
```
