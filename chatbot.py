import random
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Load intents data from JSON file
with open('data.json', 'r') as json_data:
    intents = json.load(json_data)

# Extract data from JSON
X = []
y = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        X.append(pattern)
        y.append(intent['tag'])

# Vectorize data using TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Map labels to numerical values
labels = list(set(y))
label_to_idx = {label: idx for idx, label in enumerate(labels)}
y_numeric = np.array([label_to_idx[label] for label in y])

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1, 10]
}

# Create an SVM Model
svm_model = SVC()
grid_search = GridSearchCV(svm_model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_tfidf, y_numeric)

# Get the best model from GridSearchCV
best_svm_model = grid_search.best_estimator_

# Function to classify user input intent
def classify_intent(user_input):
    user_input_tfidf = vectorizer.transform([user_input])
    predicted_label_numeric = best_svm_model.predict(user_input_tfidf)
    predicted_label = labels[predicted_label_numeric[0]]
    return predicted_label

# Function to simulate the chatbot conversation
def chatbot_loop(user_input, intent):
    bot_name = "Faculty"
    
    user_input_tfidf = vectorizer.transform([user_input])
    similarity_scores = user_input_tfidf.dot(X_tfidf.T).toarray()[0]
    max_similarity = np.max(similarity_scores)

    if max_similarity < 0.5:
        return f"{bot_name}: I do not understand..."

    for intent_data in intents['intents']:
        if intent_data['tag'] == intent:
            responses = intent_data['responses']
            response = f"{bot_name}: " + random.choice(responses)
            return response
    
    return f"{bot_name}: I do not understand..."
