from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import re

app = Flask(__name__)

# Load the pre-trained Keras model
model = load_model(r'bal_data_model.keras')  # Replace with the path to your model

# Load the tokenizer using pickle
with open(r'tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load the emoji scores data
with open(r'emoji_scores.pkl', 'rb') as emoji_scores_file:
    emoji_scores = pickle.load(emoji_scores_file)

# Define the function to calculate emoji scores
def calculate_emoji_scores_for_tweets(tweet_data, emoji_scores):
    emoji_positive_scores_train = []

    for tweet in tweet_data:
        positive_scores = []  # List to store positive scores for each emoji in the tweet

        emojis = re.findall(r'[^\w\s,]', tweet)  # Extract emojis from the tweet

        for emoji_char in emojis:
            if emoji_char in emoji_scores:
                positive_score = emoji_scores[emoji_char]['positive']
                positive_scores.append(positive_score)

        # Calculate the average positive score for emojis in the tweet
        avg_positive_score = sum(positive_scores) / len(positive_scores) if positive_scores else 0.0
        emoji_positive_scores_train.append(avg_positive_score)

    return emoji_positive_scores_train

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    content = [request.form['content']]
    sequence = tokenizer.texts_to_sequences(content)
    sequence = pad_sequences(sequence, maxlen=100)

    # Calculate emoji scores for the user's input
    emoji_score = calculate_emoji_scores_for_tweets(content, emoji_scores)

    prediction = model.predict([sequence, np.array([emoji_score])])

    response = {
        'positive': float(prediction[0][0]),
        'negative': float(prediction[0][1]),
        'neutral': float(prediction[0][2]),
    }

    return render_template('result.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)
