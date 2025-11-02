from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---------------------------------------------------------
# Load Dataset Once (for faster performance)
# ---------------------------------------------------------
data = pd.read_csv("music_dataset.csv", encoding="utf-8")

# Select relevant columns
features = [
    'genre', 'artist_name', 'track_name',
    'acousticness', 'danceability', 'energy',
    'instrumentalness', 'liveness', 'loudness',
    'speechiness', 'tempo', 'valence'
]
data = data[features].dropna().reset_index(drop=True)
data = data.drop_duplicates(subset=['track_name', 'artist_name'])

num_features = [
    'acousticness', 'danceability', 'energy',
    'instrumentalness', 'liveness', 'loudness',
    'speechiness', 'tempo', 'valence'
]
data[num_features] = (
    data[num_features] - data[num_features].min()
) / (data[num_features].max() - data[num_features].min())

# ---------------------------------------------------------
# Helper Function: Recommend Songs
# ---------------------------------------------------------
def recommend_songs(song_name, top_n=5):
    if song_name not in data['track_name'].values:
        similar = data[data['track_name'].str.contains(song_name.split()[0], case=False, na=False)]
        return None, similar[['track_name']].head(5)
    
    song_vector = data.loc[data['track_name'] == song_name, num_features].values
    similarities = cosine_similarity(song_vector, data[num_features])[0]
    indices = similarities.argsort()[::-1][1:top_n+1]
    recommendations = data.iloc[indices][['track_name', 'artist_name', 'genre']].reset_index(drop=True)
    return recommendations, None

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    song_name = request.form['song_name']
    recommendations, suggestions = recommend_songs(song_name)

    if recommendations is not None:
        return render_template('results.html', song=song_name, results=recommendations)
    else:
        return render_template('index.html', not_found=True, suggestions=suggestions.values.tolist(), song=song_name)

# ---------------------------------------------------------
# Run App
# ---------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)

