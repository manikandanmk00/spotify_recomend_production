from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the models and data
df = joblib.load("models/data.pkl")
cosine_sim = joblib.load("models/cosine_sim.pkl")

class SongRequest(BaseModel):
    song_name: str  # song name input from the user
    top_n: int = 5  # default number of recommendations to return

@app.get("/")
def home():
    return {"message": "Welcome to the Song Recommender API"}

@app.post("/recommend/")
def recommend_songs(req: SongRequest):
    song_name = req.song_name
    top_n = req.top_n

    # Search for the song in the dataset
    idx = df[df['song'].str.lower() == song_name.lower()].index

    if len(idx) == 0:
        return {"error": f"Song '{song_name}' not found in the dataset."}

    idx = idx[0]  # Get the index of the song
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Get top_n similar songs (exclude the song itself)

    # Extract the indices of the top_n most similar songs
    song_indices = [i[0] for i in sim_scores]

    # Return the recommendations as a list of dictionaries
    recommendations = df[["artist", "song"]].iloc[song_indices]
    return recommendations.to_dict(orient="records")
