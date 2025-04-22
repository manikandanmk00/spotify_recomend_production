from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

df = joblib.load("models/data.pkl")
cosine_sim = joblib.load("models/cosine_sim.pkl")

class SongRequest(BaseModel):
    song_name: str
    top_n: int = 5

@app.get("/")
def home():
    return {"message": "Welcome to the Song Recommender API"}

@app.post("/recommend/")
def recommend_songs(req: SongRequest):
    song_name = req.song_name
    top_n = req.top_n

    idx = df[df['song'].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        return {"error": "Song not found in dataset"}

    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    song_indices = [i[0] for i in sim_scores]

    recommendations = df[["artist", "song"]].iloc[song_indices]
    return recommendations.to_dict(orient="records")
