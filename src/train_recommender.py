import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess_text

mlflow.set_tracking_uri("http://127.0.0.1:5000") 

def train_model(csv_path, save_dir="models/"):
    df = pd.read_csv(csv_path).drop(columns=['link'], errors='ignore').dropna()
    df = df.sample(10000).reset_index(drop=True)

    df["cleaned_text"] = df["text"].apply(preprocess_text)

    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["cleaned_text"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    joblib.dump(tfidf, f"{save_dir}tfidf.pkl")
    joblib.dump(df, f"{save_dir}data.pkl")
    joblib.dump(cosine_sim, f"{save_dir}cosine_sim.pkl")

    

    with mlflow.start_run(run_name="Train Recommender"):
        
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("max_features", 5000)
        mlflow.log_metric("num_samples", df.shape[0])
        mlflow.log_metric("num_features", tfidf_matrix.shape[1])

        mlflow.sklearn.log_model(tfidf, "tfidf_vectorizer")
        mlflow.log_artifact(f"{save_dir}data.pkl")
        mlflow.log_artifact(f"{save_dir}cosine_sim.pkl")

    print("✅ Model training and artifact saving complete.")

if __name__ == "__main__":
    train_model("data/spotify_millsongdata.csv")
