import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import mlflow
import dagshub

# Nyambungin ke dagshub
username_dagshub = 'lintinggg'
nama_repository = 'SMSML_Muhammad-iqbal-faza'

dagshub.init(repo_owner=username_dagshub, repo_name=nama_repository, mlflow=True)

mlflow.set_experiment("Movie_Recommendation_Tuning")

def main():
    print("=== Membaca dataset bersih ===")
    df = pd.read_csv("movie_feelings_dataset_preprocessing.csv")
    
    kolom_bukan_fitur = ['imdb_id', 'title_year']
    features = df.drop(columns=kolom_bukan_fitur, errors='ignore')
    features = features.select_dtypes(include=['number'])

    # Skenario 3: Kombinasi Nilai K dan Metrik Jarak
    k_values = [3, 5, 7]
    metrics = ['cosine', 'euclidean']
    
    tuning_results = []
    plot_data = {'cosine': [], 'euclidean': []}

    print("\n=== Memulai Hyperparameter Tuning Skenario 3 ===")
    
    for metric in metrics:
        for k in k_values:
            run_name = f"Tuning_K{k}_{metric}"
            with mlflow.start_run(run_name=run_name):
                print(f"Melatih KNN (K={k}, metric={metric})...")

                mlflow.log_param("n_neighbors", k)
                mlflow.log_param("metric", metric)

                model = NearestNeighbors(n_neighbors=k, metric=metric)
                model.fit(features)

                distances, _ = model.kneighbors(features)
                mean_dist = distances.mean()
                
                plot_data[metric].append(mean_dist)

                mlflow.log_metric("mean_distance", mean_dist)

                mlflow.sklearn.log_model(model, "model")

                tuning_results.append({
                    "n_neighbors": k,
                    "metric": metric,
                    "mean_distance": mean_dist
                })

    print("\n=== Membuat dan Mengirim 2 Artefak Tambahan ===")
    with mlflow.start_run(run_name="Advanced_Artifacts"):
        
        metadata_filename = "model_metadata.txt"
        with open(metadata_filename, "w") as f:
            f.write("=== Informasi Training Model Rekomendasi ===\n")
            f.write(f"Total film yang dilatih: {len(features)} baris\n")
            f.write(f"Total fitur emosi (vektor): {features.shape[1]} kolom\n")
            f.write("Algoritma yang digunakan: Nearest Neighbors (KNN)\n")
            f.write("Skenario Tuning: K=[3,5,7] dengan metrik Cosine & Euclidean.\n")
        
        mlflow.log_artifact(metadata_filename)
        print(f"Berhasil mengunggah Artefak 1: {metadata_filename}")

        sample_index = 0 
        distances_sample, indices_sample = model.kneighbors(features.iloc[[sample_index]])
        
        rekomendasi_df = pd.DataFrame({
            "Urutan_Terdekat": range(1, len(indices_sample[0]) + 1),
            "Index_Film_Rekomendasi": indices_sample[0],
            "Skor_Jarak": distances_sample[0]
        })
        
        sample_filename = "sample_recommendation.csv"
        rekomendasi_df.to_csv(sample_filename, index=False)
        
        mlflow.log_artifact(sample_filename)
        print(f"Berhasil mengunggah Artefak 2: {sample_filename}")
        print("\n=== EKSPERIMEN KRITERIA 2 SELESAI===")


if __name__ == "__main__":
    main()