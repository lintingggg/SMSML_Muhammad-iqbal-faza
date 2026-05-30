import pandas as pd
from sklearn.neighbors import NearestNeighbors
import mlflow
import dagshub

# Nyambungin ke dagshub
username_dagshub = 'lintinggg'
nama_repository = 'SMSML_Muhammad-iqbal-faza'

dagshub.init(repo_owner=username_dagshub, repo_name=nama_repository, mlflow=True)

mlflow.set_experiment("Movie_Recommendation_Baseline")

# Main function / Modelling
def main():
    print("=== Mulai Membaca Dataset ===")

    df = pd.read_csv('movie_feelings_dataset_preprocessing.csv')

    kolom_bukan_fitur = ['imdb_id','title_year']

    features = df.drop(columns=kolom_bukan_fitur, errors='ignore')
    features = features.select_dtypes(include=['number'])

    # Autolog MLFlow
    mlflow.autolog()

    with mlflow.start_run(run_name="Baseline_Model"):
        print("Melatih model NearestNeighbors (Baseline)....")

        model = NearestNeighbors(n_neighbors=5, metric='cosine')

        model.fit(features)

        print("Model baseline sukses dilatih dan otomatis dicatat MLflow")

if __name__ == "__main__":
    main()