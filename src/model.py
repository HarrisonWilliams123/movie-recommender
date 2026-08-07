import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score

#Set a seed for reproducibility
np.random.seed(1234)

df = pd.read_csv("data/netflix_titles.csv")

#Clean duration column
def clean_duration(x):
    if pd.isna(x):
        return np.nan
    if "Season" in x:
        return int(x.split()[0])
    if "min" in x:
        return int(x.replace(" min", ""))
    return np.nan

df["duration_clean"] = df["duration"].apply(clean_duration)

#Extract date features
df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
df["added_year"] = df["date_added"].dt.year
df["added_month"] = df["date_added"].dt.month

#Split genres
df["listed_in"] = df["listed_in"].fillna("Unknown")
df["genre_list"] = df["listed_in"].apply(lambda x: x.split(", "))

#Create top genres
all_genres = pd.Series([g for sub in df["genre_list"] for g in sub])
top_genres = all_genres.value_counts().head(15).index.tolist()

#One-hot encode top genres
for genre in top_genres:
    df[f"genre_{genre}"] = df["genre_list"].apply(lambda x: int(genre in x))

#Select final features
categorical_cols = ["type", "country", "rating"]
numeric_cols = ["release_year", "duration_clean", "added_year", "added_month"] + [f"genre_{g}" for g in top_genres]

#Build transformers
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

#Combine the two transformers into one
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_cols),
        ("num", numeric_transformer, numeric_cols)
    ]
)

#Builds the KMeans pipeline
kmeans_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("cluster", KMeans(n_clusters=14, random_state=42))
])

#Fit the model
kmeans_model.fit(df)

#Gets the cluster labels
df["cluster"] = kmeans_model["cluster"].labels_

#Transform the data using the pipeline's preprocessing step
X_processed = kmeans_model["preprocess"].transform(df)

#Compute silhouette score
sil_score = silhouette_score(X_processed, df["cluster"])
print("Silhouette Score:", sil_score)

#Saves the clusters to a csv file
df.to_csv("data/model_predictions2.csv", index=False)

