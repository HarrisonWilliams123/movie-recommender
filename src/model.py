import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

#Set a seed for reproducibility
np.random.seed(1234)

df = pd.read_csv("data/netflix_titles.csv")

#Created arrays to differ categorical and numerical features
categorical_cols = ['type', 'title', 'director', 'cast', 'country', 'date_added', 'rating', 'duration', 'listed_in', 'description']
numerical_cols = ['release_year']

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
        ("num", numeric_transformer, numerical_cols)
    ]
)

#Builds the KMeans pipeline
kmeans_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("cluster", KMeans(n_clusters=4, random_state=42))
])

#Fit the model
kmeans_model.fit(df)

#Gets the cluster labels
df["cluster"] = kmeans_model["cluster"].labels_

#Saves the clusters to a csv file
df.to_csv("data/model_predictions.csv", index=False)

