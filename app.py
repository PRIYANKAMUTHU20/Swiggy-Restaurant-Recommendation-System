import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import load_npz


# LOAD DATA & MODELS

@st.cache_resource
def load_resources():
    cleaned_df = pd.read_csv("data/swiggy_cleaned.csv")

    # Fix cost column (₹ already converted in cleaning stage)
    cleaned_df['cost'] = pd.to_numeric(cleaned_df['cost'], errors='coerce')
    cleaned_df = cleaned_df.dropna(subset=['cost', 'rating'])

    # Load encoded data and model
    encoded_data = load_npz("encoded_data.npz")

    with open("kmeans_model.pkl", "rb") as f:
        kmeans_model = pickle.load(f)

    # Ensure alignment
    min_len = min(len(cleaned_df), encoded_data.shape[0])
    cleaned_df = cleaned_df.iloc[:min_len].reset_index(drop=True)
    encoded_data = encoded_data[:min_len]

    # Precompute cluster labels ONCE
    cluster_labels = kmeans_model.predict(encoded_data)

    return cleaned_df, encoded_data, cluster_labels


cleaned_df, encoded_data, cluster_labels = load_resources()


# APP CONFIG

st.set_page_config(
    page_title="Swiggy Restaurant Recommendation System",
    layout="wide"
)

st.title("🍽️ Swiggy Restaurant Recommendation System")


# SIDEBAR INPUTS (HARD FILTERS)

st.sidebar.header("User Preferences")

city = st.sidebar.selectbox(
    "Select City",
    sorted(cleaned_df['city'].dropna().unique())
)

cuisine = st.sidebar.text_input(
    "Preferred Cuisine (optional)"
)

max_cost = st.sidebar.slider(
    "Maximum Cost (₹)",
    min_value=100,
    max_value=5000,
    value=500,
    step=50
)

min_rating = st.sidebar.slider(
    "Minimum Rating",
    min_value=0.0,
    max_value=5.0,
    value=3.5,
    step=0.1
)

top_n = st.sidebar.slider(
    "Number of Recommendations",
    5, 20, 10
)


# RECOMMENDATION LOGIC

def recommend_restaurants(city, cuisine, max_cost, min_rating, top_n):

    # STEP 1: HARD FILTERS (MANDATORY)
    filtered_df = cleaned_df[
        (cleaned_df['city'].str.lower() == city.lower()) &
        (cleaned_df['cost'] <= max_cost) &
        (cleaned_df['rating'] >= min_rating)
    ]

    if cuisine:
        filtered_df = filtered_df[
            filtered_df['cuisine'].str.contains(cuisine, case=False, na=False)
        ]

    if filtered_df.empty:
        return pd.DataFrame()

    # STEP 2: CLUSTER RELEVANCE (SOFT LOGIC)
    filtered_indices = filtered_df.index.to_numpy()
    filtered_clusters = cluster_labels[filtered_indices]

    dominant_cluster = pd.Series(filtered_clusters).mode()[0]

    filtered_df = filtered_df.copy()
    filtered_df['cluster_match'] = (
        cluster_labels[filtered_indices] == dominant_cluster
    ).astype(int)

    # STEP 3: FINAL SORTING
    filtered_df = filtered_df.sort_values(
        by=['cluster_match', 'rating', 'rating_count', 'cost'],
        ascending=[False, False, False, True]
    )

    return filtered_df[
        ['name', 'city', 'cuisine', 'rating', 'rating_count', 'cost']
    ].head(top_n)



# OUTPUT

if st.button("Get Recommendations"):
    results = recommend_restaurants(
        city, cuisine, max_cost, min_rating, top_n
    )

    if results.empty:
        st.warning("No restaurants match your preferences.")
    else:
        st.success("Recommended Restaurants")
        st.dataframe(results.reset_index(drop=True))
