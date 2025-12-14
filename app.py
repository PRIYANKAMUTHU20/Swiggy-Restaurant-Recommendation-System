import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import load_npz



# LOAD DATA, ENCODER & MODEL


@st.cache_resource
def load_resources():
    cleaned_df = pd.read_csv("data/swiggy_cleaned.csv")

    cleaned_df["cost"] = pd.to_numeric(cleaned_df["cost"], errors="coerce")
    cleaned_df["rating"] = pd.to_numeric(cleaned_df["rating"], errors="coerce")
    cleaned_df["rating_count"] = pd.to_numeric(
        cleaned_df["rating_count"], errors="coerce"
    )

    cleaned_df = cleaned_df.dropna(
        subset=["cost", "rating", "rating_count"]
    )

    # Load encoded data
    encoded_data = load_npz("encoded_data.npz")

    # Load models
    with open("kmeans_model.pkl", "rb") as f:
        kmeans_model = pickle.load(f)

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    # Align indices
    min_len = min(len(cleaned_df), encoded_data.shape[0])
    cleaned_df = cleaned_df.iloc[:min_len].reset_index(drop=True)
    encoded_data = encoded_data[:min_len]

    # Precompute cluster labels
    cluster_labels = kmeans_model.predict(encoded_data)

    return cleaned_df, encoded_data, kmeans_model, cluster_labels, encoder


cleaned_df, encoded_data, kmeans_model, cluster_labels, encoder = load_resources()


# STREAMLIT CONFIG


st.set_page_config(
    page_title="Swiggy Restaurant Recommendation System",
    layout="wide"
)

st.title("🍽️ Swiggy Restaurant Recommendation System")




# SIDEBAR INPUTS


st.sidebar.header("User Preferences")

city = st.sidebar.selectbox(
    "Select City",
    sorted(cleaned_df["city"].dropna().unique())
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


# USER INPUT ENCODING (FIXED)


def encode_user_input(city, cuisine, max_cost, min_rating, encoder):
    """
    Create a full feature vector matching training schema
    """

    user_df = pd.DataFrame([{
        "city": city,
        "cuisine": cuisine if cuisine else "Unknown",
        "rating": min_rating,               # reasonable default
        "rating_count": cleaned_df["rating_count"].median(),
        "cost": max_cost
    }])

    return encoder.transform(user_df)



# RECOMMENDATION LOGIC (TRUE CLUSTER-BASED)


def recommend_restaurants(city, cuisine, max_cost, min_rating, top_n):

    # STEP 1: HARD FILTER (NON-NEGOTIABLE)
    filtered_df = cleaned_df[
        (cleaned_df["city"].str.lower() == city.lower()) &
        (cleaned_df["cost"] <= max_cost) &
        (cleaned_df["rating"] >= min_rating)
    ]

    if cuisine:
        filtered_df = filtered_df[
            filtered_df["cuisine"].str.contains(
                cuisine, case=False, na=False
            )
        ]

    if filtered_df.empty:
        return pd.DataFrame()

    # STEP 2: Encode USER
    user_vector = encode_user_input(
        city, cuisine, max_cost, min_rating, encoder
    )

    # STEP 3: Compute similarity via cluster distance
    filtered_indices = filtered_df.index.to_numpy()
    filtered_vectors = encoded_data[filtered_indices]

    distances = np.linalg.norm(
        filtered_vectors.toarray() - user_vector.toarray(),
        axis=1
    )

    filtered_df = filtered_df.copy()
    filtered_df["similarity_score"] = distances

    # STEP 4: Rank by similarity + business value
    filtered_df = filtered_df.sort_values(
        by=["similarity_score", "rating", "rating_count"],
        ascending=[True, False, False]
    )

    return filtered_df[
        ["name", "city", "cuisine", "rating", "rating_count", "cost"]
    ].head(top_n)


# OUTPUT


if st.button("Get Recommendations"):
    results = recommend_restaurants(
        city, cuisine, max_cost, min_rating, top_n
    )

    if results.empty:
        st.warning("No restaurants found for your preferences.")
    else:
        st.success("🍽️ Recommended Restaurants")
        st.dataframe(results.reset_index(drop=True))
