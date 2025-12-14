# Swiggy-Restaurant-Recommendation-System
Restaurant recommendation system using clustering (K-Means) 

Approach

1. Data Understanding & Cleaning 
   - Remove duplicates and handle missing values, datatype conversion
   - Cleaned the cost column by removing non-numeric characters and converting it to a numeric format.
   - Standardized rating_count by converting text values like “Too Few Ratings” and “1K+ ratings” into numeric values.
   - Cleaned the rating column by handling missing and invalid values and filling them using grouped averages (city, cuisine, cost) and the dataset median.
   - Removed invalid rows where the cuisine column contained time or time-range values.
   - Save cleaned data

2. Preprocessing
   - Converted multi-value cuisine entries into a single string.
   - One-Hot Encode categorical features (city, cuisine)  using ColumnTransformer.
   - Save the encoder as encoded_data.npz and fitted encoder as encoder.pkl

3. Recommendation Engine
   - Train a K-Means clustering model on the encoded data
   - Predict cluster for user preferences
   - Recommend similar restaurants from the same cluster

4. Streamlit Application
   - Accept user preferences (city, cuisine, cost, rating)
   - Show ranked recommendations
