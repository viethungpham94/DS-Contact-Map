import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Title of the app
st.title("Relevance Scoring App with Custom Weights and Single Topic")

# Load the pre-trained model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Sidebar for weight sliders
st.sidebar.header("Adjust Weights")
weights = {
    'about': st.sidebar.slider("Weight for 'About'", min_value=0.0, max_value=1.0, value=0.2, step=0.05),
    'education & experience': st.sidebar.slider("Weight for 'Education & Experience'", min_value=0.0, max_value=1.0, value=0.6, step=0.05),
    'social media': st.sidebar.slider("Weight for 'Social Media'", min_value=0.0, max_value=1.0, value=0.2, step=0.05),
}

# Ensure weights sum to 1
if sum(weights.values()) != 1.0:
    st.sidebar.warning("Weights do not sum to 1. Normalize before proceeding.")

@st.cache_data
def load_data(file_path):
    # First, check if the Parquet file already exists. If not, convert the CSV to Parquet.
    parquet_file_path = file_path.replace('.csv', '.parquet')
    try:
        df = pd.read_parquet(parquet_file_path)
        st.write("Loaded data from Parquet file.")
    except FileNotFoundError:
        st.write("Parquet file not found, converting CSV to Parquet...")
        df = pd.read_csv(file_path)
        df.to_parquet(parquet_file_path)  # Save the CSV as Parquet for future use
        st.write("CSV file converted to Parquet.")
    return df

# Provide the CSV file path
file_path = 'merged_for_AI_test.csv'

# Load the data (either from CSV or Parquet)
df = load_data(file_path)

# Display the data (Read-only view)
st.write("Embedded Data:")
st.write(df.head())  # Show only a preview for large datasets

# Topic input from the user
topic_input = st.text_input("Enter your topic")

# Batch processing function for scoring
def calculate_relevance_scores(batch_df, topic_embedding):
    for col in ['about_embedding', 'edu_exp_embedding', 'social_media_embedding']:
        if col in batch_df:
            batch_df[f'Relevance Scores - {col.split("_")[0]}'] = batch_df[col].apply(
                lambda emb: float(util.cos_sim(np.array(eval(emb)), topic_embedding).squeeze()) if pd.notna(emb) else 0.0
            )
    return batch_df

# Processing and scoring
lock_inputs = st.button("Run Scoring")

if lock_inputs and topic_input.strip():
    with st.spinner("Processing..."):
        # Encode topic embedding
        topic_embedding = model.encode([topic_input], convert_to_tensor=True)

        # Process the DataFrame in batches
        batch_size = 500
        results = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size].copy()
            batch = calculate_relevance_scores(batch, topic_embedding)
            results.append(batch)

        # Combine processed batches
        df_processed = pd.concat(results).reset_index(drop=True)

        # Calculate weighted scores
        df_processed['Weighted_Score'] = (
            df_processed['Relevance Scores - about'] * weights['about'] +
            df_processed['Relevance Scores - education'] * weights['education & experience'] +
            df_processed['Relevance Scores - social_media'] * weights['social media']
        )

        # Sort by weighted scores
        df_sorted = df_processed.sort_values(by='Weighted_Score', ascending=False).reset_index(drop=True)

        # Pagination for results display
        page_size = 100
        total_pages = len(df_sorted) // page_size + (len(df_sorted) % page_size > 0)
        page_number = st.number_input("Page Number",
