import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import Dict, List
import torch

# Page config for better UI
st.set_page_config(
    page_title="Relevance Scoring App",
    page_icon="🎯",
    layout="wide"
)

# Constants
MODEL_NAME = 'all-MiniLM-L6-v2'
REQUIRED_COLUMNS = {'url', 'about', 'education & experience', 'social media'}

# Initialize session state for caching results
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = None

# Cache the model loading
@st.cache_resource
def load_model() -> SentenceTransformer:
    """Load and cache the sentence transformer model."""
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Cache data loading
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and cache the dataset."""
    try:
        return pd.read_csv('merged_for_AI_test.csv')
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'merged_for_AI_test.csv' is in the correct location.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Cache text encoding
@st.cache_data
def encode_text(_model: SentenceTransformer, text: str) -> np.ndarray:
    """Encode text using the model."""
    if not isinstance(text, str) or not text.strip():
        return np.zeros((1, 384))  # Return zero vector for empty/invalid text
    return _model.encode(text, convert_to_tensor=True).cpu().numpy()

def calculate_relevance_scores(
    texts: List[str],
    topic_embedding: np.ndarray,
    model: SentenceTransformer
) -> np.ndarray:
    """Calculate relevance scores for a list of texts."""
    scores = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            scores.append(0.0)
            continue
        text_embedding = encode_text(model, text)
        similarity = util.cos_sim(torch.tensor(text_embedding), torch.tensor(topic_embedding))[0][0]
        scores.append(float(similarity))
    return np.array(scores)

def main():
    st.title("Relevance Scoring App with Custom Weights")
    
    # Load model and data
    model = load_model()
    df = load_data()
    
    if model is None or df.empty:
        st.stop()
    
    # Verify required columns
    if not REQUIRED_COLUMNS.issubset(df.columns):
        st.error(f"Missing required columns. Expected: {REQUIRED_COLUMNS}")
        st.stop()
    
    # Sidebar weights with normalization
    st.sidebar.header("Adjust Weights")
    weights = {}
    total_weight = 0
    
    for column in REQUIRED_COLUMNS:
        weight = st.sidebar.slider(
            f"Weight for '{column}'",
            min_value=0.0,
            max_value=1.0,
            value=0.33,
            step=0.01
        )
        weights[column] = weight
        total_weight += weight
    
    # Normalize weights
    if total_weight > 0:
        weights = {k: v/total_weight for k, v in weights.items()}
    
    # Topic input
    topic_input = st.text_input("Enter your topic")
    
    # Process button
    if st.button("Run Scoring", type="primary"):
        if not topic_input.strip():
            st.warning("Please enter a topic before processing.")
            st.stop()
            
        with st.spinner("Processing scores..."):
            # Encode topic
            topic_embedding = encode_text(model, topic_input)
            
            # Calculate scores for each column
            scores = {}
            for column in REQUIRED_COLUMNS:
                scores[f'Relevance Scores - {column}'] = calculate_relevance_scores(
                    df[column].tolist(),
                    topic_embedding,
                    model
                )
            
            # Calculate weighted scores
            weighted_score = sum(
                scores[f'Relevance Scores - {col}'] * weights[col]
                for col in REQUIRED_COLUMNS
            )
            
            # Create results DataFrame
            results_df = df[['id', 'name', *REQUIRED_COLUMNS]].copy()
            for col, score in scores.items():
                results_df[col] = score
            results_df['Weighted_Score'] = weighted_score
            
            # Sort and display results
            results_df = results_df.sort_values(
                by='Weighted_Score',
                ascending=False
            ).reset_index(drop=True)
            
            # Cache results in session state
            st.session_state.processed_results = results_df
            
            # Display results with pagination
            page_size = 10
            total_pages = len(results_df) // page_size + (1 if len(results_df) % page_size else 0)
            page = st.selectbox('Select page:', range(1, total_pages + 1)) - 1
            
            start_idx = page * page_size
            end_idx = min(start_idx + page_size, len(results_df))
            
            st.write(f"Showing results {start_idx + 1} to {end_idx} of {len(results_df)}")
            st.dataframe(
                results_df.iloc[start_idx:end_idx],
                use_container_width=True,
                hide_index=True
            )
            
            # Add download button
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Results",
                csv,
                "relevance_scores.csv",
                "text/csv",
                key='download-csv'
            )

if __name__ == "__main__":
    main()