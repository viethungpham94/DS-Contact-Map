import streamlit as st
import pandas as pd
from sentence_transformers import CrossEncoder
import numpy as np
from typing import Dict, List

# Page config for better UI
st.set_page_config(
    page_title="Cross-Encoder Relevance Scoring App",
    page_icon="🎯",
    layout="wide"
)

# Constants
MODEL_NAME = 'cross-encoder/nli-deberta-v3-base'
REQUIRED_COLUMNS = {'about', 'education & experience', 'social media'}
PAGE_SIZE = 10

# Initialize session state
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0

# Cache the model loading
@st.cache_resource
def load_model() -> CrossEncoder:
    """Load and cache the cross-encoder model."""
    try:
        return CrossEncoder(MODEL_NAME)
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

def calculate_cross_encoder_scores(
    texts: List[str],
    topic: str,
    model: CrossEncoder
) -> np.ndarray:
    """Calculate relevance scores using cross-encoder."""
    # Prepare pairs of topic and texts
    text_topic_pairs = [[topic, text] for text in texts]
    
    # Predict scores
    try:
        scores = model.predict(text_topic_pairs)
        return scores
    except Exception as e:
        st.error(f"Error in cross-encoder scoring: {str(e)}")
        return np.zeros(len(texts))

def display_paginated_results(results_df: pd.DataFrame):
    """Handle pagination and display of results."""
    total_pages = len(results_df) // PAGE_SIZE + (1 if len(results_df) % PAGE_SIZE else 0)
    
    # Create columns for pagination controls
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("Previous", disabled=st.session_state.current_page <= 0):
            st.session_state.current_page -= 1
            st.rerun()
            
    with col2:
        st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")
        
    with col3:
        if st.button("Next", disabled=st.session_state.current_page >= total_pages - 1):
            st.session_state.current_page += 1
            st.rerun()
    
    start_idx = st.session_state.current_page * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, len(results_df))
    
    st.write(f"Showing results {start_idx + 1} to {end_idx} of {len(results_df)}")
    st.dataframe(
        results_df.iloc[start_idx:end_idx],
        use_container_width=True,
        hide_index=True
    )

def main():
    st.title("Cross-Encoder Relevance Scoring App")
    
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
            # Reset page to 0 when running new search
            st.session_state.current_page = 0
            
            # Calculate scores for each column
            scores = {}
            for column in REQUIRED_COLUMNS:
                # Filter out non-string or empty values
                valid_texts = df[column].fillna('').astype(str)
                
                # Calculate cross-encoder scores
                scores[f'Relevance Scores - {column}'] = calculate_cross_encoder_scores(
                    valid_texts.tolist(),
                    topic_input,
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
            
            # Sort results
            results_df = results_df.sort_values(
                by='Weighted_Score',
                ascending=False
            ).reset_index(drop=True)
            
            # Store in session state
            st.session_state.processed_results = results_df
    
    # Display results if available
    if st.session_state.processed_results is not None:
        display_paginated_results(st.session_state.processed_results)
        
        # Add download button
        csv = st.session_state.processed_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results",
            csv,
            "relevance_scores.csv",
            "text/csv",
            key='download-csv'
        )

if __name__ == "__main__":
    main()