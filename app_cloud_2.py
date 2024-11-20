import streamlit as st
import pandas as pd
from sentence_transformers import CrossEncoder
import numpy as np
from typing import List

# Page configuration
st.set_page_config(page_title="Relevance Scoring App", page_icon="🎯", layout="wide")

# Configuration
MODEL_NAME = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
REQUIRED_COLUMNS = ['about', 'education & experience', 'social media']
PAGE_SIZE = 10

# Initialize session state
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0

@st.cache_resource
def load_model():
    """Load cross-encoder model with error handling."""
    try:
        return CrossEncoder(MODEL_NAME, max_length=512)
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

@st.cache_data(ttl=3600)
def load_data():
    """Load data with robust error handling."""
    try:
        df = pd.read_csv('merged_for_AI_test.csv')
        
        # Ensure required columns exist and have data
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                st.error(f"Missing column: {col}")
                return pd.DataFrame()
            
            # Convert columns to string and fill NaNs
            df[col] = df[col].fillna('').astype(str)
        
        return df
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return pd.DataFrame()

def compute_cross_encoder_scores(
    texts: List[str], 
    topic: str, 
    model: CrossEncoder,
    min_text_length: int = 5
) -> np.ndarray:
    """Compute cross-encoder scores with robust preprocessing."""
    # Filter and prepare texts
    valid_texts = [
        text.strip() 
        for text in texts 
        if len(text.strip()) >= min_text_length
    ]
    
    if not valid_texts:
        return np.zeros(len(texts))
    
    # Prepare topic-text pairs
    text_topic_pairs = [[topic, text] for text in valid_texts]
    
    try:
        # Compute raw scores
        raw_scores = model.predict(text_topic_pairs)
        
        # Normalize scores
        normalized_scores = 1 / (1 + np.exp(-raw_scores))
        
        # Map scores back to original array length
        full_scores = np.zeros(len(texts))
        for i, valid_idx in enumerate(
            [j for j, text in enumerate(texts) if len(text.strip()) >= min_text_length]
        ):
            full_scores[valid_idx] = normalized_scores[i]
        
        return full_scores
    
    except Exception as e:
        st.error(f"Scoring error: {e}")
        return np.zeros(len(texts))

def main():
    st.title("Relevance Scoring App")
    
    # Load resources
    model = load_model()
    df = load_data()
    
    if model is None or df.empty:
        st.warning("Unable to load resources.")
        return
    
    # Column weight configuration
    st.sidebar.header("Column Weights")
    weights = {}
    for column in REQUIRED_COLUMNS:
        weights[column] = st.sidebar.slider(
            f"{column.capitalize()} Weight",
            min_value=0.0, max_value=1.0,
            value=1/len(REQUIRED_COLUMNS),
            step=0.01
        )
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Topic input
    topic_input = st.text_input("Enter Search Topic")
    
    if st.button("Analyze", type="primary"):
        if not topic_input.strip():
            st.warning("Please enter a topic.")
            return
        
        with st.spinner("Processing..."):
            # Reset pagination
            st.session_state.current_page = 0
            
            # Score each column
            column_scores = {}
            for column in REQUIRED_COLUMNS:
                column_scores[column] = compute_cross_encoder_scores(
                    df[column].tolist(), 
                    topic_input, 
                    model
                )
            
            # Compute weighted score
            weighted_score = sum(
                column_scores[col] * weights[col] 
                for col in REQUIRED_COLUMNS
            )
            
            # Prepare results
            results_df = df[['id', 'name', *REQUIRED_COLUMNS]].copy()
            for col in REQUIRED_COLUMNS:
                results_df[f'{col}_score'] = column_scores[col]
            results_df['weighted_score'] = weighted_score
            results_df = results_df.sort_values('weighted_score', ascending=False)
            
            st.session_state.processed_results = results_df
    
    # Display results
    if st.session_state.processed_results is not None:
        results_df = st.session_state.processed_results
        total_pages = len(results_df) // PAGE_SIZE + (1 if len(results_df) % PAGE_SIZE else 0)
        
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
        
        st.dataframe(
            results_df.iloc[start_idx:end_idx],
            use_container_width=True,
            hide_index=True
        )
        
        # Download option
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results",
            csv,
            "relevance_scores.csv",
            "text/csv"
        )

if __name__ == "__main__":
    main()