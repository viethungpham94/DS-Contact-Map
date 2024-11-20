import streamlit as st
import pandas as pd
from sentence_transformers import CrossEncoder
import numpy as np
from typing import List

# Page configuration
st.set_page_config(
    page_title="Relevance Scoring App", 
    page_icon="🎯", 
    layout="wide"
)

# Configuration
MODEL_NAME = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'  # Smaller, more efficient model
REQUIRED_COLUMNS = {'about', 'education & experience', 'social media'}
PAGE_SIZE = 10

# Initialize session state
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0

@st.cache_resource
def load_model():
    """Lightweight model loading with error handling."""
    try:
        return CrossEncoder(MODEL_NAME, max_length=512)
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """Data loading with robust error handling."""
    try:
        return pd.read_csv('merged_for_AI_test.csv')
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return pd.DataFrame()

def batch_cross_encoder_scores(
    texts: List[str], 
    topic: str, 
    model: CrossEncoder, 
    batch_size: int = 50
) -> np.ndarray:
    """Batch processing to reduce memory load."""
    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        text_topic_pairs = [[topic, text] for text in batch_texts]
        try:
            batch_scores = model.predict(text_topic_pairs)
            normalized_scores = 1 / (1 + np.exp(-batch_scores))
            all_scores.extend(normalized_scores)
        except Exception as e:
            st.error(f"Scoring error in batch {i}: {e}")
            all_scores.extend([0] * len(batch_texts))
    return np.array(all_scores)

def display_results(results_df: pd.DataFrame):
    """Streamlined results display with pagination."""
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

def main():
    st.title("Relevance Scoring App")
    
    # Load resources
    model = load_model()
    df = load_data()
    
    if model is None or df.empty:
        st.warning("Unable to load resources.")
        return
    
    # Column validation
    if not REQUIRED_COLUMNS.issubset(df.columns):
        st.error(f"Missing columns: {REQUIRED_COLUMNS - set(df.columns)}")
        return
    
    # Dynamic weight configuration
    st.sidebar.header("Column Weights")
    weights = {}
    total_weight = 0
    
    for column in REQUIRED_COLUMNS:
        weight = st.sidebar.slider(
            f"{column.capitalize()} Weight",
            min_value=0.0, max_value=1.0,
            value=1/len(REQUIRED_COLUMNS),
            step=0.01
        )
        weights[column] = weight
        total_weight += weight
    
    # Normalize weights
    weights = {k: v/total_weight for k, v in weights.items()} if total_weight > 0 else {}
    
    # Topic input and processing
    topic_input = st.text_input("Enter Search Topic")
    
    if st.button("Analyze", type="primary"):
        if not topic_input.strip():
            st.warning("Please enter a topic.")
            return
        
        with st.spinner("Processing..."):
            st.session_state.current_page = 0
            
            # Score calculation
            scores = {}
            for column in REQUIRED_COLUMNS:
                valid_texts = df[column].fillna('').astype(str)
                scores[column] = batch_cross_encoder_scores(
                    valid_texts.tolist(), 
                    topic_input, 
                    model
                )
            
            # Weighted scoring
            weighted_score = sum(
                scores[col] * weights[col] 
                for col in REQUIRED_COLUMNS
            )
            
            # Results preparation
            results_df = df[['id', 'name', *REQUIRED_COLUMNS]].copy()
            for col in REQUIRED_COLUMNS:
                results_df[f'{col}_score'] = scores[col]
            results_df['weighted_score'] = weighted_score
            results_df = results_df.sort_values('weighted_score', ascending=False)
            
            st.session_state.processed_results = results_df
    
    # Results display
    if st.session_state.processed_results is not None:
        display_results(st.session_state.processed_results)
        
        # Download option
        csv = st.session_state.processed_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results",
            csv,
            "relevance_scores.csv",
            "text/csv"
        )

if __name__ == "__main__":
    main()