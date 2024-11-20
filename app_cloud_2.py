import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import Dict, List
import torch
from pathlib import Path

# Constants for column names
SCORE_COLUMNS = {
    'about': {'weight_key': 'about', 'embed_col': 'about_embedding', 'score_col': 'about_score'},
    'edu_exp': {'weight_key': 'education & experience', 'embed_col': 'edu_exp_embedding', 'score_col': 'edu_exp_score'},
    'social_media': {'weight_key': 'social media', 'embed_col': 'social_media_embedding', 'score_col': 'social_media_score'}
}

class RelevanceScorer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = self._load_model(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    @st.cache_resource
    def _load_model(model_name: str) -> SentenceTransformer:
        return SentenceTransformer(model_name)
    
    def encode_topic(self, topic: str) -> torch.Tensor:
        return self.model.encode([topic], convert_to_tensor=True)
    
    def calculate_relevance_scores(self, batch_df: pd.DataFrame, topic_embedding: torch.Tensor) -> pd.DataFrame:
        result_df = batch_df.copy()
        
        for key, cols in SCORE_COLUMNS.items():
            embed_col = cols['embed_col']
            score_col = cols['score_col']
            
            if embed_col in batch_df:
                result_df[score_col] = batch_df[embed_col].apply(
                    lambda emb: float(util.cos_sim(
                        np.array(eval(emb)) if isinstance(emb, str) else emb,
                        topic_embedding
                    ).squeeze()) if pd.notna(emb) else 0.0
                )
        
        return result_df

class DataLoader:
    @staticmethod
    @st.cache_data
    def load_data(file_path: str) -> pd.DataFrame:
        file_path = Path(file_path)
        parquet_path = file_path.with_suffix('.parquet')
        
        try:
            df = pd.read_parquet(parquet_path)
            st.success("✅ Data loaded successfully from Parquet file")
        except FileNotFoundError:
            try:
                st.info("Converting CSV to Parquet format...")
                df = pd.read_csv(file_path)
                df.to_parquet(parquet_path)
                st.success("✅ CSV successfully converted to Parquet")
            except FileNotFoundError:
                st.error(f"❌ No data file found at {file_path}")
                raise
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                raise
        
        return df

def create_weight_sliders() -> Dict[str, float]:
    st.sidebar.header("Adjust Weights")
    weights = {}
    
    for key, cols in SCORE_COLUMNS.items():
        weight_key = cols['weight_key']
        weights[weight_key] = st.sidebar.slider(
            f"Weight for '{weight_key.title()}'",
            0.0, 1.0, 0.33, 0.05
        )
    
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.001:
        st.sidebar.error(f"⚠️ Weights sum to {total_weight:.2f}. Please adjust to sum to 1.0")
    
    return weights

def calculate_weighted_scores(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    # Calculate weighted sum using the score columns
    df['Weighted_Score'] = sum(
        df[SCORE_COLUMNS[key]['score_col']] * weights[cols['weight_key']]
        for key, cols in SCORE_COLUMNS.items()
    )
    return df

def display_results(df: pd.DataFrame, page_size: int = 100):
    total_pages = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        page_number = st.number_input("Page", min_value=1, max_value=total_pages, step=1)
    with col2:
        st.write(f"Total Pages: {total_pages}")
    
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    
    # Prepare display columns
    display_columns = ['id', 'name', 'Weighted_Score']
    display_columns.extend(cols['score_col'] for cols in SCORE_COLUMNS.values())
    
    # Prepare formatting
    format_dict = {'Weighted_Score': '{:.3f}'}
    format_dict.update({
        cols['score_col']: '{:.3f}'
        for cols in SCORE_COLUMNS.values()
    })
    
    st.dataframe(
        df.iloc[start_idx:end_idx][display_columns].style.format(format_dict),
        use_container_width=True
    )

def process_batches(df: pd.DataFrame, scorer: RelevanceScorer, topic_embedding: torch.Tensor, 
                   batch_size: int, progress_text: str = "Processing...") -> pd.DataFrame:
    results = []
    total_batches = (len(df) + batch_size - 1) // batch_size  # Ceiling division
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for batch_idx, i in enumerate(range(0, len(df), batch_size)):
        # Update status
        status_text.text(f"{progress_text} - Batch {batch_idx + 1}/{total_batches}")
        
        # Process batch
        batch = df.iloc[i:i+batch_size]
        processed_batch = scorer.calculate_relevance_scores(batch, topic_embedding)
        results.append(processed_batch)
        
        # Update progress (ensure it never exceeds 1.0)
        progress = min(1.0, (batch_idx + 1) / total_batches)
        progress_bar.progress(progress)
    
    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return pd.concat(results, ignore_index=True)

def main():
    st.set_page_config(page_title="Relevance Scoring App", layout="wide")
    st.title("🎯 Relevance Scoring App")
    
    # Initialize components
    scorer = RelevanceScorer()
    weights = create_weight_sliders()
    
    # Load data
    try:
        df = DataLoader.load_data('merged_for_AI_test.csv')
        st.write("Preview of loaded data:")
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return
    
    # Topic input
    topic_input = st.text_input("🔍 Enter your topic", help="Enter the topic you want to score against")
    
    if st.button("Run Scoring", type="primary", disabled=not topic_input.strip()):
        if abs(sum(weights.values()) - 1.0) > 0.001:
            st.error("Please adjust weights to sum to 1.0 before proceeding")
            return
            
        try:
            # Encode topic
            topic_embedding = scorer.encode_topic(topic_input)
            
            # Process in batches with improved progress tracking
            df_processed = process_batches(
                df, 
                scorer, 
                topic_embedding, 
                batch_size=500, 
                progress_text="Calculating relevance scores"
            )
            
            # Calculate weighted scores
            df_processed = calculate_weighted_scores(df_processed, weights)
            
            # Sort and display
            df_sorted = df_processed.sort_values('Weighted_Score', ascending=False, ignore_index=True)
            st.success("✅ Scoring completed successfully!")
            display_results(df_sorted)
            
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.error(f"Error details: {type(e).__name__}")
            import traceback
            st.error(traceback.format_exc())
    else:
        st.info("👆 Enter a topic and click 'Run Scoring' to start processing")

if __name__ == "__main__":
    main()