import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import Dict, List
import torch
from pathlib import Path

# Constants for column names and weights
COLUMN_CONFIG = {
    'about': {
        'weight_key': 'about',
        'weight_label': 'About',
        'embed_col': 'about_embedding',
        'score_col': 'about_score',
        'default_weight': 0.33
    },
    'edu_exp': {
        'weight_key': 'education & experience',
        'weight_label': 'Education & Experience',
        'embed_col': 'edu_exp_embedding',
        'score_col': 'edu_exp_score',
        'default_weight': 0.34
    },
    'social_media': {
        'weight_key': 'social media',
        'weight_label': 'Social Media',
        'embed_col': 'social_media_embedding',
        'score_col': 'social_media_score',
        'default_weight': 0.33
    }
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
    
    def calculate_similarity(self, embedding_str: str, topic_embedding: torch.Tensor) -> float:
        try:
            if pd.isna(embedding_str):
                return 0.0
            embedding = np.array(eval(embedding_str)) if isinstance(embedding_str, str) else embedding_str
            return float(util.cos_sim(embedding, topic_embedding).squeeze())
        except Exception as e:
            st.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def calculate_relevance_scores(self, batch_df: pd.DataFrame, topic_embedding: torch.Tensor) -> pd.DataFrame:
        result_df = batch_df.copy()
        
        for config in COLUMN_CONFIG.values():
            embed_col = config['embed_col']
            score_col = config['score_col']
            
            if embed_col in result_df.columns:
                result_df[score_col] = result_df[embed_col].apply(
                    lambda x: self.calculate_similarity(x, topic_embedding)
                )
            else:
                st.warning(f"Missing embedding column: {embed_col}")
                result_df[score_col] = 0.0
                
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
    
    for config in COLUMN_CONFIG.values():
        weights[config['weight_key']] = st.sidebar.slider(
            f"Weight for {config['weight_label']}",
            0.0, 1.0, config['default_weight'], 0.01
        )
    
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.001:
        st.sidebar.error(f"⚠️ Weights sum to {total_weight:.2f}. Please adjust to sum to 1.0")
    
    return weights

def calculate_weighted_scores(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    result_df = df.copy()
    
    # Initialize weighted score to 0
    result_df['Weighted_Score'] = 0.0
    
    # Add each component to the weighted score
    for config in COLUMN_CONFIG.values():
        score_col = config['score_col']
        weight_key = config['weight_key']
        
        if score_col in result_df.columns:
            result_df['Weighted_Score'] += result_df[score_col] * weights[weight_key]
    
    return result_df

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
    score_columns = [config['score_col'] for config in COLUMN_CONFIG.values()]
    display_columns.extend(score_columns)
    
    # Prepare formatting
    format_dict = {col: '{:.3f}' for col in ['Weighted_Score'] + score_columns}
    
    st.dataframe(
        df.iloc[start_idx:end_idx][display_columns].style.format(format_dict),
        use_container_width=True
    )

def process_batches(df: pd.DataFrame, scorer: RelevanceScorer, topic_embedding: torch.Tensor, 
                   batch_size: int, progress_text: str = "Processing...") -> pd.DataFrame:
    results = []
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for batch_idx, i in enumerate(range(0, len(df), batch_size)):
            status_text.text(f"{progress_text} - Batch {batch_idx + 1}/{total_batches}")
            
            batch = df.iloc[i:i+batch_size]
            processed_batch = scorer.calculate_relevance_scores(batch, topic_embedding)
            results.append(processed_batch)
            
            progress = min(1.0, (batch_idx + 1) / total_batches)
            progress_bar.progress(progress)
    except Exception as e:
        st.error(f"Error in batch processing: {str(e)}")
        raise
    finally:
        progress_bar.empty()
        status_text.empty()
    
    return pd.concat(results, ignore_index=True)

def main():
    st.set_page_config(page_title="Relevance Scoring App", layout="wide")
    st.title("🎯 Relevance Scoring App")
    
    try:
        scorer = RelevanceScorer()
        weights = create_weight_sliders()
        
        df = DataLoader.load_data('merged_for_AI_test.csv')
        st.write("Preview of loaded data:")
        st.dataframe(df.head(), use_container_width=True)
        
        topic_input = st.text_input("🔍 Enter your topic", help="Enter the topic you want to score against")
        
        if st.button("Run Scoring", type="primary", disabled=not topic_input.strip()):
            if abs(sum(weights.values()) - 1.0) > 0.001:
                st.error("Please adjust weights to sum to 1.0 before proceeding")
                return
                
            try:
                topic_embedding = scorer.encode_topic(topic_input)
                
                with st.spinner("Processing scores..."):
                    df_processed = process_batches(
                        df, 
                        scorer, 
                        topic_embedding, 
                        batch_size=500, 
                        progress_text="Calculating relevance scores"
                    )
                    
                    df_processed = calculate_weighted_scores(df_processed, weights)
                    df_sorted = df_processed.sort_values('Weighted_Score', ascending=False, ignore_index=True)
                    
                    st.success("✅ Scoring completed successfully!")
                    display_results(df_sorted)
                    
            except Exception as e:
                st.error("❌ Error during processing")
                st.error(f"Error details: {str(e)}")
                st.error("Stack trace:", exception=True)
        else:
            st.info("👆 Enter a topic and click 'Run Scoring' to start processing")
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Stack trace:", exception=True)

if __name__ == "__main__":
    main()