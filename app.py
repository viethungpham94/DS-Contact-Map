import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import time

# Title of the app
st.title("Relevance Scoring App with Custom Weights and Single Topic")

# Load the pre-trained model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Weight sliders
st.sidebar.header("Adjust Weights")
weights = {
    'about': st.sidebar.slider("Weight for 'About'", min_value=0.0, max_value=1.0, value=0.2, step=0.05),
    'education & experience': st.sidebar.slider("Weight for 'Education & Experience'", min_value=0.0, max_value=1.0, value=0.6, step=0.05),
    'social media': st.sidebar.slider("Weight for 'Social Media'", min_value=0.0, max_value=1.0, value=0.2, step=0.05),
}

# Ensure the weights sum to 1
if sum(weights.values()) != 1.0:
    st.sidebar.warning("Weights do not sum to 1. Normalize before proceeding.")

# Embed the CSV file
file_path = r'F:\Flourishing Group\HQ\Network Mapping\merged_for_AI_test.csv'
@st.cache_data
def load_embedded_data():
    # Replace 'data.csv' with the path to your embedded CSV file
    return pd.read_csv(file_path)

# Load the embedded data
df = load_embedded_data()

# Display the data (Read-only view)
st.write("Embedded Data:")
st.write(df)

# Topic input from the user (only one topic)
topic_input = st.text_input("Enter your topic")

# Flag to lock inputs while running the code
lock_inputs = st.button("Run Scoring")

if lock_inputs and topic_input.strip():
    with st.spinner("Processing..."):
        time.sleep(2)  # Simulate processing delay

        # Encode topic embeddings
        @st.cache_resource
        def encode_topic(topic):
            return model.encode([topic], convert_to_tensor=True)

        topic_embedding = encode_topic(topic_input)

        # Ensure required columns exist
        if {'about', 'education & experience', 'social media'}.issubset(df.columns):
            # Define scoring function
            def calculate_relevance_scores(text):
                if not isinstance(text, str) or text.strip() == '':
                    return 0.0  # Return a float value
                text_embedding = model.encode(text, convert_to_tensor=True)
                similarity_score = util.cos_sim(text_embedding, topic_embedding).squeeze()
                return float(similarity_score)  # Ensure it returns a float

            # Calculate relevance scores for each column
            df['Relevance Scores - About'] = df['about'].apply(calculate_relevance_scores)
            df['Relevance Scores - Education & Experience'] = df['education & experience'].apply(calculate_relevance_scores)
            df['Relevance Scores - Social Media'] = df['social media'].apply(calculate_relevance_scores)

            # Calculate the weighted score for each row
            df['Weighted_Score'] = (
                df['Relevance Scores - About'] * weights['about'] +
                df['Relevance Scores - Education & Experience'] * weights['education & experience'] +
                df['Relevance Scores - Social Media'] * weights['social media']
            )

            # Sort the DataFrame by weighted scores in descending order
            df_sorted = df.sort_values(by='Weighted_Score', ascending=False).reset_index(drop=True)

            # Create a new DataFrame with only the relevant columns
            id_and_weighted_scores_df = df_sorted[[
                'id', 'name', 'about', 'education & experience', 'social media',
                'Relevance Scores - About', 
                'Relevance Scores - Education & Experience', 
                'Relevance Scores - Social Media', 
                'Weighted_Score'
            ]]

            # Display the results

            st.write("ID, Scores, and Weighted Scores (Read-Only):")
            st.write(id_and_weighted_scores_df)

        else:
            st.warning("The dataset does not contain the required columns: 'about', 'education & experience', 'social media'.")
else:
    st.warning("Please enter a topic and click the 'Run Scoring' button to start processing.")