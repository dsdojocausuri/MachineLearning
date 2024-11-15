import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import textdistance

# Streamlit App Title
st.title("Data Clustering & Mastering App")
st.write("Cluster and merge data based on similarity metrics.")

# Initialize session state for DataFrame and cluster ID
if 'df' not in st.session_state:
    st.session_state.df = None
if 'current_cluster_id' not in st.session_state:
    st.session_state.current_cluster_id = 0

def clean_text(text):
    """Clean and standardize text inputs."""
    return text.strip().upper() if pd.notna(text) else ""

def calculate_match_score(value1, value2, algorithm):
    """Calculate match score using different algorithms."""
    if algorithm == "fuzz":
        return fuzz.ratio(value1, value2) / 100
    elif algorithm == "token_sort":
        return fuzz.token_sort_ratio(value1, value2) / 100
    elif algorithm == "jaro_winkler":
        return textdistance.jaro_winkler(value1, value2)
    elif algorithm == "levenshtein":
        return textdistance.levenshtein.normalized_similarity(value1, value2)
    return 0.0

# Step 1: Data Input
st.header("Step 1: Select Input Method")
input_method = st.radio("Choose Input Method", ["Upload CSV", "Enter Data Manually"])

# Handle CSV Upload
if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df["Producername"] = df["Producername"].apply(clean_text)
        df["Subproducername"] = df["Subproducername"].apply(clean_text)
        df["subproduceraddress"] = df["subproduceraddress"].apply(clean_text)
        st.session_state.df = df

# Handle Manual Input
if input_method == "Enter Data Manually":
    producer_input = st.text_area("Producername (comma-separated)", "")
    subproducer_input = st.text_area("Subproducername (comma-separated)", "")
    address_input = st.text_area("Address (comma-separated)", "")

    if st.button("Submit"):
        producers = [clean_text(x) for x in producer_input.split(",") if x.strip()]
        subproducers = [clean_text(x) for x in subproducer_input.split(",") if x.strip()]
        addresses = [clean_text(x) for x in address_input.split(",") if x.strip()]

        if len(producers) == len(subproducers) == len(addresses):
            df = pd.DataFrame({
                "Producername": producers,
                "Subproducername": subproducers,
                "subproduceraddress": addresses
            })
            st.session_state.df = df
        else:
            st.error("Ensure all fields have the same number of entries.")

df = st.session_state.df

if df is not None:
    st.sidebar.header("Adjust Weights & Thresholds")
    config = [
        {'column': "Producername", "algorithm": "fuzz", "threshold": st.sidebar.slider("Producername Threshold", 0.0, 1.0, 0.9), "weight": st.sidebar.slider("Producername Weight", 0.0, 1.0, 0.6)},
        {'column': "Subproducername", "algorithm": "token_sort", "threshold": st.sidebar.slider("Subproducername Threshold", 0.0, 1.0, 0.85), "weight": st.sidebar.slider("Subproducername Weight", 0.0, 1.0, 0.8)},
        {'column': "subproduceraddress", "algorithm": "jaro_winkler", "threshold": st.sidebar.slider("Address Threshold", 0.0, 1.0, 0.85), "weight": st.sidebar.slider("Address Weight", 0.0, 1.0, 0.75)}
    ]
    final_threshold = st.sidebar.slider("Final Threshold", 0.0, 1.0, 0.85)

    df['BlockKey'] = df['Producername'].str[:5] + "_" + df['Subproducername'].str[:5]

    # Function to assign clusters with continuous IDs
    def assign_clusters_block(df_block, config, final_threshold):
        df_block = df_block.reset_index(drop=True)
        df_block['ClusterId'] = np.nan
        df_block['WeightedScore'] = np.nan

        for conf in config:
            match_score_col = f"{conf['column']}MatchScore"
            df_block[match_score_col] = np.nan

        for i in range(len(df_block)):
            if pd.isna(df_block.at[i, 'ClusterId']):
                df_block.at[i, 'ClusterId'] = st.session_state.current_cluster_id
                for j in range(i + 1, len(df_block)):
                    total_score = 0
                    total_weight = 0

                    for conf in config:
                        column = conf['column']
                        score = calculate_match_score(df_block.at[i, column], df_block.at[j, column], conf['algorithm'])
                        df_block.at[j, f"{column}MatchScore"] = score

                        if score >= conf['threshold']:
                            total_score += score * conf['weight']
                        total_weight += conf['weight']

                    if total_weight > 0:
                        final_score = total_score / total_weight
                        df_block.at[j, 'WeightedScore'] = final_score

                        if final_score >= final_threshold:
                            df_block.at[j, 'ClusterId'] = st.session_state.current_cluster_id
                st.session_state.current_cluster_id += 1

        return df_block

    blocks = df['BlockKey'].unique()
    processed_blocks = [assign_clusters_block(df[df['BlockKey'] == block], config, final_threshold) for block in blocks]
    clustered_df = pd.concat(processed_blocks, ignore_index=True)

    st.write("Clustered Data with Scores:")
    st.dataframe(clustered_df)

    def merge_clusters_concatenated(df):
        merged_data = []
        for cluster_id in df['ClusterId'].unique():
            cluster_df = df[df['ClusterId'] == cluster_id]
            merged_row = {
                'Producername': " | ".join(cluster_df['Producername'].unique()),
                'Subproducername': " | ".join(cluster_df['Subproducername'].unique()),
                'subproduceraddress': " | ".join(cluster_df['subproduceraddress'].unique())
            }
            merged_data.append(merged_row)
        return pd.DataFrame(merged_data)

    def merge_clusters_golden(df):
        merged_data = []
        for cluster_id in df['ClusterId'].unique():
            cluster_df = df[df['ClusterId'] == cluster_id]
            merged_row = {
                'Producername': cluster_df.loc[cluster_df['Producername'].apply(len).idxmax(), 'Producername'],
                'Subproducername': cluster_df.loc[cluster_df['Subproducername'].apply(len).idxmax(), 'Subproducername'],
                'subproduceraddress': cluster_df.loc[cluster_df['subproduceraddress'].apply(len).idxmax(), 'subproduceraddress']
            }
            merged_data.append(merged_row)
        return pd.DataFrame(merged_data)

    merged_df_concat = merge_clusters_concatenated(clustered_df)
    st.write("Matched Merged Concatenated Dataset:")
    st.dataframe(merged_df_concat)

    merged_df_golden = merge_clusters_golden(clustered_df)
    st.write("Merged Golden Record Dataset:")
    st.dataframe(merged_df_golden)
