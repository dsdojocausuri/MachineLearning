import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import textdistance

# Streamlit App Title
# This is just an example on how to use streamlit
st.title("Data Clustering & Mastering App")
st.write("This app allows you to cluster and merge data based on similarity metrics. Adjust the thresholds, weights, and algorithms below to control the clustering behavior.")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Convert columns to uppercase for case insensitivity
    df["Producername"] = df["Producername"].str.upper()
    df["Subproducername"] = df["Subproducername"].str.upper()
    df["subproduceraddress"] = df["subproduceraddress"].str.upper()

    # Initialize ClusterId column with NaN
    df['ClusterId'] = np.nan

    # Sidebar for user inputs
    st.sidebar.header("Adjust Weights & Thresholds")
    
    # Weights and thresholds sliders
    producer_weight = st.sidebar.slider("Weight for Producername", 0.0, 1.0, 0.6)
    producer_threshold = st.sidebar.slider("Threshold for Producername", 0.0, 1.0, 0.9)
    
    subproducer_weight = st.sidebar.slider("Weight for Subproducername", 0.0, 1.0, 0.8)
    subproducer_threshold = st.sidebar.slider("Threshold for Subproducername", 0.0, 1.0, 0.85)
    
    address_weight = st.sidebar.slider("Weight for Address", 0.0, 1.0, 0.75)
    address_threshold = st.sidebar.slider("Threshold for Address", 0.0, 1.0, 0.85)
    
    final_threshold = st.sidebar.slider("Final Threshold", 0.0, 1.0, 0.85)
    
    # Algorithm options
    algorithms = {
        "Fuzzy Ratio": "fuzz",
        "Token Sort Ratio": "token_sort",
        "Jaro-Winkler": "jaro_winkler",
        "Levenshtein": "levenshtein"
    }
    
    # Select algorithms for each column
    st.sidebar.header("Select Algorithms")
    producer_algorithm = st.sidebar.selectbox("Algorithm for Producername", list(algorithms.keys()))
    subproducer_algorithm = st.sidebar.selectbox("Algorithm for Subproducername", list(algorithms.keys()))
    address_algorithm = st.sidebar.selectbox("Algorithm for Address", list(algorithms.keys()))

    # Configuration: List of column names and selected algorithms
    config = [
        {'column': "Producername", "algorithm": algorithms[producer_algorithm], "threshold": producer_threshold, "weight": producer_weight},
        {'column': "Subproducername", "algorithm": algorithms[subproducer_algorithm], "threshold": subproducer_threshold, "weight": subproducer_weight},
        {'column': "subproduceraddress", "algorithm": algorithms[address_algorithm], "threshold": address_threshold, "weight": address_weight}
    ]

    # Create a blocking key
    df['BlockKey'] = df['Producername'].str[:5] + df['Subproducername'].str[:5]

    for conf in config:
        match_score_col = f"{conf['column']}MatchScore"
        df[match_score_col] = np.nan  # Initialize with NaN

    df['WeightedScore'] = np.nan

    # Function to calculate match score
    def calculate_match_score(value1, value2, algorithm):
        if algorithm == "fuzz":
            return fuzz.ratio(value1, value2) / 100
        elif algorithm == "token_sort":
            return fuzz.token_sort_ratio(value1, value2) / 100
        elif algorithm == "jaro_winkler":
            return textdistance.jaro_winkler(value1, value2)
        elif algorithm == "levenshtein":
            return textdistance.levenshtein.normalized_similarity(value1, value2)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    # Function to assign clusters within each block
    def assign_cluster_ids_block(block_df, config, final_threshold, start_cluster_id):
        block_df = block_df.reset_index(drop=True)
        current_cluster_id = start_cluster_id

        for i in range(len(block_df)):
            if pd.isna(block_df.at[i, "ClusterId"]):
                block_df.at[i, "ClusterId"] = current_cluster_id
                for j in range(i + 1, len(block_df)):
                    weighted_score_sum = 0
                    total_weight = 0

                    for conf in config:
                        column = conf["column"]
                        algorithm = conf["algorithm"]
                        threshold = conf["threshold"]
                        weight = conf["weight"]

                        match_score = calculate_match_score(block_df.at[i, column], block_df.at[j, column], algorithm)

                        if match_score >= threshold:
                            weighted_score_sum += match_score * weight
                        total_weight += weight

                    if total_weight > 0:
                        final_weighted_score = weighted_score_sum / total_weight

                        if final_weighted_score >= final_threshold:
                            block_df.at[j, "ClusterId"] = block_df.at[i, "ClusterId"]
                            block_df.at[j, "WeightedScore"] = final_weighted_score
                current_cluster_id += 1
        return block_df, current_cluster_id

    # Process each block
    unique_blocks = df['BlockKey'].unique()
    start_cluster_id = 0
    processed_blocks = []

    for key in unique_blocks:
        block_df = df[df['BlockKey'] == key]
        processed_block, start_cluster_id = assign_cluster_ids_block(block_df, config, final_threshold, start_cluster_id)
        processed_blocks.append(processed_block)

    df = pd.concat(processed_blocks, ignore_index=True)

    # Display the results
    st.write("Clustered Data:")
    st.dataframe(df)
    
    # Merge rows into a single row per cluster
    def merge_clusters(df):
        merged_data = []
        for cluster_id in df["ClusterId"].unique():
            cluster_df = df[df["ClusterId"] == cluster_id]
            merged_row = {
                'ClusterId': cluster_id,
                'Producername': cluster_df['Producername'].iloc[0],
                'Subproducername': cluster_df['Subproducername'].iloc[0],
                'subproduceraddress': cluster_df['subproduceraddress'].iloc[0]
            }
            merged_data.append(merged_row)
        return pd.DataFrame(merged_data)

    merged_df = merge_clusters(df)
    
    st.write("Merged Master Data:")
    st.dataframe(merged_df)

    # Download buttons
    clustered_csv = df.to_csv(index=False).encode('utf-8')
    merged_csv = merged_df.to_csv(index=False).encode('utf-8')

    st.download_button("Download Clustered Data", clustered_csv, "clustered_data.csv", "text/csv")
    st.download_button("Download Merged Data", merged_csv, "merged_data.csv", "text/csv")
