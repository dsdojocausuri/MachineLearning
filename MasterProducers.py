import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import textdistance

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

# Utility function to clean and standardize text
def clean_text(text):
    return text.strip().upper() if pd.notna(text) else ""

# Function to calculate Damerau-Levenshtein similarity
def damerau_levenshtein_score(value1, value2):
    return textdistance.damerau_levenshtein.normalized_similarity(value1, value2)

# Function to calculate cosine similarity using textdistance
def cosine_similarity_score(value1, value2):
    return textdistance.cosine.normalized_similarity(value1, value2)

# Function to convert DataFrame to CSV for download
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

st.title("Master Data Clustering and Merging App")

# Input Data
input_method = st.radio("Choose Input Method", ["Upload CSV", "Enter Data Manually"])

# Handling CSV upload
if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required_columns = ['ProducerID', 'SubproducerID', 'Producername', 'Subproducername', 'subproduceraddress']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing column: {col}")
                st.stop()
        df["Producername"] = df["Producername"].apply(clean_text)
        df["Subproducername"] = df["Subproducername"].apply(clean_text)
        df["subproduceraddress"] = df["subproduceraddress"].apply(clean_text)
        st.session_state.df = df

# Load DataFrame from session state
df = st.session_state.df

if df is not None:
    # Sidebar settings for thresholds and weights
    st.sidebar.header("Adjust Weights & Thresholds")
    
    # Threshold sliders
    producer_threshold = st.sidebar.slider("Producername Threshold", 0.0, 1.0, 0.9)
    subproducer_threshold = st.sidebar.slider("Subproducername Threshold", 0.0, 1.0, 0.85)
    address_threshold = st.sidebar.slider("Address Threshold", 0.0, 1.0, 0.85)
    final_threshold = st.sidebar.slider("Final Matching Threshold", 0.0, 1.0, 0.85)

    # Weight sliders
    producer_weight = st.sidebar.slider("Weight for Producername", 0.0, 1.0, 0.6)
    subproducer_weight = st.sidebar.slider("Weight for Subproducername", 0.0, 1.0, 0.8)
    address_weight = st.sidebar.slider("Weight for Address", 0.0, 1.0, 0.75)

    # Function to assign clusters with individual match scores and weighted score
    def assign_clusters(df, producer_threshold, subproducer_threshold, address_threshold, final_threshold):
        df['ClusterId'] = np.nan
        df['ProducerScore'] = np.nan
        df['SubproducerScore'] = np.nan
        df['AddressScore'] = np.nan
        df['WeightedScore'] = np.nan

        cluster_id = 0

        for i in range(len(df)):
            if pd.isna(df.at[i, 'ClusterId']):
                df.at[i, 'ClusterId'] = cluster_id
                for j in range(i + 1, len(df)):
                    # Calculate similarity scores
                    producer_score = fuzz.ratio(df.at[i, 'Producername'], df.at[j, 'Producername']) / 100
                    subproducer_score = fuzz.token_sort_ratio(df.at[i, 'Subproducername'], df.at[j, 'Subproducername']) / 100
                    address_score = cosine_similarity_score(df.at[i, 'subproduceraddress'], df.at[j, 'subproduceraddress'])

                    # Calculate weighted score
                    total_weight = producer_weight + subproducer_weight + address_weight
                    weighted_score = (
                        (producer_score * producer_weight) +
                        (subproducer_score * subproducer_weight) +
                        (address_score * address_weight)
                    ) / total_weight

                    # Assign to cluster if above thresholds
                    if (producer_score >= producer_threshold and 
                        subproducer_score >= subproducer_threshold and 
                        address_score >= address_threshold and 
                        weighted_score >= final_threshold):
                        
                        df.at[j, 'ClusterId'] = cluster_id
                        df.at[j, 'ProducerScore'] = producer_score
                        df.at[j, 'SubproducerScore'] = subproducer_score
                        df.at[j, 'AddressScore'] = address_score
                        df.at[j, 'WeightedScore'] = weighted_score
                cluster_id += 1

        return df

    # Assign clusters to the DataFrame
    clustered_df = assign_clusters(df, producer_threshold, subproducer_threshold, address_threshold, final_threshold)

    # Ensure IDs are displayed without commas
    clustered_df['ProducerID'] = clustered_df['ProducerID'].astype(int).astype(str)
    clustered_df['SubproducerID'] = clustered_df['SubproducerID'].astype(int).astype(str)

    # Display Clustered Data with Scores
    st.write("Clustered Data with Individual Scores and Weighted Score:")
    st.dataframe(clustered_df)
    csv_clustered = convert_df_to_csv(clustered_df)
    st.download_button("Download Clustered Data", data=csv_clustered, file_name="clustered_data.csv", mime='text/csv')

    # Matched Merged Concatenated Dataset
    def merge_clusters_concatenated(df):
        merged_data = []
        for cluster_id in df['ClusterId'].unique():
            cluster_df = df[df['ClusterId'] == cluster_id]
            merged_row = {
                'ProducerID': " | ".join(cluster_df['ProducerID'].unique()),
                'SubproducerID': " | ".join(cluster_df['SubproducerID'].unique()),
                'Producername': " | ".join(cluster_df['Producername'].unique()),
                'Subproducername': " | ".join(cluster_df['Subproducername'].unique()),
                'subproduceraddress': " | ".join(cluster_df['subproduceraddress'].unique())
            }
            merged_data.append(merged_row)
        return pd.DataFrame(merged_data)

    merged_df_concat = merge_clusters_concatenated(clustered_df)
    st.write("Matched Merged Concatenated Dataset:")
    st.dataframe(merged_df_concat)
    csv_concat = convert_df_to_csv(merged_df_concat)
    st.download_button("Download Concatenated Data", data=csv_concat, file_name="concatenated_data.csv", mime='text/csv')

    # Merged Golden Record Dataset
    def merge_clusters_golden(df):
        merged_data = []
        for cluster_id in df['ClusterId'].unique():
            cluster_df = df[df['ClusterId'] == cluster_id]
            merged_row = {
                'ProducerID': str(int(cluster_df['ProducerID'].max())),
                'SubproducerID': str(int(cluster_df['SubproducerID'].max())),
                'Producername': cluster_df.loc[cluster_df['Producername'].apply(len).idxmax(), 'Producername'],
                'Subproducername': cluster_df.loc[cluster_df['Subproducername'].apply(len).idxmax(), 'Subproducername'],
                'subproduceraddress': cluster_df.loc[cluster_df['subproduceraddress'].apply(len).idxmax(), 'subproduceraddress']
            }
            merged_data.append(merged_row)
        return pd.DataFrame(merged_data)

    merged_df_golden = merge_clusters_golden(clustered_df)
    st.write("Merged Golden Record Dataset:")
    st.dataframe(merged_df_golden)
    csv_golden = convert_df_to_csv(merged_df_golden)
    st.download_button("Download Golden Records", data=csv_golden, file_name="golden_records.csv", mime='text/csv')
