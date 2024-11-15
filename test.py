import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textdistance
from rapidfuzz import fuzz

st.title("Advanced Data Clustering & Mastering App")
st.write("Cluster and merge data based on advanced similarity metrics.")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'current_cluster_id' not in st.session_state:
    st.session_state.current_cluster_id = 0

def clean_text(text):
    return text.strip().upper() if pd.notna(text) else ""

def cosine_similarity_score(value1, value2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([value1, value2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]

def damerau_levenshtein_score(value1, value2):
    return textdistance.damerau_levenshtein.normalized_similarity(value1, value2)

def tfidf_ngram_similarity(value1, value2):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    tfidf_matrix = vectorizer.fit_transform([value1, value2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

input_method = st.radio("Choose Input Method", ["Upload CSV", "Enter Data Manually"])

if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required_columns = ['ProducerID', 'SubproducerID', 'Producername', 'Subproducername', 'subproduceraddress']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Column '{col}' not found in uploaded file")
                st.stop()
        df["Producername"] = df["Producername"].apply(clean_text)
        df["Subproducername"] = df["Subproducername"].apply(clean_text)
        df["subproduceraddress"] = df["subproduceraddress"].apply(clean_text)
        st.session_state.df = df

df = st.session_state.df

if df is not None:
    st.sidebar.header("Adjust Weights & Thresholds")
    
    producer_threshold = st.sidebar.slider("Producername Threshold", 0.0, 1.0, 0.9)
    producer_weight = st.sidebar.slider("Producername Weight", 0.0, 1.0, 0.6)
    subproducer_threshold = st.sidebar.slider("Subproducername Threshold", 0.0, 1.0, 0.85)
    subproducer_weight = st.sidebar.slider("Subproducername Weight", 0.0, 1.0, 0.8)
    address_threshold = st.sidebar.slider("Address Threshold", 0.0, 1.0, 0.85)
    address_weight = st.sidebar.slider("Address Weight", 0.0, 1.0, 0.75)
    final_threshold = st.sidebar.slider("Final Weighted Score Threshold", 0.0, 1.0, 0.85)

    config = [
        {'column': "Producername", "algorithm": "cosine", "threshold": producer_threshold, "weight": producer_weight},
        {'column': "Subproducername", "algorithm": "damerau_levenshtein", "threshold": subproducer_threshold, "weight": subproducer_weight},
        {'column': "subproduceraddress", "algorithm": "tfidf_ngram", "threshold": address_threshold, "weight": address_weight}
    ]

    df['BlockKey'] = df['Producername'].str[:5] + "_" + df['Subproducername'].str[:5]

    def assign_clusters_block(df_block, config, final_threshold):
        df_block = df_block.reset_index(drop=True)
        df_block['ClusterId'] = np.nan
        df_block['WeightedScore'] = np.nan

        for i in range(len(df_block)):
            if pd.isna(df_block.at[i, 'ClusterId']):
                df_block.at[i, 'ClusterId'] = st.session_state.current_cluster_id
                for j in range(i + 1, len(df_block)):
                    total_score = 0
                    total_weight = 0

                    for conf in config:
                        column = conf['column']
                        if conf['algorithm'] == "cosine":
                            score = cosine_similarity_score(df_block.at[i, column], df_block.at[j, column])
                        elif conf['algorithm'] == "damerau_levenshtein":
                            score = damerau_levenshtein_score(df_block.at[i, column], df_block.at[j, column])
                        elif conf['algorithm'] == "tfidf_ngram":
                            score = tfidf_ngram_similarity(df_block.at[i, column], df_block.at[j, column])
                        else:
                            score = 0

                        if score >= conf['threshold']:
                            total_score += score * conf['weight']
                        total_weight += conf['weight']

                    if total_weight > 0:
                        final_score = total_score / total_weight
                        if final_score >= final_threshold:
                            df_block.at[j, 'ClusterId'] = st.session_state.current_cluster_id
                            df_block.at[j, 'WeightedScore'] = final_score
                st.session_state.current_cluster_id += 1

        return df_block

    blocks = df['BlockKey'].unique()
    processed_blocks = [assign_clusters_block(df[df['BlockKey'] == block], config, final_threshold) for block in blocks]
    clustered_df = pd.concat(processed_blocks, ignore_index=True)

    # Convert IDs to strings without commas
    clustered_df['ProducerID'] = clustered_df['ProducerID'].astype(int).astype(str)
    clustered_df['SubproducerID'] = clustered_df['SubproducerID'].astype(int).astype(str)

    st.write("Clustered Data with Scores:")
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
