import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import textdistance
from joblib import Parallel, delayed

# Clean and standardize text inputs
def clean_text(text):
    return text.strip().upper() if pd.notna(text) else ""

# Calculate match score using different algorithms
def calculate_match_score(value1, value2, algorithm):
    if algorithm == "fuzz":
        return fuzz.ratio(value1, value2) / 100
    elif algorithm == "token_sort":
        return fuzz.token_sort_ratio(value1, value2) / 100
    elif algorithm == "jaro_winkler":
        return textdistance.jaro_winkler(value1, value2)
    elif algorithm == "levenshtein":
        return textdistance.levenshtein.normalized_similarity(value1, value2)
    return 0.0

# Load CSV file
def load_csv(file_path):
    return pd.read_csv(file_path)

# Generate a blocking key with more specificity
def generate_block_key(df):
    return (
        df['Producername'].str[:5] + "_" +
        df['Subproducername'].str[:5] + "_" +
        df['subproduceraddress'].str[:5]
    )

# Assign clusters within each block with a global cluster ID counter
def assign_clusters_block(df_block, config, final_threshold, cluster_id):
    df_block = df_block.reset_index(drop=True)
    df_block['ClusterId'] = np.nan
    df_block['Producername_Score'] = np.nan
    df_block['Subproducername_Score'] = np.nan
    df_block['Address_Score'] = np.nan
    df_block['Weighted_Score'] = np.nan

    for i in range(len(df_block)):
        if pd.isna(df_block.at[i, "ClusterId"]):
            df_block.at[i, "ClusterId"] = cluster_id
            for j in range(i + 1, len(df_block)):
                producer_score = 0
                subproducer_score = 0
                address_score = 0
                weighted_score_sum = 0
                total_weight = 0

                # Calculate match scores for each configured column
                for conf in config:
                    column = conf['column']
                    algorithm = conf['algorithm']
                    threshold = conf['threshold']
                    weight = conf['weight']

                    # Ensure both values are non-null before comparing
                    if pd.notna(df_block.at[i, column]) and pd.notna(df_block.at[j, column]):
                        score = calculate_match_score(df_block.at[i, column], df_block.at[j, column], algorithm)
                    else:
                        score = 0

                    if column == 'Producername':
                        producer_score = score
                    elif column == 'Subproducername':
                        subproducer_score = score
                    elif column == 'subproduceraddress':
                        address_score = score

                    if score >= threshold:
                        weighted_score_sum += score * weight
                    total_weight += weight

                final_score = weighted_score_sum / total_weight if total_weight > 0 else 0

                # Assign to cluster if final score is above the threshold
                if final_score >= final_threshold:
                    df_block.at[j, "ClusterId"] = cluster_id
                    df_block.at[j, 'Producername_Score'] = producer_score
                    df_block.at[j, 'Subproducername_Score'] = subproducer_score
                    df_block.at[j, 'Address_Score'] = address_score
                    df_block.at[j, 'Weighted_Score'] = final_score

            cluster_id += 1

    return df_block, cluster_id

# Merge clusters into concatenated dataset with concatenated SubproducerID
def merge_clusters_concatenated(df):
    merged_data = []
    for cluster_id in df['ClusterId'].unique():
        cluster_df = df[df['ClusterId'] == cluster_id]
        if not cluster_df.empty:
            merged_row = {
                'Producername': " | ".join(cluster_df['Producername'].unique()),
                'Subproducername': " | ".join(cluster_df['Subproducername'].unique()),
                'subproduceraddress': " | ".join(cluster_df['subproduceraddress'].unique()),
                'ProducerID': cluster_df['ProducerID'].max(),
                'SubproducerID': " | ".join(map(str, cluster_df['SubproducerID'].unique()))
            }
            merged_data.append(merged_row)
    return pd.DataFrame(merged_data)

# Merge clusters into golden record dataset, picking the max ProducerID and SubproducerID
def merge_clusters_golden(df):
    merged_data = []
    for cluster_id in df['ClusterId'].unique():
        cluster_df = df[df['ClusterId'] == cluster_id]
        if not cluster_df.empty:
            merged_row = {
                'Producername': cluster_df.loc[cluster_df['Producername'].apply(len).idxmax(), 'Producername'],
                'Subproducername': cluster_df.loc[cluster_df['Subproducername'].apply(len).idxmax(), 'Subproducername'],
                'subproduceraddress': cluster_df.loc[cluster_df['subproduceraddress'].apply(len).idxmax(), 'subproduceraddress'],
                'ProducerID': cluster_df['ProducerID'].max(),
                'SubproducerID': cluster_df['SubproducerID'].max()
            }
            merged_data.append(merged_row)
    return pd.DataFrame(merged_data)

# Main Function
def main(file_path):
    df = load_csv(file_path)
    df['Producername'] = df['Producername'].apply(clean_text)
    df['Subproducername'] = df['Subproducername'].apply(clean_text)
    df['subproduceraddress'] = df['subproduceraddress'].apply(clean_text)

    # Add Blocking Key
    df['BlockKey'] = generate_block_key(df)

    # Configuration for Matching
    config = [
        {'column': "Producername", 'algorithm': "fuzz", 'threshold': 0.9, 'weight': 0.6},
        {'column': "Subproducername", 'algorithm': "token_sort", 'threshold': 0.85, 'weight': 0.8},
        {'column': "subproduceraddress", 'algorithm': "jaro_winkler", 'threshold': 0.85, 'weight': 0.75}
    ]
    final_threshold = 0.85

    blocks = df['BlockKey'].unique()
    global_cluster_id = 0

    # Process each block and assign clusters sequentially
    results = []
    for block in blocks:
        block_result, global_cluster_id = assign_clusters_block(
            df[df['BlockKey'] == block].copy(), config, final_threshold, global_cluster_id
        )
        results.append(block_result)

    clustered_df = pd.concat(results, ignore_index=True)

    # Generate merged datasets
    merged_df_concat = merge_clusters_concatenated(clustered_df)
    merged_df_golden = merge_clusters_golden(clustered_df)

    # Save outputs to CSV files
    clustered_df.to_csv("/content/sample_data/clustered_data.csv", index=False)
    merged_df_concat.to_csv("/content/sample_data/merged_concatenated.csv", index=False)
    merged_df_golden.to_csv("/content/sample_data/merged_golden.csv", index=False)

    print("Clustering and merging completed successfully!")

# Run the program
if __name__ == "__main__":
    file_path = "/content/sample_data/POCDQ.csv"
    main(file_path)