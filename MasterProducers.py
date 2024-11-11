import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import textdistance

# Streamlit App Title
st.title("Data Clustering & Mastering App")
st.write("This app allows you to cluster and merge data based on similarity metrics. Adjust the thresholds, weights, and algorithms below to control the clustering behavior.")

# Option for Input Type: Upload CSV or Enter Data Manually
input_option = st.radio("Choose Input Method", ("Upload CSV", "Enter Data Manually"))

# Function to clean and standardize text inputs
def clean_text(text):
    if pd.isna(text):
        return ""
    return text.strip().upper()

# Initialize the DataFrame
df = None

# Upload CSV Option
if input_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # Convert columns to uppercase for case insensitivity
        df["Producername"] = df["Producername"].apply(clean_text)
        df["Subproducername"] = df["Subproducername"].apply(clean_text)
        df["subproduceraddress"] = df["subproduceraddress"].apply(clean_text)

# Manual Data Entry Option
elif input_option == "Enter Data Manually":
    st.write("Enter your data below. Add multiple rows separated by commas.")
    producer_input = st.text_area("Enter Producername (separated by commas)", "")
    subproducer_input = st.text_area("Enter Subproducername (separated by commas)", "")
    address_input = st.text_area("Enter Address (separated by commas)", "")

    if st.button("Submit Data"):
        # Convert input data to lists
        producers = [clean_text(x) for x in producer_input.split(",")]
        subproducers = [clean_text(x) for x in subproducer_input.split(",")]
        addresses = [clean_text(x) for x in address_input.split(",")]

        # Check if all lists are of the same length
        if len(producers) == len(subproducers) == len(addresses):
            df = pd.DataFrame({
                "Producername": producers,
                "Subproducername": subproducers,
                "subproduceraddress": addresses
            })
        else:
            st.error("Please ensure all fields have the same number of entries.")

# Proceed if DataFrame is available
if df is not None:
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

    df['BlockKey'] = df['Producername'].str[:5] + df['Subproducername'].str[:5]

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

    # Assign clusters
    def assign_cluster_ids(df, config, final_threshold):
        cluster_id = 0
        for i in range(len(df)):
            if pd.isna(df.at[i, "ClusterId"]):
                df.at[i, "ClusterId"] = cluster_id
                for j in range(i + 1, len(df)):
                    total_score = 0
                    total_weight = 0
                    for conf in config:
                        score = calculate_match_score(df.at[i, conf['column']], df.at[j, conf['column']], conf['algorithm'])
                        if score >= conf['threshold']:
                            total_score += score * conf['weight']
                        total_weight += conf['weight']
                    if total_weight > 0 and (total_score / total_weight) >= final_threshold:
                        df.at[j, "ClusterId"] = cluster_id
                cluster_id += 1
        return df

    df = assign_cluster_ids(df, config, final_threshold)

    # Display the clustered data
    st.write("Clustered Data:")
    st.dataframe(df)

    # Download the results
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Clustered Data", csv, "clustered_data.csv", "text/csv")
