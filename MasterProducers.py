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

# Initialize DataFrame as None initially
df = None

@st.cache_data
def clean_text(text):
    """Clean and standardize text inputs."""
    if pd.isna(text):
        return ""
    return text.strip().upper()

@st.cache_data
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
    else:
        return 0.0

# Conditional Data Loading Based on User Selection
if input_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df["Producername"] = df["Producername"].apply(clean_text)
        df["Subproducername"] = df["Subproducername"].apply(clean_text)
        df["subproduceraddress"] = df["subproduceraddress"].apply(clean_text)

elif input_option == "Enter Data Manually":
    st.write("Enter your data below. Add multiple rows separated by commas.")
    producer_input = st.text_area("Enter Producername (comma-separated)", "")
    subproducer_input = st.text_area("Enter Subproducername (comma-separated)", "")
    address_input = st.text_area("Enter Address (comma-separated)", "")
    
    if st.button("Submit Data"):
        # Convert input data to lists
        producers = [clean_text(x) for x in producer_input.split(",")]
        subproducers = [clean_text(x) for x in subproducer_input.split(",")]
        addresses = [clean_text(x) for x in address_input.split(",")]

        if len(producers) == len(subproducers) == len(addresses):
            df = pd.DataFrame({
                "Producername": producers,
                "Subproducername": subproducers,
                "subproduceraddress": addresses
            })
        else:
            st.error("Please ensure all fields have the same number of entries.")

# Proceed if DataFrame is not None
if df is not None:
    st.sidebar.header("Adjust Weights & Thresholds")
    
    producer_weight = st.sidebar.slider("Weight for Producername", 0.0, 1.0, 0.6)
    producer_threshold = st.sidebar.slider("Threshold for Producername", 0.0, 1.0, 0.9)
    
    subproducer_weight = st.sidebar.slider("Weight for Subproducername", 0.0, 1.0, 0.8)
    subproducer_threshold = st.sidebar.slider("Threshold for Subproducername", 0.0, 1.0, 0.85)
    
    address_weight = st.sidebar.slider("Weight for Address", 0.0, 1.0, 0.75)
    address_threshold = st.sidebar.slider("Threshold for Address", 0.0, 1.0, 0.85)
    
    final_threshold = st.sidebar.slider("Final Threshold", 0.0, 1.0, 0.85)

    st.write("Data Loaded:")
    st.dataframe(df)
