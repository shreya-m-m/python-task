
import streamlit as st
import pandas as pd
import kagglehub
import shutil
import os

# Title of the app
st.title("Sales Forecasting Dataset")

# Description
st.write("""
Download the dataset from Kaggle, saves it and displays the data from the CSV file in a tabular format.
""") 

# Download dataset using kagglehub
@st.cache_data  # Cache this function to avoid re-downloading the dataset on each run
def download_and_save_dataset():
    # Download the dataset using kagglehub
    dataset_path = kagglehub.dataset_download("rohitsahoo/sales-forecasting")

    print(f"Dataset downloaded to: {dataset_path}")

    local_save_path = os.path.join(os.getcwd(), "sales_forecasting_dataset")
    
    print(f"Local save path: {local_save_path}")
    
    # Move the downloaded dataset to the local save path
    if os.path.exists(local_save_path):
        shutil.rmtree(local_save_path)  # Clear any existing dataset folder
    
    shutil.move(dataset_path, local_save_path)
    return local_save_path

# Step 1: Download and save dataset
st.write("### Step 1: Downloading Dataset...")
local_dataset_path = download_and_save_dataset()

if local_dataset_path:
    st.success(f"Dataset downloaded and saved successfully! Files are in: {local_dataset_path}")

    # Locate the CSV file in the downloaded folder
    csv_file = f"{local_dataset_path}/train.csv"

    # Step 2: Load the dataset
    @st.cache_data
    def load_data(file_path):
        # Load the CSV data into a Pandas DataFrame
        data = pd.read_csv(file_path)
        return data

    st.write("### Step 2: Displaying Dataset...")
    if os.path.exists(csv_file):
        data = load_data(csv_file)
        st.dataframe(data.style.hide(axis="index"), use_container_width=True)
        print(data)
    else:
        st.error(f"CSV file not found in the dataset folder: {local_dataset_path}")
else:
    st.error("Failed to download the dataset. Please check your Kaggle setup.")
