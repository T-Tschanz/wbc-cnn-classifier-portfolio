
import pandas as pd
import psutil
import streamlit as st

def export_results_to_csv(df):
    csv = df.to_csv(index=False)
    st.download_button("Download results as CSV", data=csv, file_name="results.csv", mime="text/csv")

def monitor_memory():
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 ** 2
    st.sidebar.write(f"Memory Usage: {mem:.2f} MB")
    return mem
