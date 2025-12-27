# styles.py

import streamlit as st

def apply_text_size(text_size_percent: int = 100):
    """Scale all Streamlit text by the given percentage."""
    st.markdown(f"""
    <style>
      html, body, [class*="css"] {{
        font-size: {text_size_percent}%;
      }}
    </style>
    """, unsafe_allow_html=True)


