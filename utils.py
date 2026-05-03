import streamlit as st

def display_recommendation(pred):
    if pred == 1:
        st.markdown("""
        ### ⚠️ Recommendations:
        - Consult a doctor immediately
        - Monitor BP and cholesterol
        - Avoid junk food
        - Do daily exercise
        """)
    else:
        st.markdown("""
        ### ✅ Healthy Tips:
        - Maintain balanced diet
        - Regular checkup
        - Exercise daily
        """)