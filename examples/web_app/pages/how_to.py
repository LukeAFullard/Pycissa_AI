import streamlit as st

# Load CSS
with open("../assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("How To / Help")

st.write("This is where you will find instructions on how to use the app.")
