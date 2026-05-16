import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="pycissa",
    page_icon="assets/pycissa.jpg",
    layout="wide",
)

# Load CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("assets/pycissa.jpg", width=150)
    st.title("pycissa")

# App header
st.title("pycissa: Time Series Analysis with CISSA")

# Main content
tab1, tab2, tab3 = st.tabs(["Description", "Load Data", "Run CISSA"])

with tab1:
    st.write("Welcome to the pycissa web app! This is the main page.")

with tab2:
    from pages import data_upload

with tab3:
    st.write("This is where you will run CISSA.")
