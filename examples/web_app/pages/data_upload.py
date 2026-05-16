import streamlit as st
import pandas as pd
import statsmodels.datasets as datasets
import matplotlib.pyplot as plt

# Load CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Data Upload")

# Function to load and print basic info about a dataset
def load_dataset(dataset_name, dataset_loader):
    data = dataset_loader()
    df = data.data  # Most datasets return a Bunch object with a 'data' attribute
    return df

# Data loading options
data_source = st.radio(
    "Choose data source", ("Upload a file", "Load a preset dataset")
)

if data_source == "Upload a file":
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_excel(uploaded_file)
        st.write(dataframe)
        time_column = st.selectbox("Select time column", dataframe.columns)
        data_column = st.selectbox("Select data column", dataframe.columns)
        st.session_state["time_array"] = dataframe[time_column]
        st.session_state["data_array"] = dataframe[data_column]


elif data_source == "Load a preset dataset":
    preset_dataset = st.selectbox(
        "Choose a preset dataset",
        ("EPC", "Sunspots", "Nile", "CO2", "Macrodata"),
    )
    if st.button("Load Dataset"):
        if preset_dataset == "EPC":
            dataframe = pd.read_excel(
                "https://github.com/LukeAFullard/pyCiSSA/raw/refs/heads/main/data/EPC.xlsx",
                sheet_name="Data",
            )
            st.write(dataframe)
            st.session_state["time_array"] = dataframe["Date"]
            st.session_state["data_array"] = dataframe[
                "Electric power consumption in Spain"
            ]

        elif preset_dataset == "Sunspots":
            sunspots_df = load_dataset("Sunspots", datasets.sunspots.load_pandas)
            st.write(sunspots_df)
            st.session_state["time_array"] = sunspots_df["YEAR"]
            st.session_state["data_array"] = sunspots_df["SUNACTIVITY"]

        elif preset_dataset == "Nile":
            nile_df = load_dataset("Nile", datasets.nile.load_pandas)
            st.write(nile_df)
            st.session_state["time_array"] = nile_df["year"]
            st.session_state["data_array"] = nile_df["volume"]

        elif preset_dataset == "CO2":
            co2_df = load_dataset("CO2", datasets.co2.load_pandas)
            st.write(co2_df)
            st.session_state["time_array"] = co2_df.index
            st.session_state["data_array"] = co2_df["co2"]

        elif preset_dataset == "Macrodata":
            macrodata_df = load_dataset("Macrodata", datasets.macrodata.load_pandas)
            st.write(macrodata_df)
            st.session_state["time_array"] = (
                macrodata_df["year"] + macrodata_df["quarter"] / 4
            )
            st.session_state["data_array"] = macrodata_df["realgdp"]

if "time_array" in st.session_state and "data_array" in st.session_state:
    fig, ax = plt.subplots()
    ax.plot(st.session_state["time_array"], st.session_state["data_array"])
    st.pyplot(fig)
