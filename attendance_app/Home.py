import streamlit as st

st.set_page_config(page_title="Attendance System", layout="wide")

st.header("Attendance System using Face Recognition")

with st.spinner("Loading Models and Connecting to Redis db..."):
    import face_rec

st.success("Model Loaded successfully")
st.success("Redis db successfully connected")
