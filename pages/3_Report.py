import streamlit as st
import face_rec

# st.set_page_config(page_title="Reporting", layout="wide")
st.subheader("Reporting")

# Retrieve log data and show in report.py
# extract data from redis list
name = "attendance:logs"


def load_logs(name, end=-1):
    logs_list = face_rec.r.lrange(name, 0, end)
    return logs_list


# tabs to show the info
tab1, tab2 = st.tabs(["Registered Data", "Logs"])

with tab1:
    if st.button("Refresh Data"):
        with st.spinner("Retrieving Data from Redis DB..."):
            redis_face_db = face_rec.retrieve_data(name="academy:register")
            st.dataframe(redis_face_db[["Name", "Role"]])

with tab2:
    if st.button("Refresh Logs"):
        st.write(load_logs(name=name))
