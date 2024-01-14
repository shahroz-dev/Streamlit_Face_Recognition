import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

# st.set_page_config(page_title="Prediction")
st.subheader("Real-Time Attendance System")

# Retrieve the data from Redis Database
with st.spinner("Retrieving Data from Redis DB..."):
    redis_face_db = face_rec.retrieve_data(name="academy:register")
st.success("Data Successfully Retrieved from Redis")

# time
waitTime = 5  # time in sec
setTime = time.time()
realtimepred = face_rec.RealTimePred()


# Real Time Prediction
# streamlit webrtc
def video_frame_callback(frame):
    global setTime

    img = frame.to_ndarray(format="bgr24")  # 3-dimension numpy array
    # operation that you can perform on the array
    pred_img = realtimepred.face_prediction(img, redis_face_db, "facial_features", ["Name", "Role"], 0.5)

    timenow = time.time()
    difftime = timenow - setTime
    if difftime > waitTime:
        realtimepred.saveLogs_redis()
        setTime = time.time()

        print("Save Data to Redis Database")

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="realtimePrediction",
                rtc_configuration={  # Add this config
                    "iceServers": [{"urls": ["stun:global.stun.twilio.com:3478?transport=udp"]}]
                },
                media_stream_constraints={
                    "video": True,
                    "audio": False
                },
                video_frame_callback=video_frame_callback)
