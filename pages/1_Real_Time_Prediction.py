import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer, ClientSettings
import av
import time
from twilio.rest import Client

account_sid = "ACbf54242144dc7d8f163ed4d03e94d86d"
auth_token = "83ef734bee74f7e02f28c5c1386cdb3c"
client = Client(account_sid, auth_token)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    )

token = client.tokens.create()


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


webrtc_streamer(key="realtimePrediction", client_settings=WEBRTC_CLIENT_SETTINGS, video_frame_callback=video_frame_callback)
