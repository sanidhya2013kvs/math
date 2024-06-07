import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
#from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
genai.configure(api_key="AIzaSyBKaofeR0V-mt115AA54Dym8uJ34Hk1oWA")
model = genai.GenerativeModel('gemini-1.5-flash')
st.set_page_config(layout="wide")
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)


def getHandInfo(img):
    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(img, draw=True, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand)
        print(fingers)
        return fingers, lmList
    else:
        return None

def draw(info,prev_pos,canvas):
    fingers, lmList = info
    current_pos= None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None: prev_pos = current_pos
        cv2.line(canvas,current_pos,prev_pos,(255,0,255),10)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(img)

    return current_pos, canvas


prev_pos= None
canvas=None
image_combined = None
output_text= ""
col1, col2 = st.columns([3,2])
with col1:
  class VideoProcessor:
    def recv(self, frame):
      img = frame.to_ndarray(format="bgr24")
      img = cv2.flip(img, 1)

      if canvas is None:
        canvas = np.zeros_like(img)
      info = getHandInfo(img)
      if info:
        fingers, lmList = info
        prev_pos,canvas = draw(info, prev_pos,canvas)
        image_combined= cv2.addWeighted(img,0.7,canvas,0.3,0)
        img=image_combined



      return av.VideoFrame.from_ndarray(img, format="bgr24")

  ctx=webrtc_streamer(key="example",video_processor_factory=VideoProcessor,
      rtc_configuration={
          "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
              })
