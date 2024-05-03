import streamlit as st
from servo import Servo
import time
import requests

from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

manual_text = '''
Press keys on keyboard to control servos!
    Q: Quit
    W: tilt up
    S: tilt down
    A: pan left
    D: pan right
'''
pan = Servo(pin=13, min_angle=-90, max_angle=90)   # pan_servo_pin (BCM)
tilt = Servo(pin=12, min_angle=-90, max_angle=90)  # be careful to limit the angle of the steering gear

def Control_Servo(key, panAngle=0, tiltAngle=0):
    if key == ord('q'):
        exit()
    elif key == ord('w'):
        tiltAngle -= 1
    elif key == ord('s'):
        tiltAngle += 1
    elif key == ord('a'):
        panAngle += 1
    elif key == ord('d'):
        panAngle -= 1
    pan.set_angle(panAngle)
    tilt.set_angle(tiltAngle)
    return panAngle, tiltAngle

def send_line_notify(notification_message, image_path=None):
    line_notify_token = 'AeUFlx5kbUL0NygRanR5YGJ5qvkHmu8IQ07uF9QLtKQ'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'\n{notification_message}'}
    files = {'imageFile': open(image_path, 'rb')} if image_path else None
    response = requests.post(line_notify_api, headers=headers, data=data, files=files)
    if files:
        files['imageFile'].close()    
    return response.status_code
    
def capture_and_notify(frame_PIL, notification_message):
    temp_image_path = 'temp.jpg'
    frame_array = np.array(frame_PIL)
    frame_array_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_image_path, frame_array_bgr)
    send_line_notify(notification_message, temp_image_path)

#main
def main():
    model = YOLO("yolov8n-face.pt")
    if 'cap' not in st.session_state:
        st.session_state.cap = cv2.VideoCapture('tcp://127.0.0.1:5000', cv2.CAP_FFMPEG)

    if 'flg' not in st.session_state:
        st.session_state.flg = False

    interval = 60
    last_inference_time = time.time()

    frame_holder = st.empty()
    alert_holder = st.empty()
    st.title('Baby Camera')
    
    if st.button("Check",help='Please press the button to resume monitoring after checking on the baby.'):
        st.session_state.flg = False

    st.session_state.outing = st.checkbox("Pause monitoring ")

    if st.button("Take a photo!"):
        ret, frame = st.session_state.cap.read()
        if ret:
            frame = cv2.flip(frame, -1)
            img_pil = Image.fromarray(frame[:, :, ::-1])  # Convert BGR to RGB
            status_code = capture_and_notify(img_pil,'写真を送ります')

    while True:
        if st.session_state.outing == False:
            ret, frame = st.session_state.cap.read()
            if ret:
                current_time = time.time()
                frame = cv2.flip(frame, -1)
                results = model(frame, imgsz=160)
                img_annotated = results[0].plot()  # Get annotated image
                img_pil = Image.fromarray(img_annotated[:, :, ::-1])  # Convert BGR to RGB
                frame_holder.image(img_pil)

                if len(results[0].boxes) > 0:
                    last_inference_time = current_time
                    alert_holder.empty()
                else:
                    if current_time - last_inference_time > interval:
                        alert_holder.markdown('**赤ちゃんを確認してください**')
                        if st.session_state.flg == False:
                            status_code = capture_and_notify(img_pil,'赤ちゃんを確認してください')
                        st.session_state.flg = True  # Set flag to true if condition is met

if __name__ == "__main__":    
    main()
