import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time


mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



lbl=['Close','Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

# Add these variables after other initializations
# Modify these values after initialization
# Modify these values
ALARM_THRESHOLD = 15
ALARM_COOLDOWN = 5  # seconds
BLINK_THRESHOLD = 1  # reduced from 3 to make closed detection faster
SCORE_INCREASE = 1.5  # new constant for score increase rate

last_alarm_time = time.time()
alarm_active = False
blink_counter = 0

# Add after other imports
if not os.path.exists('detected'):
    os.makedirs('detected')

while(True):
    ret, frame = cap.read()
    if not ret:
        break
    
    height,width = frame.shape[:2] 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    
    # Only process eyes if face is detected
    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Use the first face
        roi_gray = gray[y:y+h, x:x+w]
        
        left_eye = leye.detectMultiScale(roi_gray)
        right_eye = reye.detectMultiScale(roi_gray)
        
        eyes_detected = len(left_eye) > 0 and len(right_eye) > 0
        eyes_closed = False
        
        if eyes_detected:
            # Process right eye
            for (ex,ey,ew,eh) in right_eye:
                r_eye = roi_gray[ey:ey+eh, ex:ex+ew]
                r_eye = cv2.resize(r_eye,(24,24))
                r_eye = r_eye/255
                r_eye = r_eye.reshape(24,24,-1)
                r_eye = np.expand_dims(r_eye,axis=0)
                rpred = np.argmax(model.predict(r_eye), axis=-1)
                break
                
            # Process left eye
            for (ex,ey,ew,eh) in left_eye:
                l_eye = roi_gray[ey:ey+eh, ex:ex+ew]
                l_eye = cv2.resize(l_eye,(24,24))
                l_eye = l_eye/255
                l_eye = l_eye.reshape(24,24,-1)
                l_eye = np.expand_dims(l_eye,axis=0)
                lpred = np.argmax(model.predict(l_eye), axis=-1)
                break
                
            eyes_closed = (rpred[0]==0 and lpred[0]==0)
            
            if eyes_closed:
                blink_counter += 1
                if blink_counter > BLINK_THRESHOLD:  # Only increase score after blink threshold
                    score = min(score + SCORE_INCREASE, ALARM_THRESHOLD + 5)
                cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            else:
                blink_counter = 0
                score = max(score - 0.5, 0)  # Decrease score more slowly
                cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
                if alarm_active:
                    sound.stop()
                    alarm_active = False
        
        # Handle alarm
        current_time = time.time()
        if score >= ALARM_THRESHOLD and not alarm_active and (current_time - last_alarm_time) > ALARM_COOLDOWN:
            try:
                sound.play()
                alarm_active = True
                last_alarm_time = current_time
                cv2.imwrite(os.path.join('detected', f'drowsy_{int(current_time)}.jpg'), frame)
            except:
                pass
            
            thicc = min(thicc + 2, 16)
        else:
            thicc = max(2, thicc - 1)  # Smoother thickness decrease
            
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
    
    # Display info
    cv2.putText(frame,f'Score:{int(score)}',(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # Add quit button and instructions
    cv2.putText(frame, "Press 'q' to quit", (width-150, height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)
    cv2.imshow('frame',frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting program...")
        break
cap.release()
cv2.destroyAllWindows()
