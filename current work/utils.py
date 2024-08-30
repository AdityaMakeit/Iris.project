import cv2
import numpy as np
import mediapipe as mp



def calEratio(eye_landmarks, frameW, frameH):
    eyePoints = [(int(landmark.x * frameW), int(landmark.y * frameH)) for landmark in eye_landmarks]
    A = np.linalg.norm(np.array(eyePoints[1]) - np.array(eyePoints[5]))
    B = np.linalg.norm(np.array(eyePoints[2]) - np.array(eyePoints[4]))
    C = np.linalg.norm(np.array(eyePoints[0]) - np.array(eyePoints[3]))
    ear = (A + B) / (2.0 * C)
    return ear

def Dfacerect(frame, face_landmarks, frameW, frameH):
    minX, minY, maxX, maxY = frameW, frameH, 0, 0
    for landmark in face_landmarks.landmark:
        x, y = int(landmark.x * frameW), int(landmark.y * frameH)
        if x < minX:
            minX = x
        if x > maxX:
            maxX = x
        if y < minY:
            minY = y
        if y > maxY:
            maxY = y
    cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 255, 0), 2)
    return frame

def DeyeLMD(frame, face_landmarks, frameW, frameH, prevX, prevY):
    for landmID in range(468, 478):  # Landmark IDs for both eyes
        landmark = face_landmarks.landmark[landmID]
        x = int(landmark.x * frameW)
        y = int(landmark.y * frameH)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # Track eye movement (right n left)
        if landmID == 468:  # we here consider only one eye for simplicity
            if prevX is not None:
                displayTxT = "None"
                if x > prevX:
                    displayTxT = "Right ->"
         
                elif x < prevX:
                    displayTxT = "Left <-"
              
                    

                cv2.putText(frame, displayTxT, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            prevX = x

        if landmID == 468:
            # Track eye movement (up nd down)
            if prevY is not None:
                displayTxT_y = "None"
                if y < prevY:
                    displayTxT_y = "Up ^"
                elif y > prevY:
                    displayTxT_y = "Down v"
                cv2.putText(frame, displayTxT_y, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            prevY = y

    return frame, prevX, prevY

#calculating eye aspect ratio funtion
def calEAR(face_landmarks, frameW, frameH):
    L_eyeLMD = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
    R_eyeLMD = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
    earL = calEratio(L_eyeLMD, frameW, frameH)
    earR = calEratio(R_eyeLMD, frameW, frameH)
    avg_ear = (earL + earR) / 2.0
    return avg_ear
