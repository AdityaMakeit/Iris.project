import cv2
import mediapipe as mp
import screen_brightness_control as sbc
from utils import Dfacerect, DeyeLMD, calEAR

def increaseBrightness():
    currentBrightness = sbc.get_brightness(display=0)[0]
    newBrightness = min(currentBrightness + 10, 100)
    sbc.set_brightness(newBrightness, display=0)

def initialize_webcam():
    return cv2.VideoCapture(0)

def process_frame(frame, face_mesh, earRatio, blinkFrm, blinkCount, blinkTotal, taskBLinkCount, prevX, prevY):
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landMP = output.multi_face_landmarks
    frameH, frameW, _ = frame.shape

    if landMP:
        for faceLMD in landMP:
            frame = Dfacerect(frame, faceLMD, frameW, frameH)
            frame, prevX, prevY = DeyeLMD(frame, faceLMD, frameW, frameH, prevX, prevY)
            avg_ear = calEAR(faceLMD, frameW, frameH)

            if avg_ear < earRatio:
                blinkCount += 1
            else:
                if blinkCount >= blinkFrm:
                    blinkTotal += 1
                    blinkCount = 0
                    if blinkTotal % taskBLinkCount == 0:
                        increaseBrightness()
                else:
                    blinkCount = 0

            cv2.putText(frame, f"Blinks: {blinkTotal}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame, blinkCount, blinkTotal, prevX, prevY
