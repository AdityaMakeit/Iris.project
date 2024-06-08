import cv2
import mediapipe as mp

# Initialize MediaPipe FaceMesh model
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Initialize webcam
cam = cv2.VideoCapture(0)

# Initialize previous position of the eye marker
prev_eye_x = None
prev_eye_y = None
while True:
    ret, frame = cam.read()
    if not ret:
        continue

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with FaceMesh model
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    # Check if landmarks are detected
    if landmark_points:
        for face_landmarks in landmark_points:
            # Draw square around the face
            x_min, y_min, x_max, y_max = frame_w, frame_h, 0, 0
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * frame_w), int(landmark.y * frame_h)
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw different design for eye landmarks
            for landmark_id in range(468, 478):  # Landmark IDs for both eyes
                landmark = face_landmarks.landmark[landmark_id]
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.drawMarker(frame, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

                # Track eye movement (right or left)
                if landmark_id == 468:  # Consider only one eye for simplicity (adjust as needed)
                    if prev_eye_x is not None:
                        direction_text  = "None"
                        if x > prev_eye_x:
                            direction_text = "right la baghtoy tu lavdya"
                        elif x < prev_eye_x:
                            direction_text = "left la baghtoy tu lavdya"
                        cv2.putText(frame, direction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                    prev_eye_x = x

                if landmark_id == 468:
                    # Consider only one eye for simplicity (adjust as needed)
                    if prev_eye_y is not None:
                        direction_text_y = "None"
                        if y < prev_eye_y:
                            direction_text_y = "Eye is moving up"
                        elif y > prev_eye_y:
                            direction_text_y = "Eye is moving down"
                        cv2.putText(frame, direction_text_y, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                    prev_eye_y = y
    # Display annotated frame
    cv2.imshow('Face and Eye Detection', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close OpenCV windows
cam.release()
cv2.destroyAllWindows()
