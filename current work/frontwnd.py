import tkinter as tk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
from main import initialize_webcam, process_frame
from ttkbootstrap import Style, Button, Progressbar
import psutil

# Initialize global variables
running = False
earRatio = 0.25
blinkFrm = 3
blinkCount = 0
blinkTotal = 0
taskBLinkCount = 2
prevX = None
prevY = None

def start_action():
    global running
    running = True

def stop_action():
    global running
    running = False

def update_webcam():
    global running, blinkCount, blinkTotal, prevX, prevY
    if running:
        ret, frame = cam.read()
        if ret:
            frame, blinkCount, blinkTotal, prevX, prevY = process_frame(
                frame, face_mesh, earRatio, blinkFrm, blinkCount, blinkTotal, taskBLinkCount, prevX, prevY
            )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_rgb)
            frame_photo = ImageTk.PhotoImage(frame_image)
            webcam_label.configure(image=frame_photo)
            webcam_label.image = frame_photo

            blink_label.config(text=f"Blinks: {blinkTotal}")

    root.after(100, update_webcam)

def update_progress():
    cpu_usage = psutil.cpu_percent(interval=1)
    progress_bar["value"] = cpu_usage
    cpu_label.config(text=f"CPU Usage: {cpu_usage}%")
    root.after(1000, update_progress)  # Update every second

# Initialize Tkinter
root = tk.Tk()
root.title("IRIS")
root.geometry("1200x800")

# Apply the ttkbootstrap theme
style = Style(theme='darkly')

# Create and place webcam frame with border
webcam_frame = tk.Frame(root, bg="#3d646b", bd=5, relief='solid')
webcam_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)

# Add a label for the webcam feed inside the bordered frame
webcam_label = tk.Label(webcam_frame)
webcam_label.pack(fill='both', expand=True)

# Create and place control buttons
control_frame = tk.Frame(root, bg="#3d6466")
control_frame.pack(side='right', fill='y', padx=10, pady=10)

# Start button
start_button = Button(
    control_frame,
    text="Start",
    command=start_action,
    bootstyle="success",
    padding=10
)
start_button.pack(pady=10, fill='x')

# Stop button
stop_button = Button(
    control_frame,
    text="Stop",
    command=stop_action,
    bootstyle="danger",
    padding=10
)
stop_button.pack(pady=10, fill='x')

# Blink count label
blink_label = tk.Label(control_frame, text="Blinks: 0", bg="#3d646b", fg="white", font=('Helvetica', 12))
blink_label.pack(pady=10)

# CPU usage progress bar
progress_bar = Progressbar(
    control_frame,
    bootstyle="success",
    orient="horizontal",
    length=300,
    mode="determinate"
)
progress_bar.pack(pady=10, fill='x')

# CPU usage label
cpu_label = tk.Label(control_frame, text="CPU Usage: 0%", bg="#3d646b", fg="white", font=('Helvetica', 12))
cpu_label.pack(pady=10)

# Initialize face mesh and webcam
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
cam = initialize_webcam()

# Start updating the webcam feed and CPU progress
update_webcam()
update_progress()

# Start Tkinter event loop
root.mainloop()

# Release the camera when done
cam.release()
cv2.destroyAllWindows()


