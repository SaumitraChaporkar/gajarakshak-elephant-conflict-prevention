import cv2
from ultralytics import YOLO
import smtplib
from email.message import EmailMessage
import mimetypes
from twilio.rest import Client
import serial
import time
import torch
import os

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# ==============================
# 🔹 Email Credentials
# ==============================
PASSWORD = "jyfu onmo qzfe csuj"   # Gmail App Password
SENDER = "sschaporkar_ar@jspmrscoe.edu.in"
RECEIVER = "saumitrachaporkar@gmail.com"

# ==============================
# 🔹 Twilio Credentials
# ==============================
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_number = "+12283009317"   # Twilio number
my_number = "+918999612084"     # Your mobile number

client = Client(account_sid, auth_token)

# ==============================
# 🔹 Arduino Setup
# ==============================
arduino = serial.Serial("COM6", 9600, timeout=1)  # Change COM port if needed
time.sleep(2)  # wait for Arduino to reset

# ==============================
# 🔹 Email Function
# ==============================
def send_email(image_path):
    email = EmailMessage()
    email["From"] = SENDER
    email["To"] = RECEIVER
    email["Subject"] = "Elephant Alert 🚨"
    email.set_content("An elephant has been detected!")

    with open(image_path, "rb") as file:
        file_data = file.read()
        file_type, _ = mimetypes.guess_type(image_path)

    if file_type is None:
        file_type = "image/png"

    maintype, subtype = file_type.split("/")
    email.add_attachment(file_data, maintype=maintype, subtype=subtype, filename="elephant.png")

    with smtplib.SMTP("smtp.gmail.com", 587) as gmail:
        gmail.starttls()
        gmail.login(SENDER, PASSWORD)
        gmail.send_message(email)

# ==============================
# 🔹 SMS Function
# ==============================
def send_sms_alert():
    message = client.messages.create(
        body="🚨 Elephant Detected! Please stay alert.",
        from_=twilio_number,
        to=my_number
    )
    print("SMS sent! SID:", message.sid)

# ==============================
# 🔹 YOLO Detection
# ==============================
model = YOLO("yolov8s.pt")

video_path = r"C:\Users\saumi\Downloads\elephant.mp4"  # safe path format
cap = cv2.VideoCapture(video_path)

alert_sent = False

while cap.isOpened():
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    dl_start = time.time()
    results = model(frame, device=device)
    dl_end = time.time()


    elephant_detected = 0

    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        class_name = model.names[cls_id]

        if class_name == "elephant" and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            elephant_detected = 1
            image_path = "detected_elephant.png"
            cv2.imwrite(image_path, frame)

            if not alert_sent:
                send_email(image_path)   # Email Alert
                send_sms_alert()         # SMS Alert
                arduino.write(b"1")      # Activate buzzer
                alert_sent = True
        print("Email + SMS + Buzzer Alert Sent!")

    print("Elephant:", elephant_detected)

    if elephant_detected == 0:
        alert_sent = False
        arduino.write(b'0')  # Tell Arduino to stop buzzer

    cv2.imshow("Elephant Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    end = time.time()
    print("Time required for 1 frame = ", end-start)
    print("Time required for 1 frame : DL model = ", dl_end-dl_start)

cap.release()
cv2.destroyAllWindows()
