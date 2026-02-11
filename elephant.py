import cv2
from ultralytics import YOLO
import smtplib
from email.message import EmailMessage
import mimetypes
from twilio.rest import Client
import serial
from flask import Flask, request
import threading
from pyngrok import ngrok
import os

# ==============================
# 🔹 Email Credentials
# ==============================
PASSWORD = "lwbe rwaj drrl gdjk"
SENDER = "sschaporkar_ar@jspmrscoe.edu.in"
RECEIVER = "saumitrachaporkar@gmail.com"

# ==============================
# 🔹 Twilio Credentials
# ==============================

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")

twilio_number = "+12283009317"   # Twilio phone number
my_number = "+918999612084"      # Officer phone number

client = Client(account_sid, auth_token)

# ==============================
# 🔹 Arduino Setup
# ==============================
arduino = serial.Serial("COM6", 9600, timeout=1)

# ==============================
# 🔹 Flask App for Confirmation
# ==============================
app = Flask(__name__)
alert_confirmed = False

@app.route("/confirm")
def confirm_alert():
    global alert_confirmed
    alert_confirmed = True
    # SMS + Buzzer
    send_sms_alert()
    arduino.write(b'1')
    return "✅ Elephant confirmed. SMS sent & buzzer activated!"

def run_flask():
    app.run(port=5000)

# ==============================
# 🔹 Email Function
# ==============================
def send_email(image_path, public_url):
    email = EmailMessage()
    email["From"] = SENDER
    email["To"] = RECEIVER
    email["Subject"] = "Elephant Alert 🚨"
    email.set_content(f"An elephant has been detected!\n\nConfirm: {public_url}/confirm")

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
        body="🚨 Elephant Detected! Confirmed by officer.",
        from_=twilio_number,
        to=my_number
    )
    print("SMS sent! SID:", message.sid)

# ==============================
# 🔹 YOLO Detection
# ==============================
def start_detection(public_url):
    global alert_confirmed
    model = YOLO("yolov8s.pt")

    video_path = r"C:\Users\saumi\Downloads\hi.mp4"
    cap = cv2.VideoCapture(video_path)

    alert_sent = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
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
                    send_email(image_path, public_url)   # Send email with ngrok link
                    alert_sent = True
                    print("Email sent! Waiting for officer confirmation...")

        print(elephant_detected)

        if elephant_detected == 0:
            alert_sent = False
            alert_confirmed = False

        cv2.imshow("Elephant Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==============================
# 🔹 MAIN
# ==============================
if __name__ == "__main__":
    # Start Flask server in background
    threading.Thread(target=run_flask, daemon=True).start()

    # Setup ngrok tunnel
    ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")   # Replace with your ngrok token
    public_url = ngrok.connect(5000)
    print("🌍 Public URL:", public_url)

    # Start detection loop
    start_detection(public_url)
