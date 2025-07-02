# app.py
from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import pygame
import time
import threading
import numpy as np
import csv
from ultralytics import YOLO

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

yolo_model = YOLO("yolov8n.pt")

pygame.mixer.init()
warning_count = 0

# === Logging ===
def log_event(message):
    with open("log_deteksi.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), message])

# === Audio ===
def play_warning():
    try:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load("static/Warning.mp3")
            pygame.mixer.music.play()
    except Exception as e:
        print(f"Error play_warning: {e}")

def play_dangdut():
    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.load("static/Dangdut.mp3")
        pygame.mixer.music.play()
        print("Lagu dangdut diputar!")
    except Exception as e:
        print(f"Error play_dangdut: {e}")

# === Utils ===
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# === Kamera Stream ===
def generate_frames():
    global warning_count
    global eye_closed_start

    cap = cv2.VideoCapture(0)
    last_warn_time = 0
    yawn_start_time = None
    hp_detected_start = None
    eye_closed_start = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)
        h, w, _ = frame.shape

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                lm = face_landmarks.landmark

                # === Mata ===
                left_eye = [lm[159], lm[145]]
                right_eye = [lm[386], lm[374]]

                left_ear = euclidean((left_eye[0].x * w, left_eye[0].y * h), (left_eye[1].x * w, left_eye[1].y * h))
                right_ear = euclidean((right_eye[0].x * w, right_eye[0].y * h), (right_eye[1].x * w, right_eye[1].y * h))
                ear = (left_ear + right_ear) / 2

                # === Mulut (menguap) ===
                top_lip = lm[13]
                bottom_lip = lm[14]
                mouth_open = euclidean((top_lip.x * w, top_lip.y * h), (bottom_lip.x * w, bottom_lip.y * h))

                # === Deteksi mata tertutup minimal 1 detik ===
                if ear < 12:
                    if eye_closed_start is None:
                        eye_closed_start = time.time()
                    elif time.time() - eye_closed_start > 1.0:
                        if time.time() - last_warn_time > 3:
                            threading.Thread(target=play_warning).start()
                            last_warn_time = time.time()
                            log_event("Mengantuk! (Mata)")
                            warning_count += 1
                            if warning_count >= 5:
                                threading.Thread(target=play_dangdut).start()
                                warning_count = 0
                            cv2.putText(frame, "Mengantuk! (Mata)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    eye_closed_start = None  # Reset

                # === Deteksi menguap selama >3 detik ===
                if mouth_open > 20:
                    if yawn_start_time is None:
                        yawn_start_time = time.time()
                    elif time.time() - yawn_start_time >= 3:
                        if time.time() - last_warn_time > 3:
                            threading.Thread(target=play_warning).start()
                            last_warn_time = time.time()
                            log_event("Mengantuk! (Menguap)")
                            warning_count += 1
                            if warning_count >= 5:
                                threading.Thread(target=play_dangdut).start()
                                warning_count = 0
                            cv2.putText(frame, "Mengantuk! (Menguap)", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    yawn_start_time = None

        # === Deteksi HP ===
        yolo_results = yolo_model.predict(frame, conf=0.5, verbose=False)[0]
        hp_found = False

        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            class_name = yolo_model.names[cls_id]
            if class_name.lower() in ['cell phone', 'mobile phone', 'phone']:
                hp_found = True
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 255), 2)
                cv2.putText(frame, "HP TERDETEKSI", (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if hp_found:
            if hp_detected_start is None:
                hp_detected_start = time.time()
            elif time.time() - hp_detected_start >= 2:
                if time.time() - last_warn_time > 3:
                    threading.Thread(target=play_warning).start()
                    last_warn_time = time.time()
                    log_event("Ayo Fokus Kejalan")
                    warning_count += 1
                    if warning_count >= 5:
                        threading.Thread(target=play_dangdut).start()
                        warning_count = 0
                    cv2.putText(frame, "Jangan main HP!", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
        else:
            hp_detected_start = None

        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, f"{timestamp}", (frame.shape[1]-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/log')
def log_data():
    logs = []
    try:
        with open("log_deteksi.csv", mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                logs.append({"timestamp": row[0], "peringatan": row[1]})
    except FileNotFoundError:
        pass
    return jsonify(logs)

@app.route('/reset')
def reset_data():
    global warning_count
    warning_count = 0

    log_path = "log_deteksi.csv"

    # Buat file log baru dengan header
    try:
        with open(log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "peringatan"])
    except Exception as e:
        print("Gagal menulis file log:", e)
        return jsonify({"status": "error", "message": "Gagal reset file log."}), 500

    # Coba stop audio
    try:
        pygame.mixer.music.stop()
    except Exception as e:
        print("Gagal stop audio:", e)
        # Tidak return error, hanya log

    return jsonify({"status": "success", "message": "Peringatan berhasil direset."})


if __name__ == '__main__':
    app.run(debug=True)
