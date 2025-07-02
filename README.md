# 🛡️ DrowsyGuard: Sistem Deteksi Kantuk & HP untuk Pengemudi 🚗📱

**DrowsyGuard** adalah aplikasi berbasis Python dan Flask yang mendeteksi **kantuk (mata tertutup / menguap)** dan **penggunaan ponsel saat berkendara** menggunakan **MediaPipe** dan **YOLOv8**, dilengkapi sistem peringatan audio otomatis dan logging CSV.

## 📸 Fitur Utama

✅ Deteksi kantuk berdasarkan mata tertutup (Eye Aspect Ratio)  
✅ Deteksi menguap berdasarkan jarak bibir (mouth open)  
✅ Deteksi penggunaan HP menggunakan YOLOv8  
✅ Peringatan suara otomatis (Warning + Lagu Dangdut jika sering lalai)  
✅ Streaming kamera langsung ke browser (Flask + OpenCV)  
✅ Logging semua peringatan ke file `log_deteksi.csv`  


---

## 🧠 Teknologi & Algoritma

| Komponen | Teknologi / Algoritma |
|---------|------------------------|
| Antarmuka Web | Flask |
| Streaming Kamera | OpenCV |
| Deteksi Kantuk & Menguap | MediaPipe Face Mesh + Euclidean Distance |
| Deteksi HP | YOLOv8 (Ultralytics) |
| Audio Peringatan | pygame.mixer |
| Logging | CSV writer |
| Multi-threading | Python threading untuk pemutaran audio |

---

## 📦 Library yang Digunakan

- `Flask`
- `opencv-python`
- `mediapipe`
- `pygame`
- `numpy`
- `ultralytics` *(YOLOv8)*
- `threading`, `csv`, `time`

Install semua dependencies:

```bash
pip install flask opencv-python mediapipe pygame numpy ultralytics


venv310\Scripts\activate