import cv2
import numpy as np
import mediapipe as mp
import math
import subprocess

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi variabel global
volume = 0
vol_bar = 400
vol_per = 0

# Mendapatkan lebar dan tinggi layar
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Mendapatkan range deteksi tangan
minVol = 0
maxVol = 100

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Menggunakan MediaPipe untuk mendeteksi tangan
    results = hands.process(imgRGB)

    # Mendapatkan landmarks tangan
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Mendapatkan posisi titik ujung jari telunjuk dan jari jempol
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            # Menghitung jarak antara titik ujung jari telunjuk dan jari jempol
            distance = math.sqrt((index_finger.x - thumb.x)**2 + (index_finger.y - thumb.y)**2)

            # Menggunakan jarak untuk mengontrol volume
            volume = np.interp(distance, [0, 0.15], [minVol, maxVol])
            vol_bar = np.interp(distance, [0, 0.15], [400, 150])
            vol_per = np.interp(distance, [0, 0.15], [0, 100])

            # Mengubah volume menggunakan amixer pada Linux
            subprocess.call(["amixer", "-D", "pulse", "sset", "Master", str(int(volume))+"%"])

            # Menggambar garis dan indikator volume
            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(vol_per)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

            # Menggambar garis hijau pada setiap jari
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

    # Menampilkan gambar
    cv2.imshow("Volume Control", img)

    # Tombol 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
