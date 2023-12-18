import cv2
import numpy as np
import mediapipe as mp

# Inisialisasi variabel global
canvas = np.ones([480, 640, 3], dtype=np.uint8) * 255  # Membuat kanvas putih
radius = 10  # Ukuran radius untuk menggambar
color = (0, 0, 255)  # Warna awal untuk menggambar (dalam format BGR)
draw_flag = False  # Menandakan status saat ini, apakah sedang menggambar atau tidak

# Inisialisasi detektor tangan
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Fungsi untuk menggambar pada kanvas
def draw(event, x, y, flags, param):
    global canvas, color, draw_flag

    if event == cv2.EVENT_LBUTTONDOWN:
        draw_flag = True
        cv2.circle(canvas, (x, y), radius, color, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        draw_flag = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if draw_flag:
            cv2.circle(canvas, (x, y), radius, color, -1)

# Membuat window OpenCV
cv2.namedWindow('Virtual Paint')
cv2.setMouseCallback('Virtual Paint', draw)

# Loop utama
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()

    if not success:
        break

    # Konversi citra menjadi BGR ke RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Deteksi tangan pada citra
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Mendapatkan koordinat landmark tangan
            for id, landmark in enumerate(hand_landmarks.landmark):
                height, width, _ = image.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)

                # Menggambar garis-garis hijau dan biru pada jari-jari
                if id in [4, 8, 12, 16, 20]:  # Jari-jari yang ingin ditandai
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),
                                                                                              thickness=2),
                                              connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0),
                                                                                               thickness=2))

                # Mengambil warna dari posisi jari telunjuk
                if id == 8:  # Jari telunjuk
                    color = tuple(map(int, image[cy, cx]))

    # Menampilkan kanvas dan citra dari kamera
    cv2.imshow('Virtual Paint', canvas)
    cv2.imshow('Camera', image)

    # Tombol 'c' untuk menghapus kanvas
    if cv2.waitKey(1) & 0xFF == ord('c'):
        canvas = np.ones([480, 640, 3], dtype=np.uint8) * 255

    # Tombol 'q' untuk keluar dari program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Membersihkan dan menutup windows OpenCV
cap.release()
cv2.destroyAllWindows()
