import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

# Inisialisasi Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    success, frame = cap.read()
    if not success:
        break

    # Ubah citra ke dalam ruang warna RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Proses frame dengan Hands
    results = hands.process(frame_rgb)

    # Lakukan sesuatu dengan hasil pelacakan tangan
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Lakukan sesuatu dengan landmarks tangan
            # Contoh: gambar titik-titik landmarks tangan pada citra
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Tampilkan frame
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
