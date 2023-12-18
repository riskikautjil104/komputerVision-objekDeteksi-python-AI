import cv2
import numpy as np
import random

# Inisialisasi window OpenCV
cv2.namedWindow("Game")

# Inisialisasi variabel permainan
score = 0
game_over = False

# Mengatur ukuran window
screen_width, screen_height = 800, 600
cv2.resizeWindow("Game", screen_width, screen_height)

# Mengatur area untuk mendeteksi mulut
mouth_roi = (300, 400, 200, 100)

# Memuat Cascade Classifier untuk mendeteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Fungsi untuk menggambar objek (buah atau racun)
def draw_object(obj_img, obj_position):
    obj_height, obj_width, _ = obj_img.shape
    x, y = obj_position
    game_screen[y:y+obj_height, x:x+obj_width] = obj_img

# Fungsi untuk memeriksa tabrakan antara mulut dan objek
def check_collision(mouth_position, obj_position):
    mouth_x, mouth_y, mouth_w, mouth_h = mouth_position
    obj_x, obj_y, obj_w, obj_h = obj_position

    if mouth_x < obj_x + obj_w and mouth_x + mouth_w > obj_x and mouth_y < obj_y + obj_h and mouth_y + mouth_h > obj_y:
        return True
    else:
        return False

# Memuat gambar buah dan racun
fruit_img = cv2.imread("fruit.jpg")
poison_img = cv2.imread("poison.jpg")

# Membuat tampilan game screen
game_screen = np.zeros((screen_height, screen_width, 3), np.uint8)

# Perulangan utama game
while True:
    # Membaca frame dari webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    # Mengekstrak area mulut
    mouth_frame = frame[mouth_roi[1]:mouth_roi[1]+mouth_roi[3], mouth_roi[0]:mouth_roi[0]+mouth_roi[2]]

    # Mengubah frame menjadi grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mendeteksi wajah dalam frame menggunakan Cascade Classifier
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    # Menggambar kotak di sekitar wajah
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Mengekstrak area mulut
        mouth_frame = frame[y+mouth_roi[1]:y+mouth_roi[1]+mouth_roi[3], x+mouth_roi[0]:x+mouth_roi[0]+mouth_roi[2]]

        # Menggambar kotak di sekitar mulut
        cv2.rectangle(roi_color, (mouth_roi[0], mouth_roi[1]), (mouth_roi[0]+mouth_roi[2], mouth_roi[1]+mouth_roi[3]), (0, 255, 0), 2)

        # Deteksi objek (buah atau racun) yang jatuh
        if not game_over:
            if random.randint(0, 100) < 3:
                object_position = (random.randint(0, screen_width-50), 0)
                object_img = fruit_img if random.randint(0, 100) < 90 else poison_img

        # Menggambar objek
        draw_object(object_img, object_position)

        # Memeriksa tabrakan dengan mulut
        if check_collision(mouth_roi, object_position):
            if object_img is fruit_img:
                score += 1
            else:
                game_over = True

    # Menampilkan poin di layar
    cv2.putText(frame, "Score: " + str(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Menampilkan frame ke window
    cv2.imshow("Game", frame)

    # Menghentikan permainan jika tombol "q" ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Membersihkan
cap.release()
cv2.destroyAllWindows()
