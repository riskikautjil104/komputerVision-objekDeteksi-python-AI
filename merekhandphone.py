import cv2
import numpy as np

# Daftar merek handphone yang akan dideteksi
brands = ["iPhone", "samsung"]

# Inisialisasi algoritma ekstraksi fitur ORB
orb = cv2.ORB_create()

# Membaca gambar referensi untuk masing-masing merek handphone
reference_images = []
for brand in brands:
    image = cv2.imread(f"reference/{brand}.jpg", cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    reference_images.append((brand, keypoints, descriptors))

# Menginisialisasi detektor fitur dengan algoritma FLANN (Fast Library for Approximate Nearest Neighbors)
flann = cv2.FlannBasedMatcher()

# Membaca gambar input
input_image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
input_keypoints, input_descriptors = orb.detectAndCompute(input_image, None)

# Melakukan pencocokan fitur dengan masing-masing gambar referensi
matches = []
for brand, keypoints, descriptors in reference_images:
    if descriptors is not None:
        result = flann.knnMatch(input_descriptors, descriptors, k=2)
        good_matches = []
        for m, n in result:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        matches.append((brand, len(good_matches)))

# Mengurutkan hasil pencocokan berdasarkan jumlah kecocokan yang paling tinggi
matches.sort(key=lambda x: x[1], reverse=True)

# Menampilkan hasil deteksi merek handphone
if len(matches) > 0:
    brand = matches[0][0]
    confidence = matches[0][1]
    print("Merek handphone terdeteksi:", brand)
    print("Kepercayaan:", confidence)
else:
    print("Merek handphone tidak terdeteksi.")

# Menampilkan gambar dengan bounding box dan label merek handphone
image_with_matches = cv2.imread("input.jpg")
cv2.putText(image_with_matches, f"{brand} ({confidence})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("Deteksi Merek Handphone", image_with_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
