import numpy as np
import matplotlib.pyplot as plt
import cv2

# Membaca gambar
image = cv2.imread('foto disini')

if image is None:
    print("Gambar tidak ditemukan. Pastikan jalur gambar sudah benar.")
else:
    # Mengubah warna menjadi RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title("Gambar Asli")
    plt.show()

    # Membentuk ulang gambar menjadi susunan piksel 2D dan 3 nilai warna (RGB)
    pixel_vals = image.reshape((-1,3))
    # Mengonversikan ke tipe float
    pixel_vals = np.float32(pixel_vals)

    # Menentukan kriteria k-means
    # Baris kode di bawah ini menentukan kriteria agar algoritme berhenti berjalan,
    # yang akan terjadi adalah 100 iterasi dijalankan atau epsilon (yang merupakan
    # akurasi yang dibutuhkan) menjadi 85%
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    # Lakukan k-means clustering dengan jumlah cluster yang ditetapkan sebagai 3
    # juga pusat acak pada awalnya dipilih untuk pengelompokan k-means
    k = 3
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Mengonversi data menjadi nilai 8-bit
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    # Membentuk ulang data menjadi dimensi gambar asli
    segmented_image = segmented_data.reshape((image.shape))
    plt.imshow(segmented_image)
    plt.title("Gambar Tersegmentasi")
    plt.show()
