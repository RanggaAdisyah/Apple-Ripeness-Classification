import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt

# Fungsi untuk membaca gambar
def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Fungsi untuk segmentasi gambar berdasarkan warna (similarity)
def segment_image(image):
    # Ubah citra menjadi data 2D untuk KMeans
    image_reshaped = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(image_reshaped)
    labels = kmeans.labels_
    segmented_image = kmeans.cluster_centers_[labels].reshape(image.shape).astype(np.uint8)
    return segmented_image

# Fungsi untuk mengklasifikasikan kematangan berdasarkan warna dominan
def classify_maturity(image):
    # Menyaring warna merah dan hijau
    red_region = np.mean(image[:, :, 0])  # Rata-rata warna merah
    green_region = np.mean(image[:, :, 1])  # Rata-rata warna hijau

    # Logika berdasarkan warna
    if red_region > green_region:
        return '80%-100%'  # Lebih merah = lebih matang
    else:
        return '20%-60%'  # Lebih hijau = kurang matang

# Fungsi untuk memproses citra dalam folder dataset
def process_images_from_folder(folder_path):
    maturity_levels = []
    images = []
    
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.jpg') or file_name.endswith('.png'):
                    image_path = os.path.join(subfolder_path, file_name)
                    image = read_image(image_path)
                    segmented_image = segment_image(image)
                    maturity = classify_maturity(segmented_image)
                    images.append(segmented_image)
                    maturity_levels.append(maturity)
    
    return images, maturity_levels

# GUI untuk menampilkan hasil
def create_gui():
    root = tk.Tk()
    root.title("Klasifikasi Kematangan Apel")

    # Fungsi untuk membuka file dan memproses gambar
    def open_image():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg")])
        if file_path:
            image = read_image(file_path)
            segmented_image = segment_image(image)
            maturity = classify_maturity(segmented_image)
            plt.imshow(segmented_image)
            plt.title(f"Kematangan: {maturity}")
            plt.show()

    open_button = tk.Button(root, text="Buka Gambar", command=open_image)
    open_button.pack(pady=20)

    root.mainloop()

# Menjalankan GUI
create_gui()

plt.show()