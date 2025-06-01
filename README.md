## Workflow of the Apple Ripeness Classification Script

### 🧰 Preparations

1. **Python Environment**

   * Ensure you have **Python 3.x** installed on your system.

2. **Required Python Packages**
   Install the following dependencies via pip:

   ```bash
   pip install opencv-python numpy scikit-learn matplotlib
   ```

   * **cv2 (OpenCV)**: for image I/O and color conversion
   * **numpy**: array operations
   * **scikit-learn**: KMeans clustering for segmentation
   * **matplotlib**: to display segmented images
   * **tkinter**: usually comes pre-installed with Python; used here to create a simple GUI

3. **Folder Structure**

   * Place this script in a working folder.
   * If you plan to use `process_images_from_folder`, organize your dataset as:

     ```
     dataset_folder/
       ├─ subfolder1/       # e.g., “ripe/”, “unripe/”
       │    ├─ imageA.jpg
       │    ├─ imageB.png
       │    └─ …
       ├─ subfolder2/
       │    └─ …
       └─ …
     ```
   * Otherwise, you only need individual image files to test via the GUI.

---

### 🚦 Script Overview

```python
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
    red_region = np.mean(image[:, :, 0])   # Rata-rata warna merah
    green_region = np.mean(image[:, :, 1]) # Rata-rata warna hijau

    # Logika berdasarkan warna
    if red_region > green_region:
        return '80%-100%'  # Lebih merah = lebih matang
    else:
        return '20%-60%'   # Lebih hijau = kurang matang

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
```

#### 1. read\_image(image\_path)

* **Input**: Path to an image file (e.g., `"Foto/1.jpg"`).
* **Steps**:

  1. `cv2.imread()` reads the image in BGR color space.
  2. Convert to RGB with `cv2.cvtColor` so colors display correctly in matplotlib.
* **Output**: A NumPy array of shape `(height, width, 3)` in RGB.

#### 2. segment\_image(image)

* **Input**: An RGB image array.
* **Steps**:

  1. Reshape `image` from `(H, W, 3)` to `(-1, 3)` for KMeans.
  2. Run `KMeans(n_clusters=3)` to partition pixels into 3 clusters (three dominant colors).
  3. Use `cluster_centers_` and `labels_` to build `segmented_image` by replacing each pixel with its cluster center.
  4. Reshape back to `(H, W, 3)` and cast to `uint8`.
* **Output**: A “posterized” image where each pixel is replaced by its cluster center color.

#### 3. classify\_maturity(image)

* **Input**: A segmented RGB image.
* **Steps**:

  1. Compute `red_region = np.mean(image[:, :, 0])`: average of the red channel.
  2. Compute `green_region = np.mean(image[:, :, 1])`: average of the green channel.
  3. If `red_region > green_region`, return `"80%-100%"` (ripe). Otherwise, return `"20%-60%"` (less ripe).
* **Output**: A string indicating ripeness level based on dominant color.

#### 4. process\_images\_from\_folder(folder\_path)

* **Input**: A path to a root folder containing subfolders of images.
* **Steps**:

  1. Iterate over each subfolder in `folder_path`.
  2. For every `.jpg` or `.png` file inside:

     * Read the image via `read_image`.
     * Segment it using `segment_image`.
     * Classify ripeness via `classify_maturity`.
     * Append the segmented image and its maturity label to lists.
* **Output**: `(images, maturity_levels)` lists, where `images[i]` is the segmented image from file `i` and `maturity_levels[i]` is its label.

> **Note**: This function is provided for batch processing, but is not directly invoked by the GUI.

#### 5. create\_gui()

* **Behavior**:

  1. Creates a simple Tkinter window titled **“Klasifikasi Kematangan Apel”**.
  2. Displays a button labeled **“Buka Gambar”**.
  3. When clicked, a file dialog opens, allowing the user to select an image (`*.jpg` or `*.png`).
  4. For the chosen image:

     * `read_image` → `segment_image` → `classify_maturity`.
     * Uses `matplotlib.pyplot` to show the segmented image, with a title that reads, for example, **“Kematangan: 80%-100%”**.
  5. The GUI mainloop keeps running until the window is closed.

---

### 🏁 How to Run

1. **Save the Script**
   Place the code above into a file, for example, `apple_maturity.py`, in your working directory.

2. **Ensure Images Are Accessible**

   * If you plan to test via the GUI, make sure you have one or more `.jpg` or `.png` images of apples somewhere on your computer.
   * If using `process_images_from_folder`, point to a folder organized as described under Prerequisites.

3. **Execute the Script**

   ```bash
   python apple_maturity.py
   ```

   * A small window titled **“Klasifikasi Kematangan Apel”** will open.
   * Click **“Buka Gambar”**, select an apple image, and a matplotlib window will pop up showing the segmented output with its ripeness label.
