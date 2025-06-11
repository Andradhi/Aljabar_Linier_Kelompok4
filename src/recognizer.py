import numpy as np
import os
from PIL import Image, ImageOps
from eigen import get_top_k_eigenvectors, euclidean_distance

# Konversi dan preprocessing gambar ke vektor
def image_to_vector(img_path, size=(100, 100)):
    img = Image.open(img_path).convert('L')  # ubah ke grayscale
    img = ImageOps.equalize(img)             # equalize histogram
    img = img.resize(size)                   # resize agar seragam
    return np.asarray(img, dtype='float64').flatten()

# Load seluruh dataset dari folder
def load_dataset(folder):
    vectors = []
    filenames = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        try:
            vec = image_to_vector(path)
            vectors.append(vec)
            filenames.append(filename)
        except Exception as e:
            print(f"[WARNING] Gagal baca {filename}: {e}")
    return np.array(vectors).T, filenames

# Training eigenface: PCA, eigenvector, mean
def train(dataset_path):
    X, filenames = load_dataset(dataset_path)
    mean_face = np.mean(X, axis=1).reshape(-1, 1)
    A = X - mean_face
    C = np.dot(A.T, A)
    _, eigenvectors_small = get_top_k_eigenvectors(C, k=30)  # gunakan lebih banyak eigenface
    eigenfaces = np.dot(A, eigenvectors_small.T)
    projections = np.dot(eigenfaces.T, A)
    return eigenfaces, projections, mean_face, filenames

# Pencocokan test image terhadap dataset
def recognize(input_path, eigenfaces, projections, mean_face, filenames, threshold=1e8):
    try:
        test_vector = image_to_vector(input_path).reshape(-1, 1)
        A_test = test_vector - mean_face
        test_proj = np.dot(eigenfaces.T, A_test)

        distances = []
        for p in projections.T:
            dist = euclidean_distance(test_proj.flatten(), p.flatten())
            distances.append(dist)

        min_dist = min(distances)
        best_match = np.argmin(distances)

        print(f"[DEBUG] Min distance: {min_dist:.2f}, Match file: {filenames[best_match]}")

        if min_dist > threshold:
            return None, min_dist

        return filenames[best_match], min_dist

    except Exception as e:
        print(f"[ERROR] Gagal mengenali wajah: {e}")
        return None, None
