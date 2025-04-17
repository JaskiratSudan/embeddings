# Contrastive Learning with Siamese Neural Network

This project implements a **Siamese Neural Network** using **contrastive learning** to measure similarity between handwritten digit images from the **MNIST dataset**. A lightweight **GUI** is included for testing similarity, classification, and visualizing learned embeddings.

<img width="912" alt="Screenshot 2025-04-17 at 19 48 27" src="https://github.com/user-attachments/assets/b998b3ef-b7e9-49fc-93c3-b22e2de8650a" />

---

## 🚀 Project Overview

- **Goal:** Learn a feature embedding space where similar images (same digit) are close and dissimilar images are far apart.
- **Approach:** Use a Siamese network with contrastive loss to train on image pairs.
- **Dataset:** [MNIST](http://yann.lecun.com/exdb/mnist/)
- **Framework:** TensorFlow / Keras
- **GUI:** Built with Tkinter + Plotly

---

## 🏗️ Architecture

- **Siamese Network:**
  - Two identical Dense sub-networks (shared weights)
  - Outputs 128-dimensional embedding vectors
- **Distance Metric:**
  - Euclidean distance between embeddings
- **Loss Function:**
  - Contrastive Loss:
    \[
    L = (1 - Y) \cdot \frac{1}{2}d^2 + Y \cdot \frac{1}{2}\max(0, m - d)^2
    \]
  - \(Y = 0\) for similar, \(Y = 1\) for dissimilar, margin \(m = 1\)

---

## 📊 Results

- **Training Accuracy:** ~99% on validation pairs
- **Embedding Visualization:** Distinct class clusters in 3D PCA space
- **Few-shot classification:** Achieved via nearest neighbor on embeddings

---

## 💻 GUI Features

- 🔄 **Load model** from disk
- 🖼️ **Classify digits** using embedding similarity
- 🔁 **Compare two images** to see similarity score
- 📈 **Visualize embeddings** in 3D (via PCA + Plotly)

---

## 🧪 Usage

### 1. Install dependencies

```bash
pip install tensorflow matplotlib numpy pillow scikit-learn plotly
```

### 2. Train the Model

```bash
python siamese_train.py
```

> Outputs: `siamese_model.h5` after training.

### 3. Run the GUI

```bash
python3 contrastive_learning_gui.py
```

> GUI window allows image input, pair comparison, and embedding visualization.

---

## 🧑‍💻 Author

**Jaskirat Singh Sudan**  
