# MNIST Handwritten Digit Classification using a Convolutional Neural Network (CNN)

A lightweight, robust PyTorch implementation of a Convolutional Neural Network (CNN) trained and evaluated on the classic **MNIST** dataset, achieving an impressive test accuracy of over **99.2%**. 

Designed as an educational project, this repository walks through end-to-end deep learning workflows: from dataset loading and tensor transformations to model architecture design, training loops, backpropagation, and evaluation metrics.

---

## 🚀 Model Architecture

The CNN features a sequential feature extraction phase followed by a fully connected classification head:

1. **Feature Extraction (Convolutional Layers):**
   - `Conv2d(1, 32, kernel_size=3, padding=1)` + `ReLU` + `MaxPool2d(2, 2)`
   - `Conv2d(32, 64, kernel_size=3, padding=1)` + `ReLU` + `MaxPool2d(2, 2)`
   - `Conv2d(64, 64, kernel_size=3, padding=1)` + `ReLU` + `MaxPool2d(2, 2)`
2. **Classification (Fully Connected Layers):**
   - Flatten layer (`64 * 3 * 3` features)
   - `Linear(576, 256)` + `ReLU`
   - `Linear(256, 128)` + `ReLU`
   - `Linear(128, 10)` (Output layer for digits 0–9)

---

## 📊 Dataset Overview
- **Dataset:** MNIST (Modified National Institute of Standards and Technology database)
- **Training Set:** 60,000 grayscale images ($28 \times 28$ pixels)
- **Test Set:** 10,000 unseen grayscale images
- **Preprocessing:** Pixel values scaled to range $[0.0, 1.0]$ via PyTorch tensors.

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/mnist-cnn-pytorch.git
   cd mnist-cnn-pytorch
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃‍♂️ Usage

You can open and run the Jupyter notebook (`MNIST_CNN_Delta_(1).ipynb`) locally using Jupyter Lab/Notebook or upload it directly to Google Colab.

```bash
jupyter notebook MNIST_CNN_Delta_(1).ipynb
```

### Training Hyperparameters:
- **Optimizer:** Adam ($\text{lr} = 0.001$)
- **Loss Function:** Cross-Entropy Loss (`nn.CrossEntropyLoss`)
- **Batch Size:** 64
- **Epochs:** 10

---

## 📈 Results

After 10 epochs of training on the training split, the model yields:
- **Training Accuracy:** $\approx 99.57\%$
- **Test Accuracy:** $\approx 99.26\%$