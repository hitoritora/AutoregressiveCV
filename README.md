# Autoregressive Modeling and K-Means Clustering for MNIST and CIFAR-10

This project explores **autoregressive models** for **image generation** using the **MNIST** and **CIFAR-10** datasets, enhanced with **K-Means clustering** for pixel intensity quantization.  
It includes baseline models, variations for robustness, and comparisons between standard softmax and clustered inputs.

## Project Structure

| File | Description |
| :--- | :---------- |
| `kmeansMNIST.py` | Full pipeline for K-Means clustering on MNIST + training an autoregressive model. |
| `MNIST_Baseline_Softmax_Autoregressive.ipynb` | Baseline softmax autoregressive model for MNIST without clustering. |
| `prototype_autoregressive.ipynb` | Initial prototype for the autoregressive architecture. |
| `Softmax CIFAR-10_autoregressive.ipynb` | Baseline softmax autoregressive model for CIFAR-10. |
| `K-Means Softmax CIFAR-10_autoregressive.ipynb` | CIFAR-10 model using K-Means clustering to discretize pixel intensities. |
| `Fixed Flattening Softmax CIFAR-10_autoregressive.ipynb` | Variant that applies fixed flattening before CIFAR-10 autoregressive modeling. |
| `kmeans.ipynb` | Standalone K-Means clustering procedure notebook. |
| `MNIST_Softmax_Model.drawio` | Diagram showing the architecture of the softmax-based autoregressive model. |

---

## Setup Instructions

1. **Clone this repository** and navigate into it:
   ```bash
   git clone <repository_link>
   cd <repository_folder>
   ```

2. **Install dependencies**:
   You will need Python 3.8+ and install these packages:
   ```bash
   pip install torch torchvision scikit-learn matplotlib tqdm numpy
   ```

3. **(Optional) Enable GPU support**:
   If you have a CUDA-compatible GPU, ensure that PyTorch is installed with GPU support for faster training.

---

## How to Run
