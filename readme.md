# Embedded Machine Learning – WS 2025/2026

This repository contains my assignments and project work for the course **Embedded Machine Learning** taught by **Prof. Dr. Holger Fröning** at Heidelberg University.

---

## 📘 Course Overview

The course focuses on implementing machine learning algorithms on resource-constrained embedded hardware systems. It covers deep learning fundamentals, GPU/FPGA processors, acceleration techniques, safe/unsafe optimizations, quantization, pruning, and probabilistic modeling.

---

## 🗓 Course Schedule

| Date | Lecture | Exercise |
|------|---------|----------|
| 16.10.2025 | Introduction, ML basics, ML processors (GPU Intro) | Generalization of simple regression |
| 23.10.2025 | Neural Networks from scratch, CONVs | – |
| 30.10.2025 | *Traveling (No Lecture)* | PyTorch Basics |
| 06.11.2025 | Automatic Differentiation & Optimization | MNIST/CIFAR-10 Training I |
| 13.11.2025 | Regularization (Dropout, L1/L2, BN, Data Aug.) | MNIST/CIFAR-10 Training II |
| 20.11.2025 | *Traveling (No Lecture)* | – |
| 27.11.2025 | Neural Architecture Design (ResNets, Pooling, …) | MNIST/CIFAR-10 Training III |
| 04.12.2025 | Unsafe Optimizations – Basics | Quantization & Pruning I |
| 11.12.2025 | Unsafe Optimizations – Advanced (incl. NAS) | Quantization & Pruning II |
| 18.12.2025 | Safe Optimizations & Array Processors | – |
| 25.12.2025 | **XMAS Break** | – |
| 01.01.2026 | **XMAS Break** | – |
| 08.01.2026 | Review of ML Processors | – |
| 15.01.2026 | Advanced Neural Architectures | Quantization – TTQ |
| 22.01.2026 | Probabilistic Modeling & Scaling | Exam Examples |
| 29.01.2026 | **Final Exam** | – |

---

## ⚙️ Required Software

- **Python = 3.11.14**
- **Numpy = 1.25.2**
- **PyTorch = 2.7.1**
- **torchvision = 0.23.0**
- **CUDA = TBD**

## 🚀 Environment Setup

This project uses **Conda** to manage the development environment for better reproducibility and compatibility.

Make sure you have **Anaconda** or **Miniconda** installed. You can use this command to check:

```bash
conda -V
```


Then run:

```bash
conda env create -f environment.yml

conda activate embedded-ml
```

---

## ✅ Assignment Progress

| Week | Task                                      | Status        |
|------|-------------------------------------------|---------------|
| 01   | Generalization of Simple Regression       | ✅ Done       |
| 02   | PyTorch Basics                            | 🔄 In Progresse       |
| 03   | MNIST/CIFAR-10 Training I                 | ⏳|
| 04   | MNIST/CIFAR-10 Training II                | ⏳            |
| 05   | Architecture Design + Training III        | ⏳            |
| 06   | Quantization & Pruning I                  | ⏳            |
| 07   | Quantization & Pruning II                 | ⏳            |
| 08   | Advanced Neural Architectures             | ⏳            |
| 09   | Final Project / Exam Preparation          | ⏳            |

---

## 📜 License

This project is released under the **MIT License**.  
Feel free to use, modify and contribute.
