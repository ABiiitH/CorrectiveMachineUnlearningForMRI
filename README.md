
# 🧠 Corrective Machine Unlearning for MRI Reconstruction

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.8+-green.svg)

> “AI that learns to forget — ensuring safety, trust, and adaptability in clinical diagnostics.”

## 🔬 Overview

This project explores the **corrective machine unlearning** paradigm to enhance the reliability and robustness of MRI reconstruction systems. Our methods target both privacy-preserving unlearning (e.g., “right to be forgotten”) and **corrective forgetting** of poisoned or faulty training data — without retraining from scratch.

Our pipeline includes:
- ⚙️ **End-to-End Variational Network (E2E VarNet)** for baseline MRI reconstruction
- 🧽 **Selective Synaptic Dampening (SSD)** as our primary unlearning strategy
- 🌀 **CycleGAN** for synthesizing MRI images with cysts from healthy images
- ⚔️ **Novel adversarial training objective** to perturb healthy MRIs and introduce synthetic tumor-like features for robustness and evaluation

---

## 🧩 Key Components

### 1. 🏗️ Corrective Unlearning

We implement several unlearning strategies:
- **Selective Synaptic Dampening (SSD)**: Leverages Fisher Information to selectively dampen parameters influenced by corrupted data
- **Bad Teacher Distillation**
- **Noisy Labeling**
- **Gradient Ascent-based Unlearning**

The unlearnt model learns to discard adversarial patterns while retaining useful medical features.

### 2. 🌀 CycleGAN for Pathological Translation

We train a CycleGAN to:
- Convert **healthy brain MRIs** to **cyst-injected versions**
- Provide a data augmentation path and simulate real-world pathological variation
- Evaluate whether reconstruction models hallucinate structures when faced with such transformations

### 3. ⚔️ Adversarial Objective for Tumor Induction

A novel adversarial objective generates **perturbed versions of healthy MRIs**:
- Introduces **tumor-like patterns** while preserving anatomical plausibility
- Acts as an **evaluation and stress-testing mechanism** for reconstruction models
- Assesses **robustness and hallucination resistance**

---

## 📊 Datasets

We utilize:
- **M4RAW** — High-resolution anatomical k-space data for training E2E VarNet
- **SKM-TEA** — Multi-contrast T2 mapping MRI data for cross-modal learning
- **CMRxRecon** — Cardiac MR dataset for generalization and benchmark testing
-- **BraTS** - Brain and Tumor Segmentation dataset.
-- **ExBox** - MRI Motion and metal object artefact dataset.
---

## 📈 Results

| Metric         | Original Model | SSD (Forget Set) | SSD (Retain Set) |
|----------------|----------------|------------------|------------------|
| **SSIM**       | 0.44           | 0.0967           | 0.3824           |
| **PSNR**       | 26.92 dB       | 13.97 dB         | 22.16 dB         |
| **NMSE**       | 0.0687         | 1.6210           | 0.1807           |

🔍 Qualitative results show that reconstructions from the unlearnt model **resemble clean data**, effectively reversing poisoning effects.

---

## 🧪 Getting Started

```bash
git clone https://github.com/<your-org>/CorrectiveUnlearningMRI.git
cd CorrectiveUnlearningMRI

# Setup environment
conda create -n unlearning_env python=3.8
conda activate unlearning_env
pip install -r requirements.txt
```

To train the E2E VarNet:
```bash
python train_varnet.py --dataset M4RAW --epochs 100
```

To perform corrective unlearning with SSD:
```bash
python unlearning_.py --alpha 0.2 --lambda 0.1
```

To run the CycleGAN:
```bash
python cyclegan_train.py --dataset_path ./data/healthy_to_cyst/
```

---

## 🎥 Visual Summary

- 🧠 Baseline and Unlearnt Reconstructions
- 📊 SSD Performance Landscape
- 🔄 CycleGAN Transformations
- ⚔️ Tumor Perturbation Examples


---

## ✍️ Authors

- Aryaman Bahl – [@aryaman.bahl](mailto:aryaman.bahl@research.iiit.ac.in)
- Chinmay Sharma 
- Sairam Babu - [@sairam.babu](mailto:sairam.babu@research.iiit.ac.in)
- Pranav Subramaniam

---
