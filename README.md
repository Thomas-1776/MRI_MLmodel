# 🧠 Lightweight Brain MRI Tumor Detection via Feature Fusion & SVM

An optimized, CPU-efficient Machine Learning pipeline for binary classification of Brain MRI scans (Tumor vs. Healthy). 

Rather than fine-tuning a massive end-to-end deep neural network, this project leverages **Dual-Backbone Feature Fusion** (MobileNetV2 + DenseNet121 / EfficientNetV2) paired with an **RBF Support Vector Machine (SVM)** classifier to achieve high diagnostic accuracy with low computational overhead.

---

## 🚀 Key Features

* **Multi-Perspective Feature Extraction:** Combines lightweight spatial geometry (MobileNetV2) with fine-grained pattern recognition (DenseNet121 / EfficientNetV2).
* **Hybrid Architecture:** Replaces dense deep learning heads with an RBF SVM to prevent overfitting on medical image representations.
* **Model Optimization:** Reduced parameter footprint by ~60% (from ~25M down to 9.29M params) while maintaining high accuracy and low CPU inference latency.
* **Class Imbalance Handling:** Implements balanced class weights for both training and cost-sensitive classification.
