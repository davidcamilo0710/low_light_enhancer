# Unsupervised Learning for Low-Light Image Enhancement

## Introduction

This project explores and compares two approaches for enhancing images captured in low-light conditions: a classical method based on digital image processing and a modern method using deep neural networks. The goal is to improve the visual quality and interpretability of dark images, a crucial pre-processing step for applications in computer vision, security, and automated analysis.

Both methods were evaluated using the [The Dark Face](https://www.kaggle.com/datasets/soumikrakshit/dark-face-dataset/data) dataset, and the results show significant improvements, though each approach has distinct strengths. The implemented deep learning model is **Zero-DCE++**, a convolutional neural network that performs enhancement without direct supervision.

## Visual Results Comparison

You can insert your comparative image here, showing the results of both methods against the original images.

<p align="center">
  <img width="897" height="832" alt="image_" src="https://github.com/user-attachments/assets/71ed98a9-5113-4aff-9b2c-4f12bc5cea60" />
</p>

## Methods Implemented

Two fundamentally different strategies were compared to address the low-light enhancement problem.

---

### 1. Classical Image Processing (Manual Adjustment)

This approach relies on a sequence of controlled image transformations to increase brightness, improve contrast, and reduce noise. The workflow is transparent, interpretable, and allows for fine-grained control over the final result.

The process consists of the following steps:
1.  **Gamma Correction**: A gamma correction (${\gamma=0.5}$) is applied to amplify low-intensity values and brighten the overall image.
2.  **Contrast Enhancement (CLAHE)**: The image is converted to the LAB color space to apply **Contrast Limited Adaptive Histogram Equalization (CLAHE)** on the Luminance (L) channel. This improves local contrast without overexposing the image.
    * **Clip Limit**: `2.0`
    * **Tile Grid Size**: `(8, 8)`
3.  **Noise Reduction**: A **Median Filter** (${\text{kernel}=3\times3}$) is used to smooth artifacts while preserving edges.
4.  **RGB Conversion**: Finally, the image is converted back to the RGB color space for display.

---

### 2. Deep Neural Network (Zero-DCE++)

This approach uses the **Zero-DCE++** model, a convolutional neural network trained to enhance images without requiring paired data (i.e., dark/light versions) for supervision. The network learns to generate an enhancement map that is iteratively applied to adaptively adjust the image's luminance.

* **Advantages**: Its main strengths are **automation** and its ability to **generalize** across a wide variety of dark scenes.
* **Training**: The model was trained with specific loss functions to preserve spatial consistency, color balance, and control exposure.
* **Implementation**: The training was performed in PyTorch using the Adam optimizer on the Dark Face dataset. You can find the training pipeline in [this repository](https://github.com/davidcamilo0710/low_light_enhancer/blob/main/pipeline.ipynb).

## Evaluation and Results

To assess the quality of the resulting images without a reference "ground truth," the following objective, no-reference evaluation metrics were used.

### Evaluation Metrics

* **BRISQUE** (Blind/Referenceless Image Spatial Quality Evaluator): Quantifies the perceptual quality of an image.
* **NIQE** (Natural Image Quality Evaluator): Measures the statistical naturalness of an image.
* **Entropy**: Indicates the amount of information or detail present in the image.
* **RMS Contrast**: Measures the standard deviation of pixel intensities (global contrast).

### Quantitative Comparison

The following table presents the metric results for a sample of 4 images from the dataset:

| Image      | Method          | BRISQUE | NIQE | Entropy | RMS Contrast |
| :--------- | :-------------- | :-----: | :--: | :-----: | :----------: |
| **1259.png** | Original        | 44.75   | 5.15 | 2.84    | 15.04        |
|            | Manual Tuning   | 41.92   | 4.78 | 4.45    | 40.20        |
|            | Zero-DCE++      | **38.93** | **3.77** | 4.17 | **80.87** |
| **2454.png** | Original        | 67.16   | 5.75 | 2.09    | 11.86        |
|            | Manual Tuning   | **48.05** | 5.10 | **3.84** | 32.56 |
|            | Zero-DCE++      | 48.40   | **4.57** | 3.59 | **67.82** |
| **2796.png** | Original        | **33.89** | 4.42 | 2.87 | 23.79        |
|            | Manual Tuning   | 39.56   | 4.88 | **4.61** | 43.83 |
|            | Zero-DCE++      | 39.98   | **3.81** | 4.48 | **74.05** |
| **2835.png** | Original        | 16.22   | 4.02 | 3.43    | 15.94        |
|            | Manual Tuning   | 35.48   | 4.37 | 4.91    | 37.83        |
|            | Zero-DCE++      | **11.37** | **2.99** | **5.00** | **52.27** |

*Table extracted from the project's results.*

### Results Analysis

* **Zero-DCE++** consistently achieves the **highest RMS contrast** and the **best (lowest) NIQE scores**, suggesting greater perceived naturalness and a more aggressive exposure correction.
* The **manual adjustment** method tends to yield **higher entropy values**, indicating better preservation of fine details. Visually, it produces more balanced results with a lower risk of overexposure.

## Conclusion

Both methods are effective, but their suitability depends on the use case:

* **Manual Adjustment** is ideal for controlled environments where **precise control** is needed, the goal is to preserve subtle details, and computational resources are limited.
* **Zero-DCE++** is superior for **automated and scalable pipelines** that require a robust, real-time response to diverse lighting conditions.

## Code and Resources

* **Language**: Python
* **Core Libraries**: PyTorch, OpenCV, NumPy, Matplotlib, PIL
* **Dataset**: [The Dark Face Dataset on Kaggle](https://www.kaggle.com/datasets/soumikrakshit/dark-face-dataset/data)
* **Training Pipeline**: [Jupyter Notebook on GitHub](https://github.com/davidcamilo0710/low_light_enhancer/blob/main/pipeline.ipynb)
* **Training Pipeline**: [Article on GitHub](https://github.com/davidcamilo0710/low_light_enhancer/blob/main/low_light_enhancer.pdf) 

## How to Cite This Project

If you use this work, please cite the repository:

```bibtex
@misc{Camilo2024LowLightEnhancer,
  author = {David Camilo Muñoz Garcia},
  title = {Unsupervised Learning for Low-Light Image Enhancement},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/davidcamilo0710/low_light_enhancer}}
}

