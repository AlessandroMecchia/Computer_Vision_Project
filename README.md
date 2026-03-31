# Computer Vision Project

A computer vision project focused on vegetation detection from image data through patch-based classification.

The workflow combines dataset preparation, binary mask generation, feature extraction, and model comparison. The implementation is developed mainly in the `project.ipynb` notebook, and the repository currently contains that notebook plus the Python environment files (`requirements.txt`, `Pipfile`, `Pipfile.lock`).

---

## Overview

The purpose of this project is to build a structured image-processing pipeline for identifying vegetation areas in visual data.
The notebook includes the full sequence of steps required to:

- inspect and validate the dataset structure;
- resize images for processing;
- generate binary vegetation masks from RGB label maps;
- extract fixed-size image patches;
- prepare classification datasets;
- train and evaluate two different classifiers:
  - K-Nearest Neighbors (KNN)
  - Convolutional Neural Network (CNN)

The project is designed as a comparative study between a classical machine learning baseline and a convolution-based image classifier applied to the same patch dataset.

---

## Repository Structure

```text
Computer_Vision_Project/
├── project.ipynb
├── requirements.txt
├── Pipfile
├── Pipfile.lock
└── README.md
```

The repository is almost entirely notebook-based and the current implementation is centered on a single main notebook.

---

## Project Goal

The main goal is to classify local image regions as vegetation or non-vegetation after transforming the original data into a patch-level supervised learning problem.

Instead of operating only on full images, the project extracts smaller local regions and uses them as classification samples. This makes the workflow easier to inspect, compare, and evaluate across different methods.

---

## Dataset Organization

The notebook works with three aligned image folders:

- `orig` — original images
- `label` — ground-truth labels
- `rgb` — RGB segmentation maps used to derive the binary vegetation mask

The dataset management section explicitly loads `.jpg` files from `orig` and `.png` files from both `label` and `rgb`.
The notebook also references an output directory such as `dataset/resized`, where resized copies of the dataset are stored.

A utility function named `resize_and_save_all(...)` is used to generate the resized dataset structure automatically while preserving the organization of the input folders.

---

## Processing Pipeline

### 1. Dataset Inspection and Consistency Checks

The notebook starts by loading the image collections and checking that the dataset is correctly structured before any training step.
This phase is useful to ensure that all required files are present and aligned across the three folders.

### 2. Image Resizing

The project includes a preprocessing function:

```python
resize_and_save_folder(input_dir, output_dir, wscale=1.0, hscale=1.0)
```

This function opens each image, rescales it using PIL with `Image.LANCZOS`, and saves the resized version to a target folder.
In the dataset management section, the notebook sets a resizing scale of `0.25`, indicating that images are downscaled before the rest of the pipeline is applied.

### 3. Binary Mask Generation

The notebook converts RGB annotation maps into binary vegetation masks.
This step reduces the original label representation to a simpler two-class format:

- vegetation
- non-vegetation

This binary representation is later used to assign labels to extracted image patches.

### 4. Patch Extraction

The core representation used for classification is based on image patches.
From the notebook outputs, the final feature matrix for the KNN classifier has shape `(2376, 3072)`.
This confirms that the project produces **2376 patch samples**, each transformed into a **3072-dimensional feature vector** for the classical classifier.

The implemented pipeline clearly operates on fixed-size patches and converts them into vectorized samples for classical classification.

### 5. Feature Construction

After patch extraction, the notebook reshapes the patch tensor into a 2D feature matrix:

- features: `(2376, 3072)`
- labels: `(2376,)`

This is the input format used by the KNN classifier.

### 6. Dataset Splitting

The notebook performs a stratified split of the classification dataset.
The reported shapes are:

- training set: `1164` samples
- validation set: `499` samples
- test set: `713` samples

These values come from a two-step split:

1. train+validation vs test
2. train vs validation

This setup gives a clear separation between model fitting, hyperparameter assessment, and final evaluation.

### 7. Model Training

The notebook imports and uses both:

- `sklearn.neighbors` for the KNN-based classifier;
- Keras / TensorFlow layers such as `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dropout` for the CNN model.

This confirms that the repository compares a traditional non-parametric classifier with a convolutional architecture trained directly on image patches.

### 8. Evaluation

The imported evaluation utilities include:

- `classification_report`
- `confusion_matrix`
- `ConfusionMatrixDisplay`
- `f1_score`
- `roc_curve`
- `auc`
- `recall_score`
- `precision_score`
- `accuracy_score`

This indicates that the notebook evaluates the models beyond simple accuracy and includes class-level and threshold-based analysis.

---

## Implemented Models

### K-Nearest Neighbors (KNN)

The KNN model is trained on flattened patch vectors.
This provides a straightforward baseline that is easy to interpret and useful for checking whether the extracted patch representation already contains enough information for vegetation discrimination.

### Convolutional Neural Network (CNN)

The CNN model is built with Keras/TensorFlow and uses convolutional layers to learn spatial patterns directly from the patch images rather than from manually flattened representations.
The notebook imports `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, and `Dropout`, together with callbacks such as `ModelCheckpoint` and `EarlyStopping`, which suggests a training workflow with validation monitoring and regularization support.
