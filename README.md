# Skin Disease Classification Using VGG16 and Transfer Learning

A deep learning image-classification project that explores identifying **five skin-condition categories** from photographs using **TensorFlow/Keras** and a pretrained **VGG16** convolutional neural network. The project is implemented in a Google Colab notebook.

> **Disclaimer:** This is an educational image-classification experiment, not a medical diagnostic tool. Predictions should not be used for diagnosis or treatment decisions.

## Project Overview

The notebook loads labeled skin images from folders, performs image preprocessing and augmentation, builds a VGG16-based transfer-learning classifier, and experiments with model training, fine-tuning, and validation.

### Classes

- Acne (`acne`)
- Eksim (`eksim`)
- Herpes (`herpes`)
- Panu (`panu`)
- Rosacea (`rosacea`)

Class names are preserved exactly as they appear in the dataset folders.

## Technology Stack

- **Language:** Python
- **Deep learning:** TensorFlow, Keras
- **Architecture:** VGG16 pretrained on ImageNet
- **Image processing:** OpenCV, Pillow
- **Data handling:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Environment:** Google Colab, Google Drive

## Dataset

The notebook reads images from `train/` and `test/` directories. Its recorded data-loading output shows:

| Split | Images |
|---|---:|
| Training | 1,188 |
| Validation | 295 |
| Test directory | 11 |

The training and validation sets are generated from the `train/` directory using an **80/20 split**. The test directory is loaded without class labels (`class_mode=None`). Dataset source and licensing are not specified in the notebook.

Expected folder layout:

```text
project_skin/
├── skin(2).ipynb
├── train/
│   ├── acne/
│   ├── eksim/
│   ├── herpes/
│   ├── panu/
│   └── rosacea/
└── test/
    ├── acne/
    ├── eksim/
    ├── herpes/
    ├── panu/
    └── rosacea/
```

## Workflow

1. Mount Google Drive in Colab and open the project directory.
2. Load training and validation images with `ImageDataGenerator`.
3. Resize images to **100 × 100 RGB** and normalize pixel values to `[0, 1]`.
4. Augment training images with rotation, zoom, translation, shear, and horizontal flipping.
5. Load **VGG16** without its original classification head, using ImageNet weights.
6. Add global average pooling, a 16-unit ReLU dense layer, and a **5-class softmax** output.
7. Compile with Adam, categorical cross-entropy, and accuracy.
8. Train and experiment with fine-tuning; evaluate using validation images.

## Model Architecture

```text
Input image (100 × 100 × 3)
           ↓
VGG16 pretrained convolutional base
           ↓
GlobalAveragePooling2D
           ↓
Dense (16 units, ReLU)
           ↓
Dense (5 units, Softmax)
           ↓
Predicted skin-condition class
```

The notebook first defines a small custom CNN, then replaces it with the VGG16 transfer-learning model. The initial VGG16 training configuration freezes its convolutional base. A later cell explores fine-tuning with a lower learning rate.

## Installation and Setup

**Option A: Google Colab (used in the notebook)**

1. Open `skin(2).ipynb` in Google Colab.
2. Make the dataset available in your Google Drive under `MyDrive/project_skin/`, or update the notebook's working directory.
3. Run the cells in order, checking the notes below about incomplete cells.

**Option B: Local Python environment**

Install the notebook's primary dependencies:

```bash
pip install tensorflow numpy pandas matplotlib opencv-python pillow jupyter
```

Then open the notebook in Jupyter or VS Code. Replace Colab-specific Google Drive commands and paths with local equivalents.

## Training Configuration

| Parameter | Value |
|---|---|
| Input size | 100 × 100 × 3 |
| Batch size | 8 |
| Validation split | 20% |
| Initial training epochs | 5 |
| Optimizer | Adam |
| Loss | Categorical cross-entropy |
| Output classes | 5 |
| Fine-tuning learning rate (experimental cell) | 0.00001 |

## Results and Current Status

The notebook confirms that images were loaded into five classes and shows the start of a five-epoch training run. **It does not contain a complete, verifiable final accuracy or loss result**, so no performance claim is made here.

### Notebook limitations

- The earlier custom CNN definition has a four-class output, which does not match the five dataset classes; the later VGG16 definition corrects this.
- The later fine-tuning cell references `model` without clearly rebuilding and wiring the VGG16 base into a new model. It needs review before use as a reproducible fine-tuning procedure.
- The evaluation cell labels validation metrics as “Test” metrics; it evaluates `valid_generator`, not the test directory.
- The final webcam/OpenCV cell is unfinished pseudocode, **not a working real-time prediction feature**.
- The notebook does not document a saved model, confusion matrix, or completed test-set assessment.

## Future Improvements

- Complete and validate the fine-tuning workflow.
- Record final training/validation curves and metrics.
- Evaluate on a properly labeled held-out test set.
- Add a confusion matrix, precision, recall, and F1-score.
- Save the trained model and create a separate image-inference script.
- Investigate dataset balance and generalization across lighting and skin tones.

## Repository Contents

```text
skin-disease-classification/
├── skin(2).ipynb
└── README.md
```

The dataset is not included in this suggested repository structure. Add an appropriate dataset citation and license information before publishing any images.

## Author

Developed as a deep learning and computer vision learning project using Python, TensorFlow/Keras, and VGG16 transfer learning.
