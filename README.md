# Image Detector

A deep learning project for classifying images into categories (cats and dogs) using a Convolutional Neural Network (CNN).

## Project Structure

```
image_detector/
├── data/                          # Training data directory
│   ├── cats/                      │   └── dogs/                      ├── model/                         │   ├── data_prep.py              │   ├── model_definition.py        │   └── train_classifier.py        ├── trained_cnn_model.keras/      ├── logs/                          │   └── fit/20251217-232402/
├── requirements.txt               └── README.md                     
```

## Features

- **Data Preprocessing**: Scripts to prepare and augment image data
- **Custom CNN Architecture**: Convolutional neural network optimized for image classification
- **Model Training**: Training pipeline with validation and logging
- **TensorBoard Integration**: Real-time training metrics visualization
- **Keras Model Serialization**: Trained model saved in `.keras` format

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

**Dependencies:**
- pandas
- numpy
- tensorflow
- matplotlib
- tensorflow.keras

## Usage

### 1. Prepare Data
```bash
python model/data_prep.py
```

### 2. Define and Train Model
```bash
python model/train_classifier.py
```

### 3. View Training Logs
```bash
tensorboard --logdir=logs/fit
```

## Model Details

- **Architecture**: Sequential CNN with convolutional, pooling, and dense layers
- **Input**: Image data from `data/cats` and `data/dogs` directories
- **Output**: Binary classification (cat or dog)
- **Saved Model**: `trained_cnn_model.keras`

## Training Logs

Training metrics are logged to `logs/fit` for visualization with TensorBoard, including:
- Training loss and accuracy
- Validation loss and accuracy
- Epoch-by-epoch performance
