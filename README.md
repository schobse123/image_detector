# Image Detector

A deep learning project for classifying images into categories (cats and dogs) using a Convolutional Neural Network (CNN).

## Project Structure

```
image_detector/
├── app/                          # Simple UI / inference entry point
│   └── ui.py
├── data/                         # Training data directory
│   ├── cats/
│   └── dogs/
├── model/                        # Training pipeline
│   ├── data_prep.py
│   ├── model_definition.py
│   └── train_classifier.py
├── logs/                         # TensorBoard logs
│   └── fit/
├── trained_cnn_model.keras        # Trained Keras model
├── trained_cnn_model_labels.json  # Class labels for the model
├── requirements.txt
└── README.md
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

Note: `tensorflow.keras` is included with `tensorflow`.

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


## CNN Documentation

### Convolution: 
$$ (f*g)(t)=\int_{-\infty}^\infty f(\tau)g(t-\tau)d\tau $$
Convolution **smooths** functions and is therefore useful for handling noisy data, especially before differentiating.
More precisely, the function $g$ is reflected across the y-axis. For each point in time $t$, $g$ is shifted and the overlap (integral of the product) yields the convolution.

The convolution in multiple dimensions is
$$ (f*g)(x_1,x_2) = \int_{-\infty}^\infty \int_{-\infty}^\infty f(\hat x_1, \hat x_2)g(x_1-\hat x_1,x_2-\hat x_2) d\hat x_1 d\hat x_2 $$
Both definitions describe a continuous problem. In many cases, the data is **discrete**, so the $n$-th component of the convolution of two vectors is:
$$(\mathbf f * \mathbf g)_n = \sum_{m=-\infty}^{\infty}f_mg_{n-m}\stackrel{\text{commutativity}}{=} \sum_{m=-\infty}^{\infty}f_{n-m}g_{m} $$
For image data, it is necessary to use **2D convolutions**. Here, $\mathbf{F}$ and $\mathbf{G}$ are matrices.
$$ (\mathbf{F*G})_{m,n} = \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} F_{i,j}G_{m-i,n-j}\stackrel{\text{commutativity}}{=} \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} F_{i,j}G_{m-i,n-j} $$
Use padding to preserve the desired output dimensions.

### Structure
The ReLU Layer ensures, that the CNN is able to model nonlinear problems:
$$ Z = ReLU(X*W) $$
Z is the Output. The ReLU activation function is applied to the convolution to create threshold values. These values are then passed to the next layer. Without the ReLU each Output would be a linear combination of the input $X$ and the filter $W$.

**Pooling** reduces the size of feature maps by summarizing small areas into single values. Pooling is done at the level of each activation map, whereas convolution simultaneously uses all feature maps in combination with a filter to produce a single feature value. Therefore pooling does not change the number of feature maps!

### Backpropagation through Convolutions:

Backpropagation is used in the convolution, ReLU and pooling layers.
### Parameters:
- ***epoch***: a complete pass through the entire training dataset 
- ***batch_size***: Number of pictures used for one training/validation run 
> A small batch_size safes RAM, but slower in each epoch
- ***image_size***: should be $L=B$
- $\mathbf{n_{filters}}$: should be in each layer should be a power to 2 - more efficient processing
- ***filter_size***: small = better (mostly);  use 3 or 5; small filter lead to deeper networks for the same parameter footprint