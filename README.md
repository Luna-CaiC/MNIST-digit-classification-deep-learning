# MNIST Handwritten Digit Classification

## Project Overview
This project implements and compares multiple deep learning architectures for handwritten digit recognition using the MNIST dataset. The implementation includes Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) models with varying complexities.

## Problem Statement
The goal is to develop an image classification system to recognize handwritten digits (0-9) using the MNIST dataset. This problem is fundamental in computer vision and has practical applications in:
- Postal mail sorting
- Bank check processing
- Digitized document workflows
- Automated data entry systems

## Dataset
**MNIST Dataset**
- Training samples: 60,000 images
- Test samples: 10,000 images
- Image dimensions: 28×28 pixels (grayscale)
- Classes: 10 digits (0-9)
- Source: Loaded directly from Keras datasets

## Technical Architecture

### Models Implemented

#### 1. Multi-Layer Perceptron (MLP)
- **Architecture**: Fully connected neural network
- **Input**: Flattened 784-dimensional vector
- **Hidden Layers**: Dense layers with ReLU activation
- **Output**: 10-class softmax layer
- **Purpose**: Baseline model for comparison

#### 2. Basic Convolutional Neural Network (CNN)
- **Architecture**: 
  - Convolutional layers for feature extraction
  - Max pooling for dimensionality reduction
  - Dense layers for classification
- **Advantages**: Captures spatial hierarchies in images
- **Performance**: Improved accuracy over MLP

#### 3. Deep CNN with Dropout
- **Architecture**:
  - Multiple convolutional layers
  - Dropout regularization
  - Batch normalization
  - Advanced pooling strategies
- **Features**: 
  - Better generalization
  - Reduced overfitting
  - Highest accuracy among all models

## Key Technologies
- **Framework**: TensorFlow 2.x / Keras 3.x
- **Language**: Python 3.12
- **Libraries**:
  - NumPy for numerical operations
  - Matplotlib for visualization
  - Keras for deep learning models

## Project Structure
```
case3/
├── CaseStudy3.ipynb          # Main Jupyter notebook
├── README.md                  # Project documentation
└── .git/                      # Git repository
```

## Implementation Details

### Data Preprocessing
1. **Normalization**: Pixel values scaled to [0, 1] range
2. **Reshaping**: Images reshaped to (28, 28, 1) for CNN input
3. **Data Type**: Converted to float32 for computational efficiency

### Model Training
- **Optimizer**: Adam optimizer
- **Loss Function**: Categorical cross-entropy
- **Metrics**: Accuracy
- **Validation**: Training/validation split for monitoring
- **Epochs**: Variable based on model complexity

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Per-class performance analysis
- **Training Curves**: Loss and accuracy over epochs
- **Error Analysis**: Visualization of misclassified samples

## Results Summary

### Model Performance Comparison
| Model | Architecture | Test Accuracy | Key Characteristics |
|-------|-------------|---------------|---------------------|
| Model 1 | MLP | ~97-98% | Simple, fast training |
| Model 2 | Basic CNN | ~98-99% | Better feature extraction |
| Model 3 | Deep CNN + Dropout | ~99%+ | Best generalization |

### Key Findings
1. **Model Complexity vs. Performance**:
   - MLP lacks spatial feature extraction capabilities
   - CNN architectures significantly outperform MLP
   - Deeper networks with regularization achieve best results

2. **Error Analysis**:
   - Common errors: Visually similar digits (4 vs 9, 3 vs 8)
   - Ambiguous handwriting causes most misclassifications
   - Deeper models reduce error rates substantially

3. **Training Dynamics**:
   - Deeper models require more training time
   - Dropout prevents overfitting effectively
   - Validation accuracy improves with model depth

## Installation & Usage

### Prerequisites
```bash
pip install tensorflow keras numpy matplotlib
```

### Running the Project
1. Clone the repository
2. Open `CaseStudy3.ipynb` in Jupyter Notebook/Lab
3. Run all cells sequentially
4. Review visualizations and model performance

### Quick Start
```python
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Build and train model (example: Basic CNN)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_split=0.2)
```

## Visualization Features
- Training/validation accuracy and loss curves
- Confusion matrices for each model
- Sample predictions with confidence scores
- Misclassified digit examples
- Class distribution analysis

## Future Improvements
1. **Data Augmentation**: Rotation, scaling, translation
2. **Advanced Architectures**: ResNet, DenseNet implementations
3. **Transfer Learning**: Pre-trained model fine-tuning
4. **Ensemble Methods**: Combining multiple models
5. **Hyperparameter Optimization**: Grid search, Bayesian optimization

## Potential Impact
- **Automation**: Reduces manual data entry and errors
- **Scalability**: Foundation for complex computer vision tasks
- **Real-world Applications**: Deployable in production systems
- **Educational Value**: Benchmark for learning deep learning concepts

## Technical Requirements
- Python 3.12+
- TensorFlow 2.18.0+
- Keras 3.8.0+
- NumPy 1.26.0+
- Matplotlib 3.8.4+


## Acknowledgments
- MNIST dataset creators (Yann LeCun et al.)
- TensorFlow/Keras development team
- Academic institution for project framework

