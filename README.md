# Simpsons-MNIST Multi-Class Perceptron Classification

## Project Overview
This project implements a multi-class perceptron classifier for recognizing characters from the Simpsons-MNIST dataset. The dataset consists of 28x28 images representing 10 different Simpsons characters in both grayscale and RGB formats. The classifier extends a binary perceptron to a multi-class setting using a one-vs-rest approach.

The key objectives include:
* Data processing for loading and normalizing images
* Implementation of binary and multi-class perceptron models from scratch using NumPy
* Training perceptron models with different stopping criteria
* Hyperparameter tuning for learning rate, weight initialization, and input normalization
* Evaluation of model performance with accuracy, precision, recall, F1 score, and confusion matrices
* Comparative analysis of RGB vs grayscale input modalities
* Code designed for reproducibility and clarity using an object-oriented approach

## Project Structure
```
project/
├── Simpsons-MNIST/
│   ├── grayscale/
│   │   ├── train/
│   │   └── test/
│   └── rgb/
│       ├── train/
│       └── test/
├── notebooks/
│   └──A2.ipynb
└── README.md
```

## Setup & Requirements

### Prerequisites
* Python 3.7 or higher

### Installation
Install the required libraries using pip:

```bash
pip install numpy matplotlib pillow scikit-learn jupyter
```

### Required Libraries
* `numpy` - For numerical computations and array operations  
* `matplotlib` - For data visualization and plotting  
* `Pillow (PIL)` - For image processing and loading  
* `scikit-learn` - For evaluation metrics and confusion matrix visualization  
* `jupyter` - For running the analysis notebook  

### Dataset Setup
Ensure you have the Simpsons-MNIST dataset downloaded and structured as follows:
```
data/
├── grayscale/
│   ├── train/
│   │   ├── class_0/
│   │   ├── class_1/
│   │   └── ... (class_0 through class_9)
│   └── test/
└── rgb/
    ├── train/
    └── test/
```

Each class folder should contain 28x28 pixel images of the corresponding Simpsons character.

## Quick Start

# Load and preprocess data
loader = Loader()
X_train, y_train, X_val, y_val, X_test, y_test = loader.load_grayscale_data()

# Create and train model
model = MultiClassPerceptron(input_size=784, num_classes=10)
trainer = PerceptronTrainer(model)
trainer.train(X_train, y_train, X_val, y_val, epochs=100)

# Evaluate
accuracy = trainer.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

## Usage Instructions
1. **Data Loading and Preprocessing:** Use the `Loader` class to load grayscale and RGB datasets, normalize pixel values, flatten images into vectors, and split into training, validation, and test sets.  

2. **Model Implementation:**
   * `BinaryPerceptron` implements binary classification using the perceptron learning rule.
   * `MultiClassPerceptron` extends to 10 classes using one-vs-rest strategy.
   * `EnhancedMultiClassPerceptron` supports different initialization strategies for experimental tuning.

3. **Training:**
   * Use `PerceptronTrainer` or `EnhancedPerceptronTrainer` to train models with options for fixed epochs, error threshold, or early stopping.
   * Monitor training progress and validation accuracy throughout.

4. **Hyperparameter Tuning:**
   * `HyperparameterTuner` automates grid search over learning rates, initialization methods (zero, constant, uniform, Gaussian), and normalization techniques (none, min-max, z-score).
   * Separate tuning for grayscale and RGB inputs.

5. **Evaluation:**
   * Evaluate final models on validation and test sets.
   * Metrics: accuracy, precision, recall, F1 score per class.
   * Generate and visualize confusion matrices.

6. **Visualization:**
   * Use `Loader` to visualize sample images.
   * Display correctly and incorrectly classified samples for analysis.

## Results
Hyperparameter tuning and evaluation results will be documented here after running the full analysis. Metrics and visualization tools help assess model performance across configurations.

## Key Concepts
* **One-vs-Rest classification:** Multi-class classification using one binary classifier per class  
* **Perceptron learning rule:** Weight updates based on difference between true label and predicted label  
* **Weight initialization:** Initial weight distributions affect training stability and convergence  
* **Normalization:** Scaling input data impacts learning behavior and accuracy  
* **Stopping criteria:** Prevents overfitting and unnecessary training by monitoring metrics  

## Running the Analysis 
1. Ensure the dataset is structured in the `Simpsons-MNIST ` folder  
2. Install dependencies: `pip install -r requirements.txt`  
3. Open and run the Jupyter notebook: `A2.ipynb`  
4. Follow the notebook cells for the complete workflow  

## Contributing
1. Follow the established object-oriented code structure  
2. Add documentation and comments  
3. Test new features with both grayscale and RGB datasets  
   

 
