# Invasive Weed Optimization (IWO) Algorithm Feature Selection

- Invasive Weed Optimization (IWO) algorithm feature selection
- You can use your own data or change number of features by "nf"
- "nVar" should be equal with total number of features
- ----------------------------------------------------------------------


![IWO Feature Selection](https://user-images.githubusercontent.com/11339420/163270016-9069e3af-55a6-40a5-9480-2cc698e6b21b.jpg)


This repository implements the **Invasive Weed Optimization (IWO)** algorithm for feature selection and evaluates its performance using a Neural Network-based cost function. The project includes loading datasets, feature selection, and comparison of classification performance before and after feature selection using K-Nearest Neighbors (KNN).

---

## Features

- **Dataset Loading**: Load input features and target values from a dataset (`iwodata`).
- **Feature Selection**: Optimize feature selection using the IWO algorithm.
- **Artificial Neural Network (ANN)**: Train and evaluate a neural network to calculate feature selection cost.
- **K-Nearest Neighbors (KNN)**: Compare classification accuracy on the original dataset and selected features.
- **Performance Visualization**: Iteration-wise cost tracking and confusion matrices for evaluation.

---

## Usage

1. **Dataset**: Place your dataset file (`iwodata.mat`) in the same directory as the scripts. The dataset should include:
   - `Inputs`: Features matrix.
   - `Targets`: Labels matrix.

2. **Run the main script**: Execute the `IWO Feature Selection.m` file in MATLAB.

3. **Results**:
   - Best-selected features are stored in `BestSol.out.S`.
   - Compare KNN performance before and after feature selection.

---

## Key Scripts

1. **Loading.m**:
    - Loads the dataset and extracts inputs and targets.
    
2. **CreateAndTrainANN.m**:
    - Creates and trains a neural network using the Levenberg-Marquardt algorithm.
    - Divides data into training, validation, and testing subsets.

3. **FeatureSelectionCost.m**:
    - Defines the cost function for IWO.
    - Evaluates the performance of selected features using ANN.

4. **IWO Feature Selection.m**:
    - Implements the Invasive Weed Optimization algorithm.
    - Compares KNN accuracy before and after feature selection.
    - Visualizes performance metrics.

---


## Dependencies

- MATLAB R2019b or later.
- Neural Network Toolbox.

---


- Quote: "If you set your goals ridiculously high and it's a failure,
- you will fail above everyone else's success
-                                                  James Cameron
