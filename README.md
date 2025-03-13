# Rice-Classification-and-Influence-Analysis

## **Overview**
This repository contains the implementation of a Random Forest machine learning model for classifying rice grains. Additionally, an influence analysis is conducted by systematically removing data points to assess their impact on model performance. The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik)

---
## **Technologies Used** 
- **Programming Language**: Python
- **Libraries**:
  - `matplotlib` – Data visualization
  - `seaborn` – Statistical data visualization
  - `pandas` – Data manipulation and analysis
  - `numpy` – Numerical computations
  - `sklearn` – Machine learning models and evaluation metrics
  - `ucimlrepo` – Fetching datasets from the UCI Machine Learning Repository

---
## **Dataset**
The dataset consists of 3810 instances with 8 morphological features extracted from images of rice grains. The classification task involves predicting the rice type (Cammeo or Osmancik) based on these features:
1. Area
2. Perimeter
3. Major_Axis_Length
4. Minor_Axis_Length
5. Eccentricity
6. Convex_Area
7. Extent
8. Class (Target Variable)

## **Model Training**
- The dataset is loaded using fetch_ucirepo(id=545)
- Preprocessing steps include:
  - Label encoding of the target variable
  - Splitting the dataset into 80% training and 20% testing
- The model was trained using:
  - K-Nearest Neighbors (KNN) with n=5
  - Random Forest Classifier (n_estimators=100, random_state=42, n_jobs=-1)
The best-performing model was Random Forest, achieving 92% F1 Score

## **Performance Evaluation**
- The evaluation metrics include:
  - F1 Score (preferred due to class imbalance)
  - Confusion Matrix
  - Precision-Recall Curve
- The results indicate strong classification performance with minimal overfitting

---
# Influence Analysis

## **Leave-One-Out (LOO) Influence**
- The effect of removing single data points was analyzed to measure their impact on model performance
- Points with high negative influence suggest potential outliers or mislabeled data
- My approach to influence analysis was inspired by concepts from [Hammoudeh & Lowd (2024)](https://arxiv.org/abs/2212.04612), which provides a comprehensive survey on training data influence estimation

## **Group-Level Influence**
- Subsets of the dataset were analyzed to understand how different sample sizes impact the model's F1 Score.
- Sample sizes ranged from 10% to 100%.

## **Shapley Values**
- Implemented Truncated Monte Carlo Shapley Values to estimate feature importance efficiently.
- The computation stops early if additional points do not significantly alter the model's performance (tolerance = 0.001).

---
# Setup
### 1. Dependencies
Ensure Python is installed along with the required libraries. Install missing packages using:
```
pip install {Package}
```

### 2. Running the Code
Run the code by entering this into terminal:
```
python model.py
```
