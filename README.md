# Rice-Classification-and-Influence-Analysis

## **Overview**
This repository contains the implementation of a Random Forest machine learning model for classifying rice grains. Additionally, an influence analysis is conducted by systematically removing data points to assess their impact on model performance. The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik)

---

## **Table of Contents**
- Dataset
- Installation
- Usage
- Model Training
- Performance Evaluation
- Influence Analysis
- Shapley Values

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

