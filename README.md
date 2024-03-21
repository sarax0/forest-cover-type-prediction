# Forest Cover Type Prediction in Roosevelt National Forest

This project utilizes machine learning algorithms to predict the dominant tree cover type for forest areas within the Roosevelt National Forest situated in northern Colorado. The project leverages a publicly available dataset from the UCI Machine Learning Repository containing 581,012 data points with 7 numeric features describing cartographic variables that influence forest cover types ([Source](https://archive.ics.uci.edu/ml/datasets/Forest+Cover+Types)). 

![image](https://github.com/sarax0/forest-cover-type-prediction/assets/122404545/756249d6-2920-4962-a754-d9c359094558)

## Project Goals

*  Employ machine learning models to predict various forest cover types (Spruce-Fir, Lodgepole Pine, Ponderosa Pine, Cottonwood/Aspen, Mixed Conifer, Deciduous) in the Roosevelt National Forest.
*  Evaluate the performance of different classification algorithms (Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, SVM, Logistic Regression) to establish a baseline.
*  Implement feature selection techniques (Best Filtered Features, Best Wrapper Features, PCA) to identify the most impactful cartographic variables for predicting forest cover types.
*  Develop and integrate Gradient Boosting Classifier and Bagging Classifier algorithms for potentially improved prediction accuracy.
*  Assess model performance using accuracy and F1 score metrics.

## Data

This project utilizes the Forest Cover Type dataset from the UCI Machine Learning Repository. The dataset includes the following features:

* **Elevation:** Elevation in meters
* **Slope:** Slope in degrees
* **Aspect:** Aspect in degrees
* **Horizontal Distance to Roadways:** Distance in meters
* **Horizontal Distance to Streams:** Distance in meters
* **Horizontal Distance to Nearest Fire Points:** Distance in meters
* **Wilderness Area:** Binary variable (0 or 1) indicating presence within a designated wilderness area
* **Target Variable:** Forest cover type (categorical)

## Exploratory Data Analysis (EDA)

An initial exploratory data analysis (EDA) will be conducted to understand the distribution of features, identify potential relationships between variables, and uncover missing values (if any). This analysis will be crucial for data preprocessing and feature selection.

## Classification Models

### Baseline Models

The project will begin by implementing various classification algorithms to establish a baseline performance for forest cover type prediction. These models will include:

* Decision Tree
* K-Nearest Neighbors (KNN)
* Gaussian Naive Bayes
* Support Vector Machine (SVM)
* Logistic Regression

The performance of these models will be evaluated using accuracy and F1 score metrics.

### Feature Selection and Dimensionality Reduction

Feature selection techniques will be employed to identify the most relevant features that contribute significantly to the prediction of forest cover types. Three methods will be explored:

* **Best Filtered Features:** This approach utilizes statistical tests to rank features based on their correlation with the target variable.
* **Best Wrapper Features:** This method iteratively evaluates subsets of features to determine the optimal combination for prediction.
* **Principal Component Analysis (PCA):** This technique reduces dimensionality by transforming features into a set of uncorrelated principal components, potentially improving model performance.

The effectiveness of each feature selection method will be compared based on the resulting model performance. 

### Gradient Boosting Classifier and Bagging Classifier

Following feature selection, the project will implement:

* **Gradient Boosting Classifier:** This ensemble learning technique combines multiple weak decision trees to create a robust model with enhanced predictive power.
* **Bagging Classifier:** This method involves training multiple decision trees on random subsets of data with replacement (bootstrapping) and aggregating their predictions to improve accuracy and reduce variance.

The performance of these advanced models will be compared against the baseline models and evaluated using the chosen metrics.

## Performance Evaluation

The performance of each model will be assessed using:

* **Accuracy:** This metric measures the proportion of correctly classified forest cover types.
* **F1 Score:** This metric considers both precision (correct positive predictions) and recall (correctly identified positives) to provide a balanced evaluation.


This project contributes to the field of forest management by providing valuable insights into forest cover distribution and offering a machine learning approach to predict future changes.
