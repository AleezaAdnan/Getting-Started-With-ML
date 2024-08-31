# Supervised Learning

Supervised learning is a type of machine learning where the algorithm is trained on a labeled dataset. This means that each training example is paired with an output label. The algorithm learns to map the input data to the correct output by finding patterns in the data. Once trained, the model can predict the output for new, unseen data. 
There are two kinds of supervised learning algorithms
- Regression 
- Classification

## Regression vs. Classification

### Regression
Regression is used to understand the relationship between different variables. It helps us predict one variable based on another by finding a pattern or trend in the data. It draws a line or curve that best represents how one variable changes in relation to another.

- **Example**: Predicting the price of a house based on its size, location, and number of bedrooms.

### Classification
Classification is used when the output variable is a category. The goal of classification is to predict which category the input data belongs to. i.e labelling the input to correct class/category. 

- **Example**: Classifying tumors as malignant or benign 

## Common Algorithms in Supervised Learning

### 1. Linear Regression
Linear Regression is used to predict a number based on a trend in the data. It finds the best-fit line through your data points (this best fit line is calculated using different mathematical formulas and each of them has specific use cases). It draws a line that best represents the relationship between input features and the output value. You use this line to make predictions.

- In simple linear regression (one predictor), the model is represented by the equation:  
  \[
  y = \beta_0 + \beta_1x + \epsilon
  \]
  - \(y\) is the dependent variable (target).
  - \(x\) is the independent variable (predictor).
  - \(\beta_0\) is the intercept (the value of \(y\) when \(x = 0\)).
  - \(\beta_1\) is the slope (how much \(y\) changes for a unit change in \(x\)).
  - \(\epsilon\) represents the error term (the difference between the observed and predicted values).

- In multiple linear regression (multiple predictors), the model extends to:
  
  \[
  y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n + \epsilon
  \]
  where \(x_1, x_2, \dots, x_n\) are the predictors.


Go through [Linear Regression Notebook](linear_regression.ipynb) to understand the implementation of linear regression using scikitlearn on the housing price prediction dataset.
 
*Model Performance*
```
Mean Squared Error (MSE): 1754318687330.669
Root Mean Squared Error (RMSE): 1324506.960
R^2 Score: 0.653

```
- MSE and RMSE: High values suggest that there is considerable error in the predictions. You might want to investigate if different preprocessing steps could improve the model’s performance.
- R² Score: A score of 0.653 is reasonably good but indicates that there is room for improvement. You might consider trying more complex models, adding more features, or exploring interactions between features.
Refer to [Fine Tuning ML Models](../../Fine%20Tuning%20ML%20Models/README.md) to learn how to improve the model's performace.

### 2. Decision Trees
Decision Trees split data into branches to make predictions. Each branch represents a decision based on a feature.
It creates a tree-like model where each branch is a decision point, leading to a prediction at the end.

It works for both Regression and Classification problems
Regression: Predicts numbers by averaging outcomes in the branches.
Classification: Sorts data into categories based on majority votes.

[Decision Tree NoteBook](decision_tree.ipynb) covers the implementation of Decision tree for classification problem on the titanic dataset.

*Model- Performance*
- The model achieves strong overall accuracy (84.9%) and high precision (92.2%), indicating it correctly identifies survivors most of the time when it predicts survival. However, the recall (72.8%) suggests it misses a notable portion of actual survivors, which could be critical in applications where missing true positives is costly. 
- To improve in more critical contexts, focusing on increasing recall, possibly by adjusting the decision threshold or using techniques like boosting, could help ensure more survivors are correctly identified.

### 3. Support Vector Machine
SVM is primarily a classification algorithm but can also be adapted for regression tasks (called Support Vector Regression, SVR).
#### SVM 
(for classification tasks) finds the optimal hyperplane that best separates the classes in the data by maximizing the margin between them (i.e they are as far away from the hyperplane as possible). In higher dimensions, this decision boundary can be more complex, especially with the use of kernels
The visualization in this [video](https://youtu.be/_YPScrckx28?si=NnkVJIk4UAxsV3Gi) will make it a bit easier to understand

**Kernel Trick:** SVM handles non-linear relationships by using kernels. A kernel function transforms the data into a higher-dimensional space where a linear separation is possible. 

#### SVR 
Support Vector Regression focuses on finding a function that approximates the target values within a specified margin of error. The goal is to minimize the prediction error while allowing for a margin where errors are tolerated.

**Epsilon-Insensitive Loss Function:** SVR uses an epsilon-insensitive loss function, which means that errors within a certain margin (epsilon) are ignored. The algorithm tries to fit the best possible line or curve within this margin, aiming to achieve a balance between model complexity and fit accuracy.

##### SVM and SVR Kernels Overview

1. **Linear Kernel**
- **SVM**: 
  - For linearly separable data.
  - **Example**: Classifying emails as spam or not spam.
- **SVR**:
  - Fits a straight line assuming a linear relationship.
  - **Example**: Predicting house prices based on size.

2. **Polynomial Kernel**
- **SVM**:
  - For more complex, non-linear boundaries.
  - **Example**: Classifying handwritten digits.
- **SVR**:
  - Fits a curved line for non-linear relationships.
  - **Example**: Predicting theme park visitors based on weather.

3. **Radial Basis Function (RBF) or Gaussian Kernel**
- **SVM**:
  - Ideal for complex, non-linear data.
  - **Example**: Classifying images with complex boundaries.
- **SVR**:
  - Models complex, non-linear relationships.
  - **Example**: Predicting stock prices from historical data.

4. **Sigmoid Kernel**
- **SVM**:
  - Mimics neural networks for complex scenarios.
  - **Example**: Classifying user behavior on a website.
- **SVR**:
  - Similar to classification, for neural network-like relationships.
  - **Example**: Modeling the relationship between advertising spend and sales.

[SVR Notebook](../datasets/WineQT.csv) shows the working of SVR for predicting wine quality 

*Model-Performance*
```
Mean Squared Error: 32.26017108108117
R-squared: 0.88639437257224
Mean Abslute Error: 3.778436859295195

```
- A lower MSE suggests that model's predictions are close to the actual values. 32.26 is relatively low, indicating a good fit.
- An R-squared value of 0.886 means that approximately 88.6% of the variance in the target variable is explained by the model. This indicates a strong relationship between the predictors and the target variable.
-an MAE of 3.78 suggests that, on average, the model's predictions are off by about 3.78 units from the actual values.

### 4. Random Forest 
Like Decision Trees, Random Forests can also be used for both classification and regression tasks. A Random Forest combines the output of multiple decision trees to reach asingle result.
Random Forest creates many trees (often hundreds). Each tree is trained on a random subset of the data and a random subset of features.
For regression tasks, the predictions from all the trees are averaged to get the final prediction. For classification, the final output is based on the majority vote of the individual trees.
It reduces the risk of overfitting and provide more accurate predictions. 

- Since each tree in the forest is trained on different subsets of the data, Random Forests are less likely to overfit compared to individual decision trees.
- Random Forests can handle missing values and maintain accuracy.
- They provide insights into which features are most important in making predictions, which is valuable for feature selection. After training a Random Forest model, you can easily retrieve the feature importance scores using built-in functions in libraries like Scikit-learn.

```
        importances = model.feature_importances_
        for feature, importance in zip(feature_names, importances):
            print(f"{feature}: {importance}")
```
In the [Credit Card](../datasets/CreditCard.csv), expenditure and shares are the most important features as shown in the bar chart in [Random Forest Notebook](random_forest.ipynb)

Hyper Parameter Tuning:
Hyperparameter tuning involves adjusting the settings of the Random Forest model to improve its performance. Common methods include:

- Grid Search: Systematically tests all possible combinations of predefined hyperparameter values.
- Random Search: Randomly samples hyperparameter values within specified ranges.
- Bayesian Optimization: Uses probabilistic models to efficiently explore the hyperparameter space.
Tuning these parameters, such as the number of trees (n_estimators) and maximum tree depth (max_depth), helps optimize the model’s accuracy and generalization

[Random Forest Notebook](random_forest.ipynb) covers the implementation of Random Forest for classification problem on the Credit Card dataset

*Model-Performance*
The model performs exceptionally well with an accuracy of 99%, correctly classifying 168 instances of class 1 and 47 instances of class 0. Precision, recall, and F1-scores for class 1 are all near-perfect, with only a few misclassifications of class 1 instances as class 0. The confusion matrix shows no false positives for class 0, showing the model's accuracy.

### 5. Logistic Regression

There are three types of Logistic Regression:

- **Binary Logistic Regression** is a statistical method used for binary classification, where the goal is to predict one of two possible outcomes. Unlike Linear Regression, which predicts continuous values, Logistic Regression predicts the probabilities that an instance belongs to a particular class.
  - It uses the *Sigmoid Function* (also called the logistic function) to map predicted values to probabilities, producing an S-shaped curve that outputs values between 0 and 1.
  - The model outputs a probability, and you can set a threshold/decision boundary (commonly 0.5) to classify the result into one of the two classes.

- **ROC Curve:** The ROC (Receiver Operating Characteristic) curve is a plot that illustrates the performance of a binary classifier like Logistic Regression. It shows the trade-off between the True Positive Rate (Sensitivity) and the False Positive Rate across different threshold values. A perfect model would achieve a point in the top-left corner of the ROC space (100% sensitivity and 0% false positive rate). The closer the ROC curve is to this top-left corner, the better the model. The Area Under the Curve (AUC) is a single number summary of the ROC curve.
  
  [Logistic Regression Notebook](logistic_regression.ipynb) covers the implementation of Binary Logistic Regression for detecting spam. In our case, an AUC score of 0.98 indicates that the model is highly effective in distinguishing between the two classes (spam vs. ham).

- **Multinomial Logistic Regression** is used when the target variable has three or more unordered categories (e.g., sheep, cow, bull). You can implement Multinomial Logistic Regression in the same way as binary logistic regression by adding the `multi_class` parameter:

  ```python
  model = LogisticRegression(multi_class='multinomial')

**Ordinal Logistic Regression** is used when the target variable has three or more ordered categories. For example, if you're predicting a customer's satisfaction level as "low," "medium," or "high," ordinal logistic regression can be used because the categories have a natural order. However, the distances between these categories are not necessarily equal.

Ordinal logistic regression is not directly supported by scikit-learn. However, you can use the statsmodels library to implement it. This type of regression is more specialized and is often used in fields like social sciences or medical research.
