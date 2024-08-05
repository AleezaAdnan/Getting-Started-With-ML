# Supervised Learning

Supervised learning is a type of machine learning where the algorithm is trained on a labeled dataset. This means that each training example is paired with an output label. The algorithm learns to map the input data to the correct output by finding patterns in the data. Once trained, the model can predict the output for new, unseen data.

## Regression vs. Classification

### Regression
Regression is used to understand the relationship between different variables. It helps us predict one variable based on another by finding a pattern or trend in the data. It draws a line or curve that best represents how one variable changes in relation to another.

- **Example**: Predicting the price of a house based on its size, location, and number of bedrooms.

### Classification
Classification is used when the output variable is a category. The goal of classification is to predict which category the input data belongs to. i.e labelling the input to correct class/category. 

- **Example**: Classifying tumors as malignant or benign 

## Common Algorithms in Supervised Learning

### 1. Linear Regression
Linear Regression is used to predict a number based on a trend in the data. It finds the best-fit line through your data points. It draws a line that best represents the relationship between input features and the output value. You use this line to make predictions.

Go through [Linear Regression Notebook](linear_regression.ipynb) to understand the implementation of linear regression using scikitlearn on the housing price prediction dataset.
 
*Model Performance*
```
Mean Squared Error (MSE): 1754318687330.669
Root Mean Squared Error (RMSE): 1324506.960
R^2 Score: 0.653

```
- MSE and RMSE: High values suggest that there is considerable error in the predictions. You might want to investigate if different preprocessing steps could improve the model’s performance.
- R² Score: A score of 0.653 is reasonably good but indicates that there is room for improvement. You might consider trying more complex models, adding more features, or exploring interactions between features.
Refer to [Improving your ML Model](../../Improving%20Model%20Performance/) to learn how to improve the model's performace.

