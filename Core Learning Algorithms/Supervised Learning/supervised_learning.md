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
Linear Regression is used to predict a number based on a trend in the data. It finds the best-fit line through your data points (this best fit line is calculated using different mathematical formulas and each of them has specific use cases). It draws a line that best represents the relationship between input features and the output value. You use this line to make predictions.

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

### 2. Decision Trees
Decision Trees split data into branches to make predictions. Each branch represents a decision based on a feature.
It creates a tree-like model where each branch is a decision point, leading to a prediction at the end.

It works for both Regression and Classification problems
Regression: Predicts numbers by averaging outcomes in the branches.
Classification: Sorts data into categories based on majority votes.

[Decision Tree NoteBook](decision_tree.ipynb) covers the implementation of Decision tree for classification problem on the titanic dataset.

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



<p allign= 'center'>
<table font-size=11>
  <thead>
    <tr>
      <th>Kernel Type</th>
      <th>Application in SVM (Classification)</th>
      <th>Application in SVR (Regression)</th>
      <th>Example Use Case</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Linear Kernel</strong></td>
      <td>Used when the data is linearly separable. It finds a straight line (hyperplane) that separates the classes.</td>
      <td>Used for datasets where the relationship between features and the target is approximately linear. It fits a straight line to the data.</td>
      <td><strong>SVM:</strong> Spam email classification, where features like the presence of certain words may separate spam from non-spam.<br><strong>SVR:</strong> Predicting house prices based on features like square footage, assuming a linear relationship.</td>
    </tr>
    <tr>
      <td><strong>Polynomial Kernel</strong></td>
      <td>Used when the relationship between the classes is not linear. It finds a curved decision boundary by transforming the data into a higher dimension.</td>
      <td>Used when the relationship between the features and the target is non-linear. It fits a polynomial curve to the data.</td>
      <td><strong>SVM:</strong> Classifying images of handwritten digits, where the decision boundary between different digits is non-linear.<br><strong>SVR:</strong> Predicting the number of visitors to a theme park based on a non-linear relationship with weather conditions.</td>
    </tr>
    <tr>
      <td><strong>Radial Basis Function (RBF) Kernel</strong></td>
      <td>Ideal for complex, non-linear datasets. It creates a decision boundary that can adapt to the intricacies of the data by mapping it into an infinite-dimensional space.</td>
      <td>Used for non-linear regression tasks where the relationship between features and the target is complex and cannot be captured by a linear or polynomial function.</td>
      <td><strong>SVM:</strong> Image classification where the decision boundary between objects is highly complex.<br><strong>SVR:</strong> Predicting stock market trends where the relationship between historical data and future prices is complex and non-linear.</td>
    </tr>
    <tr>
      <td><strong>Sigmoid Kernel</strong></td>
      <td>Less commonly used but can mimic the behavior of neural networks. It maps data into a high-dimensional space similar to the hidden layers of a neural network.</td>
      <td>Similar to its use in SVM, the Sigmoid kernel can model non-linear relationships in SVR by mapping data into a higher-dimensional space.</td>
      <td><strong>SVM:</strong> Classifying user behavior on a website into categories like "interested" or "not interested," based on complex patterns.<br><strong>SVR:</strong> Modeling the relationship between advertising spend and sales, where the effect of advertising on sales is non-linear.</td>
    </tr>
  </tbody>
</table>

</p>


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

The model performs exceptionally well with an accuracy of 99%, correctly classifying 168 instances of class 1 and 47 instances of class 0. Precision, recall, and F1-scores for class 1 are all near-perfect, with only a few misclassifications of class 1 instances as class 0. The confusion matrix shows no false positives for class 0, showing the model's accuracy.


