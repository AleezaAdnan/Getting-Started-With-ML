# Regularization

Regularization introduces a penalty to the model's complexity, encouraging the model to find a simpler solution that generalizes better to new data. This helps in balancing the bias-variance trade-off, aiming for a model that is both accurate and generalizable. It directly influences the model's structure and its learning process.

## Key Techniques

### L1 Regularization (Lasso)

L1 Regularization, also known as *Lasso* (Least Absolute Shrinkage and Selection Operator), adds a penalty to the loss function equal to the absolute value of the coefficients:

$$
\text{Lasso Loss} = \text{Loss} + \lambda \sum_{i=1}^{n} |\theta_i|
$$

Here, $$\lambda$$ is the regularization parameter that controls the strength of the penalty, and $$|\theta_i|$$ represents the model coefficients. The penalty increases as the model coefficients grow larger, encouraging the model to shrink some of these coefficients towards zero.

**Key Characteristics:**

- **Feature Selection:** L1 regularization is particularly useful when you expect that only a few features are truly important. It tends to drive some coefficients to exactly zero, effectively performing feature selection. This makes it valuable in scenarios where only a subset of the features is expected to be relevant.

- **Sparsity:** L1 regularization promotes sparsity in the model by reducing some coefficients to zero. Sparse models are easier to interpret and computationally efficient.

#### **Implementation** 
L1 regularization can be applied to various ML models in sklearn by setting *penalty* parameter as *l1* and adjusting other parameters according to the chosen model

*Linear Regression:*

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)  # Initialize with regularization strength

```

*Logistic Regression:*

```python

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(penalty='l1', solver='liblinear')

```
*Support Vector Machine (SVM):*

```python

svm = LinearSVC(penalty='l1', dual=False, solver='liblinear')

```

By setting the appropriate parameters, you ensure that the model incorporates L1 regularization to enhance feature selection and create sparse models.

### L2 Regularization 

L2 Regularization, also known as *Ridge Regression*, adds a penalty to the loss function equal to the square of the magnitude of the coefficients:

$$
\text{Ridge Loss} = \text{Loss} + \lambda \sum_{i=1}^{n} \theta_i^2
$$

\(\lambda\) is the regularization parameter that controls the strength of the penalty, and \(\theta_i\) represents the model coefficients. Unlike L1 regularization, which can shrink coefficients to exactly zero, L2 regularization reduces the magnitude of coefficients but generally does not make them exactly zero.

**Key Characteristics:**

- **Shrinkage:** L2 regularization helps to prevent overfitting by shrinking the coefficients towards zero. This leads to a model that is less sensitive to fluctuations in the training data.

- **No Feature Selection:** Unlike L1 regularization, L2 regularization does not perform feature selection. All features are retained, but their impact is reduced.

- **Numerical Stability:** L2 regularization often leads to more numerically stable solutions and can be preferable when dealing with multicollinearity (highly correlated features).

#### **Implementation:** 
L2 regularization can be applied to various ML models in scikit-learn by setting the *penalty* parameter as *l2* and adjusting other parameters according to the chosen model.

*Linear Regression:*

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)  # Initialize with regularization strength

```
*Logistic Regression:*

```python

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(penalty='l2')

```
*Support Vector Machine (SVM):*

```python

svm = LinearSVC(penalty='l2')

```

### Elastic Net Regularization

Elastic Net Regularization combines the strengths of both L1 and L2 regularizations. It adds penalties to the loss function for both the absolute value and the square of the magnitude of the coefficients. This approach helps to balance feature selection and coefficient shrinkage, making it suitable for various scenarios.
The Elastic Net loss function is defined as:

$$
\text{Elastic Net Loss} = \text{Loss} + \lambda_1 \sum_{i=1}^{n} |\theta_i| + \lambda_2 \sum_{i=1}^{n} \theta_i^2
$$

Here:
- \(\lambda_1\) is the regularization parameter for L1 (Lasso) regularization.
- \(\lambda_2\) is the regularization parameter for L2 (Ridge) regularization.
- \(\theta_i\) represents the model coefficients.

**Key Characteristics:**

**Combination of L1 and L2 Regularization:** Elastic Net Regularization combines both L1 and L2 penalties, providing a balance between feature selection and coefficient shrinkage. This can be useful when dealing with highly correlated features.

**Feature Selection and Shrinkage:** By using both L1 and L2 penalties, Elastic Net encourages sparsity (like L1) while also keeping all features in the model but with reduced impact (like L2). This makes it versatile in practice.

**Handling Multicollinearity:** Elastic Net can be more effective than L1 or L2 regularization alone when there are many correlated features. It handles multicollinearity better by including a mix of L1 and L2 penalties.

#### **Implementation**

Elastic Net regularization can be applied using the ElasticNet class in scikit-learn by setting the appropriate l1_ratio parameter, which determines the balance between L1 and L2 regularization.

*Linear Regression*

```python

from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)  # Adjust alpha and l1_ratio as needed

```

*Logistic Regression*

```python

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga')  # saga solver supports elasticnet penalty

```
