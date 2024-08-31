# Hyperparameter Tuning

Hyperparameter tuning is the process of finding the best set of hyperparameters for your machine learning model. Hyperparameters are settings that you define before training, such as the learning rate or the number of layers in a neural network. The right combination of hyperparameters can significantly improve your model's performance


### Why Hyperparameter Tuning is Important

Choosing the right hyperparameters distinguishes a model that performs well and one that doesn’t. Since there's no one-size-fits-all approach, tuning allows to explore different combinations to find the most effective ones.

## Basic Techniques for Hyperparameter Tuning

Below are two most common and begginer friendly aproaches for hyperparameter tuning

### 1. Grid Search

Grid Search tests every possible combination of a set of hyperparameters. While it can be computationally expensive, it’s a straightforward way to ensure you explore all potential options.

#### Example

For tuning a simple decision tree classifier. We want to tune the `max_depth` and `min_samples_split` parameters:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()

# Define the hyperparameters and their values to try
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(DT, param_grid, cv=5)

grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)

```


**Computationally Expensive:** Evaluating every possible combination can be time-consuming, especially for large search spaces.


#### **When to Use Grid Search?**
- When you have a small set of hyperparameters or a simple model.
- When you have enough computational resources to explore all possible combinations.

### 2. Random Search

Random Search randomly selects combinations of hyperparameters from a defined range. This method is more efficient than Grid Search because it doesn't try every combination, but it still explores a wide range of options.

#### Example

Here’s how you can implement Random Search for the same decision tree classifier:

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np

DT = DecisionTreeClassifier()

param_dist = {
    'max_depth': np.arange(3, 11),
    'min_samples_split': np.arange(2, 11)
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5)

random_search.fit(X_train, y_train)

print("Best parameters found: ", random_search.best_params_)

```

**No Guarantee of Best Result:** Since it samples randomly, it might miss the optimal combination of hyperparameters. The quality of the results depends on how well the ranges are defined.

#### **When to Use Random Search?**
- When the search space is large, and you need a more efficient approach than Grid Search.
- When you have limited computational resources and want a good-enough solution quickly.

### Tips for Effective Hyperparameter Tuning

Regardless of the method you choose, here are some tips for effective hyperparameter tuning: 

- **Start with Default Parameters:** Before tuning, understand how your model performs with default parameters. This gives you a baseline to compare against.

- **Tune the Most Impactful Parameters First:** Focus on the hyperparameters that most significantly affect your model’s performance.

- **Use Cross-Validation:** Always use cross-validation to evaluate your models during tuning. This ensures that your results are not biased by the specific train-test split.

- **Iterate in Phases:**

    Phase 1: Start with a broad search using Random Search or a coarse Grid Search.
    Phase 2: Narrow down the search space based on the results and perform a more detailed search.

By following these tips, you can make your tuning process more efficient and effective.
