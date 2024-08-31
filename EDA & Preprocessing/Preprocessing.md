
# Data Preprocessing & Cleaning


Data preprocessing is like preparing a canvas before painting.

It involves the transformation of the raw dataset into an understandable format. Raw data is often messy, containing missing values, outliers, and inconsistencies that can negatively impact the performance of machine learning models. That's where data preprocessing comes in – a crucial step in the machine learning pipeline that involves cleaning and preparing the data to ensure it's in a suitable format for analysis.

## Steps In Data Preprocessing:  

>- Gathering the data  
>- Import the dataset & Libraries   
>- Dealing with Missing Values  
>- Dealing with Categorical values   
>- Divide the dataset into Dependent & Independent variable   
>- Split the dataset into training and test set  
>- Feature Scaling  
>- Handling Outliers 

### Step 1 + 2: Gathering Data & Importing Libraries 

Dataset entirely depends on what type of problem you want to solve. Each problem in machine learning has its own unique approach.

Here are some website with you to get the dataset :

1. Kaggle: Kaggle is my personal favorite one to get the dataset. https://www.kaggle.com/datasets
    
2. UCI Machine Learning Repository: One of the oldest sources on the web to get the dataset. http://mlr.cs.umass.edu/ml/

3. This awesome GitHub repository has high-quality datasets.
https://github.com/awesomedata/awesome-public-datasets  

4. And if you are looking for Government’s Open Data then here is few of them:
>- Indian Government: http://data.gov.in  
>- US Government: https://www.data.gov/  
>- British Government: https://data.gov.uk/  
>- France Government: https://www.data.gouv.fr/en/

After figuring out the dataset you want to use, the next step is to import all the essential libraries you need. 
 
```python: 
import numpy as np
import pandas as pd 
# Load the dataset
data = pd.read_csv('your_dataset.csv')
```
Now you need to understand your data. To do so you can perform certain operations listed below to develop a better idea of the dataset you are dealing with: 

- The dimensions of the dataset (rows and columns).
- The types of features present (numerical, categorical, text, etc.).
- The presence of any missing values.
- Distribution of the target variable (for supervised learning tasks).

### Step 3: Handling Missing Values

Missing values can wreak havoc on your models.

There are several approaches to deal with missing values:
- **Removal**: Remove rows or columns with missing values. However, this should be done with caution as it may result in a loss of valuable information.
- **Imputation**: Fill in missing values using various techniques such as mean, median, mode, or advanced imputation methods like K-nearest neighbors.

With the help of _info()_ we can found total number of entries as well as count of non-null values with datatype of all features.

we also can use _dataset.isna()_ to see the of null values in our dataset.

```
data.info()

# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)
```

We can use _dropna()_ to remove all the rows with missing data.

But this is not always a good idea. Instead, for replacing null values we can use the strategy that can be applied on a feature which has numeric data. We can calculate the _Mean, Median or Mode_ of the feature and replace it with the missing values.

```
# Drop rows with missing values
cleaned_data = data.dropna()

# Impute the missing values with mean
data["Your feature"].fillna(data["Your Feature"].mean(), inplace=True)
data["Your Feature"].fillna(data["Your Feature"].mean(), inplace=True)

# Impute missing categorical values with mode
data["Gender"].fillna(data["Gender"].mode()[0], inplace=True)
```
### Step 4: Dealing with Categorical Variables

Now let’s see how to deal with categorical values.  

Machine learning models typically work with numerical data, so we need to convert categorical data into numerical form.

The library that we are going to use for the task is called Scikit Learn.preprocessing. There’s a class in the library called LabelEncoder which we will use for the encoding.

```
from sklearn.preprocessing import LabelEncoder
```

The next step is usually to create an object of that class. We will call our object _lEncoder_.

```
lEncoder = LabelEncoder()
```

Now to convert this into numerical we can use following code:

```
variable_storing_feature.iloc[select the row & coloum = 
lEncoder.fit_transform(variable_storing_feature.iloc[select the coloum])
```
> (Consult the notebook for a better understanding)

The categorical values will be encoded. But there’s a problem!

The problem is still the same. Machine learning models are based on equations and it’s good that we replaced the text by numbers. However, numbers still have precedence over one another in equations and in our case that is not what we might want. Hence, to avoid this we use _**Dummy variables**_.

Dummy Variables is one that takes the value 0 or 1 to indicate the absence or presence of some categorical effect that may be expected to shift the outcome.

That means instead of having one column here we are going to have multiple columns for each categories and having 1 and 0 for their values.

Number of Columns = Types of Categories

_**OneHotEncoder**_ has been used in the example notebook where we are applying all these concepts.

### Step 5: Divide the dataset into training and test set 

The next step would be to identify the independent variable (X) and the dependent variable (Y).

Basically dataset might be labeled or unlabeled.

To read the columns, we will use iloc of pandas (used to fix the indexes for selection) which takes two parameters — [row selection, column selection].

You will store the features in a variable to manipulate and perform operations on them.
Refer to the notebook to get clarity.



### Step 6: Split the Dataset into training and test set

In machine learning we usually splits the data into Training and Testing data for applying models.

Generally we split the dataset into 70:30 or 80:20 (as per the requirement)it means, 70 percent data taken to train and 30 percent data taken to test.

For this task, we will import _train_test_split_ from _model_selection_ library of scikit.

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 7: Feature Scaling 

Imagine comparing apples to oranges — not so easy, right? The same applies to machine learning features with differing scales.

Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. 

**Why Scaling** :- Most of the times, your dataset will contain features highly varying in magnitudes, units and range. But since, most of the machine learning algorithms use Euclidean distance between two data points in their computations, this is a problem.

There are so many situations where Feature Scaling is optional or not required.

**Min-Max Scaling**:
Min-Max scaling brings features within a specified range, often 0 to 1. Let’s see how it’s done using Python and scikit-learn’s “MinMaxScaler.”

```
from sklearn.preprocessing import MinMaxScaler
# Initialize the MinMaxScaler
scaler = MinMaxScaler()
# Fit and transform the data
normalized_data = scaler.fit_transform(data[['Age', 'Salary']])
print(normalized_data)
```
Min-Max scaling ensures proportional scaling within the defined range.


**Standard Scaling** (Z-score Normalization): 
Z-score normalization, or Standard Scaling, standardizes features with a mean of 0 and a standard deviation of 1.

```
from sklearn.preprocessing import StandardScaler
# Initialize the StandardScaler
scaler = StandardScaler()
# Fit and transform the data
scaled_data = scaler.fit_transform(data[['Age', 'Salary']])
print(scaled_data)
```
Z-score normalization is particularly useful when data doesn’t adhere to a specific range, ideal for various machine learning algorithms.

### Step 8: Handling Outliers

An outlier is a data point that significantly deviates from the rest of the data. It can be either much higher or much lower than the other data points

Outliers can significantly impact model performance. You can visualize them using box plots and handle them using various techniques like truncation or capping.

Here's a link to explore this topic further in detail:  
https://www.geeksforgeeks.org/machine-learning-outlier/


## Conclusion
Data preprocessing isn’t just a box you check before diving into machine learning — it’s a crucial journey that transforms raw data into a masterpiece. By handling missing values, normalizing data, and scaling features, you’re paving the way for models that can uncover insights, predict outcomes, and drive innovation.

