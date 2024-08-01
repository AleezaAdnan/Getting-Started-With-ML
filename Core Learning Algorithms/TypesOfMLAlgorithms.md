# Core Learning Algorithms

There are three main types of machine learning algorithms: Supervised Learning, Unsupervised Learning, and Reinforcement Learning. lets see how they work, and some common algorithms within each category. Let's get started!

## Supervised Learning

 In supervised learning, we have a dataset that includes both input data and the corresponding correct output. The goal is to learn a mapping from inputs to outputs that can be used to make predictions on new, unseen data.

### How It Works
Imagine you're learning to recognize different types of fruits. You have a dataset with images of fruits (input) and labels indicating the type of fruit (output). The supervised learning algorithm analyzes this dataset and learns the relationship between the images and the labels. Once trained, the model can predict the type of fruit in new images it hasn't seen before.

Supervised learning algorithms can be broadly categorized into two types: **regression** and **classification**. Regression algorithms predict a continuous value, such as the price of a house. Classification algorithms, on the other hand, predict discrete labels, such as whether an email is spam or not. You'll learn more about these distinctions in the [Supervised Learning Algorithms](supervised_learning.md) section.

### Common Algorithms
- **Linear Regression**: Used for predicting a continuous value.
- **Decision Trees**: Used for classification and regression tasks.
- **Support Vector Machines (SVM)**: Used for classification tasks.

## Unsupervised Learning

In Unsupervised Learning, You don't have labeled data; instead, you let the algorithm find patterns and structures within the data on its own. The goal is to identify hidden patterns or groupings in the data.

### How It Works
Consider the fruit example again, but this time, you don't have labels for the images. An unsupervised learning algorithm might group the images into clusters based on similarities, such as color or shape, even though it doesn't know what each fruit is called.

### Common Algorithms
- **K-Means Clustering**: Used for grouping data into clusters.
- **Hierarchical Clustering**: Builds a tree of clusters.
- **Principal Component Analysis (PCA)**: Reduces the dimensionality of data.

Learn more about these algorithms in the [Unsupervised Learning](unsupervised_learning.md) section.

## Reinforcement Learning

Reinforcement learning is like training a pet with rewards and penalties. The algorithm learns by interacting with an environment and receiving feedback in the form of rewards or penalties. The goal is to learn a strategy that maximizes cumulative rewards.

### How It Works
Imagine teaching a robot to navigate a maze. The robot receives positive feedback (rewards) for making progress towards the exit and negative feedback (penalties) for hitting walls. Over time, the robot learns the best path to take to successfully navigate the maze.

### Common Algorithms
- **Q-Learning**: A simple reinforcement learning algorithm.
- **Deep Q-Networks (DQN)**: Uses neural networks to approximate the Q-values.

Discover more about these algorithms in the [Reinforcement Learning](reinforcement_learning.md) section.

 Each type has its strengths and is suited for different tasks. Dive into the specific sections to get a detailed understanding of how each algorithm works and how to implement them.
