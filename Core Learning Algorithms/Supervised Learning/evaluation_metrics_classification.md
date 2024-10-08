# Classification Metrics

Choosing the right evaluation metric is crucial in ensuring your model's performance meets real-world needs. Metrics like accuracy may suffice for balanced datasets, but in cases like medical diagnoses or fraud detection, precision, recall, and other advanced metrics become vital

### For Binary Classification

## Confusion Matrix

The confusion matrix is a fundamental evaluation metric for classification problems. It helps us understand how well our model performs by comparing the predicted classes with the actual classes. It is a 2x2 table for binary classification, representing the counts of true and false classifications.

Here’s a breakdown of the terms used in a confusion matrix:

<p align="center">
  <img src="../../Core Learning Algorithms/images/cnf matrix.png" alt="Confusion Matrix">
</p>

## Key Terms

- **True Positive (TP)**: When the model correctly predicts the positive class.  
  Example: Predicting a patient has a disease when they actually do.

- **True Negative (TN)**: When the model correctly predicts the negative class.  
  Example: Predicting a patient does not have a disease when they actually don't.

- **False Positive (FP)**: Also known as a "Type I Error." This happens when the model incorrectly predicts the positive class.  
  Example: Predicting a patient has a disease when they actually don't.

- **False Negative (FN)**: Also known as a "Type II Error." This happens when the model incorrectly predicts the negative class.  
  Example: Predicting a patient does not have a disease when they actually do.

## Why is the Confusion Matrix Important?

The confusion matrix provides detailed insights beyond just the overall accuracy of a model. By looking at TP, TN, FP, and FN, you can calculate other important metrics such as:

- **Accuracy**
- **Precision**
- **Recall (Sensitivity)**

Understanding the confusion matrix helps you choose the right model for your needs and fine-tune it accordingly.


# Evaluation Metrics for Classification

After understanding the confusion matrix, we can now explore the key metrics used to evaluate classification models. Each metric gives insight into a different aspect of performance, helping us determine how well a model fits our data.

---

## 1. Accuracy

**Accuracy** measures how often the classifier is correct overall. It’s the ratio of correct predictions (both positives and negatives) to the total number of predictions.

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

### When Accuracy Works Well:
Accuracy works best when the classes in your dataset are balanced (roughly the same number of positive and negative instances).

### When Accuracy Fails:
In cases of imbalanced data (e.g., a rare disease where 95% of patients are healthy and 5% are sick), accuracy can be misleading. A model that predicts all patients as healthy will have high accuracy (95%), even though it completely misses the sick patients.

---

## 2. Precision

**Precision** focuses on the positive predictions. It tells us how many of the predicted positives are actually positive.

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

### When Precision Works Well:
Precision is useful when the cost of a false positive is high. For example, in spam detection, you'd want high precision to avoid marking legitimate emails as spam.

### When Precision Fails:
When you care more about catching all actual positives, precision might not be the best metric, as it doesn't account for false negatives.

---

## 3. Recall (Sensitivity)

**Recall** (or sensitivity) measures how many actual positives were correctly identified. It’s the ratio of true positives to the total actual positives.

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

### When Recall Works Well:
Recall is crucial when missing positive instances is costly, such as in medical diagnoses where you want to ensure all patients with a disease are correctly identified.

### When Recall Fails:
Focusing solely on recall can lead to many false positives. For example, predicting every patient as having a disease will result in perfect recall, but very low precision.


## The Trade-off: Precision vs. Recall

There’s often a trade-off between precision and recall. If you increase recall, you may lower precision, and vice versa. For example, in a medical test, making the test more sensitive (higher recall) may result in more false alarms (lower precision).

---

## 4. F1 Score

The **F1 score** combines both precision and recall into a single metric. It’s the harmonic mean of precision and recall, giving more weight to lower values.

\[
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

### When F1 Score Works Well:
The F1 score is useful when you want a balance between precision and recall, especially in situations with imbalanced data.

### When F1 Score Fails:
In some cases, you might prioritize precision or recall over their balance. In these situations, using the F1 score might obscure important details about your model's performance.

---

## 5. Specificity (True Negative Rate)

**Specificity** measures how well the model identifies true negatives. It’s the proportion of actual negatives that are correctly identified as negative.

\[
\text{Specificity} = \frac{TN}{TN + FP}
\]

### When Specificity Works Well:
Specificity is important in cases where false positives are costly, such as a legal system where falsely accusing someone is worse than missing a few guilty individuals.

### When Specificity Fails:
If you only focus on specificity, you might miss too many positive cases. For instance, a model that rarely flags people as positive may have high specificity, but low recall.

---

## 6. ROC Curve and AUC Score

The **ROC curve** (Receiver Operating Characteristic curve) is a graphical representation that illustrates the performance of a classification model at various threshold levels. It plots **True Positive Rate (Recall)** against **False Positive Rate (1 - Specificity)**, giving us an idea of the trade-off between catching positive instances and avoiding false alarms.

The **AUC score** (Area Under the ROC Curve) quantifies the overall performance of the classifier. A higher AUC (closer to 1) indicates better model performance, while an AUC of 0.5 suggests a model no better than random guessing.

### When ROC-AUC Works Well:
ROC-AUC is particularly helpful when dealing with **imbalanced datasets**, as it provides a broader view of model performance across different decision thresholds. In such cases, other metrics like accuracy might be misleading because they do not account for the balance between true positives and false positives.

- **Example**: In a fraud detection system where only 1% of transactions are fraudulent, using accuracy as a metric might not reveal much. A model could achieve 99% accuracy simply by predicting every transaction as non-fraudulent. However, the ROC-AUC score would give a clearer picture, showing how well the model distinguishes between fraudulent and non-fraudulent transactions at various thresholds.

By considering both the **true positive rate** and the **false positive rate**, ROC-AUC evaluates a model's ability to distinguish between classes regardless of the class distribution. This makes it particularly powerful for imbalanced datasets where one class heavily outweighs the other.

### When ROC-AUC Fails:
In highly imbalanced datasets, even models with a high AUC score might not perform well in terms of **real-world impact**. This is because the AUC focuses on the model's ability to distinguish between classes across all thresholds, but it doesn't directly reflect the **practical consequences** of false positives and false negatives.

- **Example**: In the same fraud detection system, a model might have a high AUC score (e.g., 0.95), meaning it is good at distinguishing between fraud and non-fraud across thresholds. However, if the threshold is not set appropriately, the model might still miss most of the actual fraud cases (low recall) or flag too many legitimate transactions as fraudulent (low precision), which could harm customer experience or miss critical fraudulent activities.

Thus, while ROC-AUC gives an overall view of performance, it may **fail** to address the **operational needs** of specific problems, especially when the cost of false positives or false negatives is significantly unbalanced. In such cases, focusing on metrics like precision, recall, or F1 score may be more appropriate, depending on the business requirements

---

## 7. Matthews Correlation Coefficient (MCC)

The **Matthews Correlation Coefficient (MCC)** is a balanced measure that evaluates the quality of binary classifications. It takes into account all four categories from the confusion matrix: true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). The MCC ranges from -1 to +1, where:

- +1 indicates a perfect prediction.
- 0 indicates a random prediction.
- -1 indicates total disagreement between actual and predicted values.

### Formula

\[
\text{MCC} = \frac{(TP \times TN) - (FP \times FN)}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
\]

This formula gives a comprehensive measure by balancing the positive and negative classes, making it especially useful for imbalanced datasets.

### When MCC Works Well:
MCC is particularly beneficial when the dataset is **imbalanced**, meaning the number of positive and negative instances is significantly different. Unlike accuracy, MCC considers the balance between all the classes and handles imbalanced data better.

- **Example**: In a medical diagnosis task where only 1% of the cases are positive (disease detected), accuracy might be misleading if a model predicts "no disease" in almost every case. MCC, however, accounts for both correct and incorrect predictions in each class, providing a more accurate reflection of the model’s performance, even with unbalanced data.

### When MCC Fails:
While MCC is generally more robust than accuracy for imbalanced datasets, it can still fail in cases where the cost of false positives and false negatives has **significantly different importance**. MCC treats all types of errors equally, so if you care more about one type of error (e.g., missing a fraudulent transaction), a metric like precision or recall may be more relevant.

- **Example**: In a fraud detection system where false positives (flagging legitimate transactions as fraudulent) are not as costly as false negatives (missing actual fraud), MCC would treat both errors the same. In this case, you might prioritize a metric like **recall** to catch as many fraudulent transactions as possible.
 
### MCC vs. F1 Score:
- **F1 Score** balances precision and recall, focusing on the positive class.
- **MCC** gives a balanced evaluation by considering all aspects of the confusion matrix, including true negatives.

MCC is ideal when you need a single number to summarize the overall performance of the model across all classes.

---
---

# Multiclass Classification

In multiclass classification, the task is to classify instances into one of several categories or classes. This requires special handling of evaluation metrics as opposed to binary classification.

## Confusion Matrix for Multiclass Classification

For multiclass classification, the confusion matrix is an extension of the 2x2 matrix used in binary classification. It becomes an **NxN matrix**, where `N` is the number of classes. Each row corresponds to the **actual class**, and each column corresponds to the **predicted class**.

For example, if there are three classes (A, B, C), the confusion matrix would look like this:

<p align='center'>
<table>
  <tr>
    <th></th>
    <th>Predicted A</th>
    <th>Predicted B</th>
    <th>Predicted C</th>
  </tr>
  <tr>
    <td><strong>Actual A</strong></td>
    <td>TP (A)</td>
    <td>FP (B)</td>
    <td>FP (C)</td>
  </tr>
  <tr>
    <td><strong>Actual B</strong></td>
    <td>FN (A)</td>
    <td>TP (B)</td>
    <td>FP (C)</td>
  </tr>
  <tr>
    <td><strong>Actual C</strong></td>
    <td>FN (A)</td>
    <td>FN (B)</td>
    <td>TP (C)</td>
  </tr>
</table>
</p>

In this matrix:
- Diagonal elements (e.g., TP (A), TP (B), TP (C)) represent the **True Positives (TP)** for each class.
- Off-diagonal elements represent **False Positives (FP)** and **False Negatives (FN)**, which indicate misclassifications.

This confusion matrix helps in identifying how well the model performs for each class and where it misclassifies instances.

---

# Evaluation Metrics for Multiclass Classification

To evaluate multiclass models, we can extend the binary classification metrics like **precision**, **recall**, and **F1 score** using two common approaches:
- **Micro Averaging**
- **Macro Averaging**


## Micro Average

**Micro Average** aggregates the contributions of all classes to compute the average metric. It's useful when you care equally about all classes and want to get a global measure.

### How It Works

1. **Calculate metrics for each instance**: For each class, count the number of true positives, false positives, and false negatives.
2. **Sum up these counts** across all classes.
3. **Compute the metric** using the total counts.

### Formula

For precision, recall, and F1-score:

- **Micro Precision**: 
  \[
  \text{Micro Precision} = \frac{\text{Total TP}}{\text{Total TP} + \text{Total FP}}
  \]

- **Micro Recall**: 
  \[
  \text{Micro Recall} = \frac{\text{Total TP}}{\text{Total TP} + \text{Total FN}}
  \]

- **Micro F1-Score**: 
  \[
  \text{Micro F1-Score} = \frac{2 \times \text{Micro Precision} \times \text{Micro Recall}}{\text{Micro Precision} + \text{Micro Recall}}
  \]

### Drawbacks

- **Class Imbalance**: Micro average can be dominated by the performance on the more frequent classes.
- **Loses Class-specific Insights**: It combines all classes, so you may miss out on how the model performs on individual classes.

## Macro Average

**Macro Average** computes the metric for each class individually and then averages these values. It's useful when you want to treat all classes equally, regardless of their size.

### How It Works

1. **Calculate metrics for each class**: Compute precision, recall, and F1-score for each class.
2. **Average these metrics** across all classes.

### Formula

- **Macro Precision**: 
  \[
  \text{Macro Precision} = \frac{1}{C} \sum_{i=1}^{C} \text{Precision}_i
  \]

- **Macro Recall**: 
  \[
  \text{Macro Recall} = \frac{1}{C} \sum_{i=1}^{C} \text{Recall}_i
  \]

- **Macro F1-Score**: 
  \[
  \text{Macro F1-Score} = \frac{1}{C} \sum_{i=1}^{C} \text{F1-Score}_i
  \]

Where \( C \) is the number of classes.

### Drawbacks

- **Sensitive to Class Imbalance**: Macro average treats all classes equally, which can be misleading if some classes are very small.
- **May Not Reflect Overall Performance**: Classes with very few instances might disproportionately affect the average.

## Weighted Average

**Weighted Average** takes into account the number of true instances for each class when calculating the average metric. It provides a more balanced view considering the size of each class.

### How It Works

1. **Calculate metrics for each class**: Compute precision, recall, and F1-score for each class.
2. **Weight these metrics** by the number of true instances of each class.
3. **Sum and normalize** these weighted metrics.

### Formula

- **Weighted Precision**: 
  \[
  \text{Weighted Precision} = \frac{\sum_{i=1}^{C} (\text{Precision}_i \times \text{Support}_i)}{\sum_{i=1}^{C} \text{Support}_i}
  \]

- **Weighted Recall**: 
  \[
  \text{Weighted Recall} = \frac{\sum_{i=1}^{C} (\text{Recall}_i \times \text{Support}_i)}{\sum_{i=1}^{C} \text{Support}_i}
  \]

- **Weighted F1-Score**: 
  \[
  \text{Weighted F1-Score} = \frac{\sum_{i=1}^{C} (\text{F1-Score}_i \times \text{Support}_i)}{\sum_{i=1}^{C} \text{Support}_i}
  \]

Where \(\text{Support}_i\) is the number of true instances of class \(i\).

### Drawbacks

- **Complexity**: Weighted average can be more complex to compute and interpret compared to macro and micro averages.
- **Overemphasis on Large Classes**: It might overemphasize the performance on larger classes and potentially underrepresent smaller classes.


