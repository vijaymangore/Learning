# Confusion Matrix
Confusion Matrix
What is it?

A confusion matrix is a table that compares a classification model’s predictions with the actual labels. It shows not just how often the model is wrong, but how it is wrong. It’s called “confusion” because it shows where the model confuses one class for another.

The 2×2 layout (binary classification)
	Predicted Positive	Predicted Negative
Actual Positive	TP (True Positive)	FN (False Negative)
Actual Negative	FP (False Positive)	TN (True Negative)
TP: predicted positive, and it really was positive (correct)
TN: predicted negative, and it really was negative (correct)
FP: predicted positive, but it was actually negative (a Type I error, a “false alarm”)
FN: predicted negative, but it was actually positive (a Type II error, a “miss”)

Memory trick: the second word (Positive/Negative) is what the model predicted. The first word (True/False) says whether that prediction was right.

A worked example

A model screens 100 patients for a disease. 10 actually have it.

	Predicted Sick	Predicted Healthy
Actually Sick (10)	TP = 8	FN = 2
Actually Healthy (90)	FP = 5	TN = 85

The model caught 8 of 10 sick patients, missed 2, falsely alarmed on 5 healthy people, and correctly cleared 85.

Metrics derived from the matrix
Metric	Formula	Example	Meaning
Accuracy	(TP+TN) / total	93/100 = 0.93	Overall fraction correct
Precision	TP / (TP+FP)	8/13 = 0.615	When the model says positive, how often is it right?
Recall (Sensitivity, TPR)	TP / (TP+FN)	8/10 = 0.80	Of all real positives, how many did we catch?
Specificity (TNR)	TN / (TN+FP)	85/90 = 0.944	Of all real negatives, how many did we correctly clear?
False Positive Rate	FP / (FP+TN)	5/90 = 0.056	Fraction of negatives wrongly flagged
F1 score	2·P·R / (P+R)	≈ 0.696	Harmonic mean of precision and recall
NPV	TN / (TN+FN)	85/87 = 0.977	When the model says negative, how often is it right?

Reading the example: accuracy looks great at 93%, but precision is only 61.5%. Of the 13 people flagged as sick, about 38% were healthy.

Why accuracy alone can mislead

Imagine a lazy model that predicts “healthy” for everyone:

TP = 0, FN = 10, FP = 0, TN = 90
Accuracy = 90%, which looks good
Recall = 0%, since it catches no sick patients at all

With imbalanced classes (fraud, disease, defects), accuracy hides failure. The confusion matrix exposes it immediately.

Precision vs recall: which matters more?

They trade off against each other. Raising the decision threshold makes the model more cautious, so precision rises and recall falls. Lowering it does the opposite.

Prioritize recall when misses are costly: cancer screening, fraud detection, security threats. You’d rather have false alarms than miss a real case.
Prioritize precision when false alarms are costly: spam filters (don’t bin real email), recommending content, legal flags.
Use F1 when you need a balance and classes are imbalanced.
Where thresholds come in

Most models output a probability, and you choose a threshold (often 0.5) to convert it into a class. Each threshold produces a different confusion matrix. The ROC curve plots TPR vs FPR across all thresholds, and the Precision-Recall curve plots precision vs recall across all thresholds. Both are summaries of many confusion matrices.

Multiclass confusion matrices

For N classes, the matrix is N×N. Rows are actual classes, columns are predicted classes (in scikit-learn), and the diagonal holds correct predictions.

Example with 3 classes (Cat, Dog, Bird):

	Pred Cat	Pred Dog	Pred Bird
Actual Cat	40	8	2
Actual Dog	5	42	3
Actual Bird	1	2	47

This tells you, for example, that the model often confuses cats with dogs (8 cats predicted as dogs), which is useful for targeted improvement.

Per-class metrics are computed “one vs rest.” For Cat: TP = 40, FN = 8+2 = 10, FP = 5+1 = 6. Then you combine them:

Macro average: average the per-class metric equally (treats all classes the same)
Micro average: pool all TP/FP/FN globally (dominated by large classes)
Weighted average: average weighted by class size
Python example
python
from sklearn.metrics import (confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

y_true = [1]*10 + [0]*90
y_pred = [1]*8 + [0]*2 + [1]*5 + [0]*85

cm = confusion_matrix(y_true, y_pred)
print(cm)
# [[85  5]
#  [ 2  8]]   <- scikit-learn order: rows = actual, cols = predicted, class 0 first

tn, fp, fn, tp = cm.ravel()
print("Precision:", tp / (tp + fp))
print("Recall:   ", tp / (tp + fn))

print(classification_report(y_true, y_pred))
ConfusionMatrixDisplay(cm).plot()
plt.show()

Watch out: scikit-learn puts the negative class (0) first, so the layout is [[TN, FP], [FN, TP]], which is different from the textbook layout above. Many people misread this.

Other useful metrics from the matrix
Balanced accuracy: (Recall + Specificity) / 2. Here it’s about 0.872, a fairer score for imbalanced data.
Matthews Correlation Coefficient (MCC): a single score in [−1, 1] that uses all four cells and stays reliable under imbalance.
Cohen’s kappa: agreement between predictions and truth, corrected for chance.
Common pitfalls
Mixing up row/column orientation between libraries (always check the docs)
Trusting accuracy on imbalanced data
Evaluating on training data instead of a held-out test set
Ignoring that the matrix depends on the chosen threshold
Ignoring the real-world cost of each error type. FP and FN rarely cost the same.
Summary

A confusion matrix breaks predictions into TP, TN, FP, and FN. From it you derive accuracy, precision, recall, specificity, and F1, each answering a different question about model behavior. Look at it whenever you evaluate a classifier, especially with imbalanced classes, because it tells you which mistakes your model makes, and that is what guides improvement.

If you’d like, I can show how changing the threshold shifts the matrix, or walk through ROC and AUC next.
