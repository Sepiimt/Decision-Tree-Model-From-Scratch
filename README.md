
---
![Thumbnail](https://i.postimg.cc/JzhhQqDq/Gemini-Generated-Image-jg0reajg0reajg0r32.png)

---
# **Decision Tree Classifier Implemented From Scratch**

A complete, _fully manual implementation_ of a **binary decision tree classifier** using only NumPy for numerical operations.
No scikit-learn. No helper libraries.
Everything from Gini impurity to recursive node creation is written by hand.

**This project includes:**
	
* Numerical + Categorical splitting
* Weighted Gini decrease
* Custom tree structure
* Evaluation metrics
* Prediction by tree traversal
* `Graphviz` export (`.dot`) for tree visualization
* ROC and PR curve generation

The model is demonstrated on the **Breast Cancer Wisconsin (Diagnostic)** dataset which is available on Kaggle to access..

---
## **1. 🔭 Project Overview**

The goal is to build a working, interpretable decision tree that:
	
* chooses optimal split points
* tracks node statistics
* predicts new samples
* evaluates performance using standard metrics

**_All logic is implemented manually without external ML frameworks. (It is mentionable that only some Graphs and Plots have been created with direct the help of AI)_**

---
## **2. 🗂️Dataset**

The dataset includes `diagnostic` and 30 morphological features such as:
	
* Radius Mean
* Texture Mean / Smoothness Mean / Compactness Mean
* Radius Worst
* Area Worst
* Other clinical descriptors (`etc`)

The target label:
	
* **0 = Benign**
* **1 = Malignant**

The data had no complexity in missing values and cleaning worth mentioning.

#### 📊Here are some Plots and Graphs to visualize and emphasize the different aspects of dataset:   

**Class Distribution**
![Class_Distribution](https://i.postimg.cc/BnNMrqnV/class-distribution.png)


**Correlation Heatmap**
![Correlation_Heatmap](https://i.postimg.cc/CKQJKmDm/correlation-heatmap.png)


**Data Correlation With Diagnosis Results**
![data_correlation_with_diagnosis_results_1](https://i.postimg.cc/CK1mYWZ4/data-correlation-with-diagnosis-results-1.png)
![data_correlation_with_diagnosis_results_2](https://i.postimg.cc/N0MpYWyd/data-correlation-with-diagnosis-results-2.png)


**Feature Distributions**
![feature_distributions](https://i.postimg.cc/FRkpk3Vm/feature-distributions.png)


**Pair-Plot**
![pairplot](https://i.postimg.cc/prDBqvSC/pairplot.png)

After cleaning and encoding, the model is trained on a train/test split.

---
## **3. 🧬Decision Tree Logic**

#### **3.1 Gini Impurity**
Measures how mixed a node is:
$$
Gini = 1 - \sum_{k=1}^{K} p_k^2
$$
Where:
	
* (`p_k`) is the proportion of each class
* (`K = 2`) for binary classification

Examples:
	
* Pure node: (Gini = 0)
* 50/50 split: (Gini = 0.5)


#### **3.2 Weighted Gini After Split**

If a split produces left set (L) and right set (R):
$$
G_{after}
= \frac{|L|}{|L|+|R|} G(L)

* \frac{|R|}{|L|+|R|} G(R)
$$

This ensures large child nodes have more influence.



#### **3.3 Gini Decrease (Split Quality)**
The decision tree tries all features and split thresholds, and chooses the one maximizing:
$$
\Delta G = G_{before} - G_{after}
$$
The selected feature + value is stored inside the node:
```
node.feature_index
node.criteria_value
node.numerical or categorical flag
node.sample_count
node.gini_decrease
```

---
## **4. 🌱Tree Growth (fit)**

A node becomes a leaf when:
	
* Max depth is reached (Max depth can be set through an optional argument in `fit()`, which had been set to 5 by default.)
* All labels are identical
* A split would create an empty child

Leaf prediction is simply the majority class.

Otherwise, the node:
	
1. Evaluates all possible splits
2. Selects the one with minimal Gini impurity
3. Stores its chosen feature/threshold
4. Creates left and right child nodes
5. Recursively calls `fit()` on children

---
## **5. 🎯Prediction**

Prediction is done by recursive tree traversal:
	
* numerical features:
  * go left if `value <= threshold`$$
𝑥
[
𝑓
𝑒
𝑎
𝑡
𝑢
𝑟
𝑒
]
≤
𝑡
ℎ
𝑟
𝑒
𝑠
ℎ
𝑜
𝑙
𝑑
x[feature]≤threshold$$
	
	
* categorical features:
  * go left if the category is in the chosen subset
	
* repeat until a leaf is reached
	
* return the leaf’s label

---
## **6. 🔍Model Evaluation**

![results](https://i.postimg.cc/c4hHYjH3/Gemini-Generated-Image-e4hfoze4hfoze4hf43.png)

### **🧩Metric Explanation**

When evaluating a binary classifier, four numbers lie at the core of everything:
	
* **TP** (True Positive): model predicted *malignant* and was correct
* **TN** (True Negative): model predicted *benign* and was correct
* **FP** (False Positive): predicted malignant for a benign case
* **FN** (False Negative): predicted benign for a malignant case

From your confusion matrix:

| Actual \ Pred | 0  | 1  |
| ------------- | -- | -- |
| **0**         | 85 | 5  |
| **1**         | 3  | 49 |

This maps to:
	
* **TN = 85**
* **FP = 5**
* **FN = 3**
* **TP = 49**

Using these, our metrics are:


#### **1. Accuracy: 0.94366**$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

This measures overall correctness.
It’s fine in balanced datasets, but can mislead when one class dominates.
Ours is fairly balanced, so accuracy is genuinely informative.



#### **2. Precision: 0.9074**$$
Precision = \frac{TP}{TP + FP}
$$

How often the model was *right* when it predicted **malignant**.

Interpretation:
	
* High precision means few *false alarms*.
* Our value (~0.91) means the model rarely mislabels benign tumors as malignant.

For a medical model, good precision avoids unnecessary panic, biopsies, or treatment.



#### **3. Recall: 0.9423**
$$
Recall = \frac{TP}{TP + FN}
$$
How many malignant cases the model *successfully caught*.

Interpretation:
	
* High recall means few *missed cancers*.
* Our recall (~0.94) is strong, meaning the model rarely lets malignant cases slip through.

This is arguably the most important metric in medical detection.



#### **4. F1 Score: 0.9245**
$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$
Harmonic mean of precision and recall.

Interpretation:
	
* A single number summarizing correctness and sensitivity
* Penalizes extreme imbalance between precision and recall
* Your F1 (~0.925) indicates a well-balanced model that both **detects cancer** and **avoids misclassification**



### **❓What the numbers mean in practice**

Our model:
	
* **rarely screams “cancer” when there is none**
* **rarely misses actual cancer**
* **balances caution and correctness extremely well**
* **generalizes strongly despite being a hand-built tree**

This is very solid performance for a scratch implementation with no tricks like pruning or ensembles.


#### 📊Here are some Plots and Graphs to visualize and emphasize the results:

**Feature Importance**
![feature_importance](https://i.postimg.cc/Kz3f3Brd/feature-importance.png)


**PR Curve**
![pr_curve](https://i.postimg.cc/903pd0Lb/pr-curve.png)


**ROC Curve**
![roc_curve](https://i.postimg.cc/QCLbQCf0/roc-curve.png)

---
## **7. 💡Interpretation**

The custom tree:
	
- Successfully discovered meaningful thresholds
	
- Avoided overfitting despite small dataset size
	
- Achieved >94 percent accuracy
	
- Remained fully interpretable (unlike black-box models)

---
## **8. 🧱Tree Structure Visualization (`Graphviz`)**

There is an exported `.dot` file to represent the trained tree structure to DOT format:

```
tree.dot
```

##### **Node Statistics**

Each node stores:
	
* feature index
* split value
* sample count
* Gini decrease

These can be plotted to illustrate which features mattered most.


**Tree Structure**
![tree_graphviz](https://i.postimg.cc/Zq1CfdpM/Gemini-Generated-Image-mert8lmert8lmert.png)

---
## **9. 🔑Key Implementation Features**

* Fully manual tree-building logic
* No dependencies besides NumPy
* Supports both numerical and categorical features
* Stores metadata for visualization
* Produces reproducible ML metrics
* Easily extensible into Random Forests or Gradient Trees

---

