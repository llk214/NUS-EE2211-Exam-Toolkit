# EE2211 Machine Learning Exam Toolkit

A comprehensive, exam-ready Python toolkit covering all major topics in NUS EE2211 (Introduction to Machine Learning). Designed for open-book exams where pre-written code is allowed.

##  Features

- **Single-file design**
- **Interactive menu**
- **Step-by-step output**
- **Flexible input**
- **Exam-tested**

##  Requirements

- Python 3.7+
- NumPy

```bash
pip install numpy
```

##  Quick Start

```bash
python EE2211_ALL.py
```

##  Input Format Guide
- Exam tested fast input
- Familiarize before exam

### Matrix Input (comma-separated rows)
```
1 2 3, 4 5 6, 7 8 9    →    [[1,2,3], [4,5,6], [7,8,9]]
```

### Vector Input (space-separated)
```
1 2 3 4 5    →    [1, 2, 3, 4, 5]
```

### One-hot Labels
```
1 0 0, 0 1 0, 0 0 1    →    Classes 0, 1, 2
```

### Threshold Input (Decision Trees)
```
0: 1.5 2.5, 1: 3.0     →    Feature 0 uses [1.5, 2.5], Feature 1 uses [3.0]
1.5 2.5                →    All features use [1.5, 2.5]
3                      →    Single threshold for single-feature data
```

## Main menu:
```
╔══════════════════════════════════════════════════════════════════════════╗
║                     EE2211 EXAM TOOLKIT (All-in-One)                     ║
║  Logistic | Clustering | Regression | Neural Net | Trees | Optimizer     ║
╚══════════════════════════════════════════════════════════════════════════╝

1. Classification (Binary / Multiclass)
2. Clustering (K-Means / Fuzzy C-Means)
3. Linear & Softmax GD (Train W with GD iteratively)
4. Neural Network (MLP Forward & Backprop)
5. Regression (OLS / Ridge / Polynomial)
6. Decision Tree & Random Forest (Classification + Regression)
7. Cost Function Minimizer (Custom Formula)
0. Exit
```

##  Modules & Exam Question Types

### 1. Classification (Logistic Regression)
**Use for:** Binary/multiclass classification, sigmoid/softmax, cross-entropy loss

- Binary logistic regression with configurable threshold
- Multiclass softmax regression
- SGD, mini-batch, and full batch training
- L2 regularization with optional bias penalization
- Confusion matrix, precision, recall, F1 metrics

**Example input:**
```
X rows = 1 2, 3 4, 5 6
y = 0 1 1
```

### 2. Clustering (K-Means / Fuzzy C-Means)
**Use for:** K-means iterations, centroid updates, distortion J, FCM membership

- K-Means with manual or K-Means++ initialization
- Fuzzy C-Means (textbook version) with membership matrix
- Shows each iteration's centroids, labels, and objective

**Example (from AY21/22 Q9 - Book clustering):**
```
X = 45, 51, 60, 62, 75, 80, 85, 90
C0 = 45, 90
```
Output shows iteration-by-iteration centroids and assignments.

### 3. Linear & Softmax Gradient Descent
**Use for:** Matrix-form gradient descent, weight updates, MSE cost

- Linear regression with matrix operations
- Softmax regression for multiclass classification
- Shows Z (logits), P (probabilities), gradients at each iteration
- Configurable 1/N scaling

### 4. Neural Network (MLP)
**Use for:** Forward/backward propagation, weight updates, activation functions

- Configurable layers (1-5)
- Activation: ReLU, Sigmoid, Softmax, Linear
- Weight init: Zeros, Xavier, He, Random, Manual
- MSE and Cross-entropy loss
- Shows full forward pass and updated weights

### 5. Regression (OLS/Ridge/Polynomial)
**Use for:** Least squares, ridge regression, polynomial features, R², Pearson r

- OLS with automatic intercept
- Ridge regression with λ parameter
- Polynomial feature expansion
- Metrics: MSE, RMSE, MAE, R², Adjusted R², Pearson r

### 6. Decision Tree & Random Forest
**Use for:** Gini/entropy, information gain, MSE splits, tree building

- Classification trees (Gini, Entropy)
- Regression trees (MSE)
- Random Forest with bootstrap
- **Mode 1: Best split analysis** — Perfect for exam questions asking for parent/child impurity
- Manual threshold input for specific split points

**Example (from AY21/22 Q6 - Regression tree MSE):**
```
X = 0.2, 0.7, 1.8, 2.2, 3.7, 4.1, 4.5, 5.1, 6.3, 7.4
y = 2.1 1.5 5.8 6.1 9.1 9.5 9.8 12.7 13.8 15.9
Task = 2 (Regression)
Thresholds = 3
Mode = 1 (best split only)
```
Output:
```
Parent MSE: 20.638
Best Split: Feature index 0, Threshold 3.0
Child MSE: 5.565
```

### 7. Cost Function Minimizer (Generic Gradient Descent)
**Use for:** Minimizing arbitrary cost functions C(w), C(x,y), C(a,b,c), etc.


### Cost Function Syntax



| Math | x² | xy | 2x | sin(w) | cos(w) | eˣ | ln(x) | √x | π |
|------|----|----|----| -------|--------|-----|-------|-----|---|
| Python | `x**2` | `x*y` | `2*x` | `math.sin(w)` | `math.cos(w)` | `math.exp(x)` | `math.log(x)` | `math.sqrt(x)` | `math.pi` |

### ⚠️ Common Mistakes

| ❌ Wrong | `x^2` | `xy` or `2x` | `sin(w)` |
|----------|-------|--------------|----------|
| ✅ Correct | `x**2` | `x*y` or `2*x` | `math.sin(w)` |

### Examples by Mode

**Mode 1 — C(w):**
```
math.sin(w)**2          # sin²(w)
w**2 + 2*w + 1          # w² + 2w + 1
```

**Mode 2 — C(x,y):**
```
x**2 + x*y**2           # x² + xy²
math.sin(x) + math.cos(y)   # sin(x) + cos(y)
```

**Mode 3 — C(a,b,c):**
```
a**2 + b**2 + c**2
```

**Mode 4 — Custom variables:**
```
Expression: x1**2 + x2**2 + x3**2
Variables:  x1 x2 x3
```


**Example (from AY21/22 Q3):**
```
Mode: 1
C(w) = math.sin(w)**2
initial w = 3
lr = 0.1
iters = 1
```
Output: `w=3.028, grad=-0.279`



## ✅ Verified Against Past Papers


##  File Structure

```
EE2211_ALL.py          # Main toolkit (single file, ~1600 lines)
README.md              # This file
LICENSE                # MIT License
```

##  Contributing

Found a bug or have improvements? Feel free to open an issue or PR.

##  License

MIT License — Free to use, modify, and distribute.

## 🙏 Acknowledgments

Created for NUS EE2211, EE2213 students. Good luck with your exams!

---

*Last tested: AY2024/2025 Semester 1*
