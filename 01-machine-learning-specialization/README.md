# Week 2 — Regression with Multiple Input Variables

> Course 1: Supervised Machine Learning: Regression and Classification

## 📚 Course

Machine Learning Specialization — Course 1

## 📅 Week

Week 2

## 🎯 Main Topic

Multiple Linear Regression

---

## 🧠 What I Learned

In Week 2, I learned how to extend linear regression from using a
single input variable to using multiple input variables.

Instead of predicting an output using only one feature:

x → y

we can use multiple features:

x₁, x₂, x₃, ..., xₙ → y

This allows the model to use multiple pieces of information to make
a prediction.

---

## 🔑 Key Concepts

- Multiple linear regression
- Multiple input features
- Feature vector
- Parameters (weights and bias)
- Vectorization
- Matrix multiplication
- Cost function
- Gradient descent
- Feature scaling
- Learning rate
- Iterations
- Convergence

---

## 📐 Mathematical Model

For a single input variable:

ŷ = wx + b

For multiple input variables:

ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

In vector notation:

ŷ = w · x + b

Where:

- x = input feature vector
- w = parameter/weight vector
- b = bias
- ŷ = predicted value

---

## 🔄 How the Model Works

The basic learning process is:

Input Features
      ↓
Linear Prediction
      ↓
Calculate Error
      ↓
Calculate Cost
      ↓
Calculate Gradients
      ↓
Update Parameters
      ↓
Repeat

The goal is to find values of `w` and `b` that minimize the cost function.

---

## 💻 Implementation

I implemented multiple linear regression using NumPy.

The implementation includes:

- Computing predictions
- Computing the cost function
- Computing gradients
- Updating model parameters
- Applying gradient descent
- Feature scaling

See:

`notebooks/C1_W2_Linear_Regression_with_Multiple_Variables.ipynb`

---

## 🧮 Vectorization

One important concept I learned this week is vectorization.

Instead of calculating each feature separately:

ŷ = w₁x₁ + w₂x₂ + w₃x₃ + b

we can represent the calculation using vectors:

ŷ = w · x + b

This allows NumPy to perform the computation efficiently using
vectorized operations.

---

## 📊 Feature Scaling

When features have very different ranges, gradient descent can
become inefficient.

For example:

- House size: 500–5,000
- Number of bedrooms: 1–5
- Age: 1–100

Feature scaling helps put features into comparable ranges.

This can make gradient descent converge faster.

---

## 🚀 Gradient Descent

Gradient descent repeatedly updates the parameters:

w := w - α * ∂J/∂w

b := b - α * ∂J/∂b

Where:

- α = learning rate
- J = cost function
- ∂J/∂w = gradient with respect to weights
- ∂J/∂b = gradient with respect to bias

The goal is to move the parameters in a direction that reduces
the cost.

---

## 🔍 My Understanding

### Before

I understood linear regression mainly as:

"Find a line that best fits the data."

### After

I now understand that multiple linear regression extends this idea
to multiple features.

The model learns a weight for each feature, and each weight
represents how strongly that feature contributes to the prediction.

---

## 🧠 Mental Model

I think about multiple linear regression like this:

Features
   │
   ├── x₁ × w₁
   ├── x₂ × w₂
   ├── x₃ × w₃
   └── xₙ × wₙ
          │
          ▼
     Add Everything
          │
          + b
          │
          ▼
      Prediction

The model learns:

- How important each feature is → weights
- The baseline adjustment → bias

---

## 🐛 Problems / Debugging

During implementation, I focused on understanding:

- Matrix dimensions
- Shape of `X`
- Shape of `w`
- Dot product
- Broadcasting
- Vectorized calculations

A useful debugging habit is:

```python
print(X.shape)
print(w.shape)
print(b)
