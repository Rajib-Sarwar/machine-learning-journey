# Week 2 — Personal Notes

## 1. What is the difference between Linear and Multiple Linear Regression?

### Linear Regression

One feature:

x → prediction

Example:

House Size → House Price

### Multiple Linear Regression

Multiple features:

x₁, x₂, x₃ → prediction

Example:

House Size
Number of Bedrooms
House Age
Location

       ↓

House Price


## My Confusion

I initially thought:

w = one weight for the entire model

But I learned:

w = a vector containing one weight per feature

For example:

w = [w₁, w₂, w₃]

and:

x = [x₁, x₂, x₃]

The prediction is based on:

w₁x₁ + w₂x₂ + w₃x₃ + b

- **Why does `.T` appear here?** 
  → `.T` transposes the array, swapping rows and columns so the matrix has the correct shape for multiplication.
- **Why is the shape `(n_features, m)`?** 
  → Each column represents one training example, while each row represents one feature.
- **Why do we use `np.dot()`?** 
  → `np.dot()` calculates the dot product of weights and features to efficiently compute the weighted sum.
- **Why is `b` a scalar?** 
  → `b` is a single bias value that shifts the model's prediction for all training examples.
