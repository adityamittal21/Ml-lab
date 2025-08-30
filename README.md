# ML Lab — Assignment 2: Linear Regression

**Date:** August 20, 2025

## Objective
Implement **Linear Regression** using:  
1. **Normal Equation (closed-form)**  
2. **Batch Gradient Descent (iterative)**  
3. **Scikit-learn’s `LinearRegression`**  

Compare results and visualize training/validation performance.

---

## Dataset
- `train.csv` — Training data  
- `test.csv` — Test/validation data

---

## Tasks

**1. Data Preprocessing**  
- Log-transform skewed features (`total_rooms`, `total_bedrooms`, `population`, `households`)  
- Feature engineering: `bedroom_ratio`, `household_rooms`  
- Standardize features and add bias column

**2. Normal Equation**  
\[
\hat{\theta} = (X^T X)^{-1} X^T y
\]

**3. Gradient Descent**  
- MSE loss:  
\[
J(\theta) = \frac{1}{2n} \sum_{i=1}^{n} (h_\theta(x^{(i)}) - y^{(i)})^2
\]  
- Update: \(\theta \leftarrow \theta - \alpha \nabla J(\theta)\)  
- Support learning rate, iterations, early stopping  
- Track training loss

**4. Scikit-learn Comparison**  
- Fit `LinearRegression()` and compare metrics

---

## Visualizations
- **Training Loss vs Iterations** (gradient descent, different learning rates)  
- **Validation Loss**  
- **Gradient/Loss Surface** (optional 3D plot)

All plots saved as **PNG/PDF**.

---

## Evaluation Metrics
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R²  
- Mean Absolute Error (MAE)

Compare metrics across all three implementations.

---

