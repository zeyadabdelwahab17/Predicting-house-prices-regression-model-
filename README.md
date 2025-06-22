

# 🏡 Housing Price Prediction Project

## 📌 Overview

This machine learning project aims to predict housing prices using various regression models, with the best performance achieved using **Gradient Boosting Regressor** with an **R² score of 0.98**. The process covers data cleaning, feature engineering, model selection, hyperparameter tuning, and evaluation.

---

## 🧹 1. Data Preparation

### ✔ Data Cleaning

* No missing or duplicated values were found.
* Column `stories` was renamed to `floors` for improved clarity and understanding.
* Binary categorical features were encoded using **binary encoding**.
* `furnishingstatus` was encoded using **ordinal encoding** as it has a natural order:

  ```
  unfurnished     -> 0  
  semi-furnished  -> 1  
  furnished       -> 2
  ```

### ✔ Outlier Treatment

To reduce the effect of extreme values:

```python
df['price'] = df['price'].clip(upper=df['price'].quantile(0.99))
df['area'] = df['area'].clip(upper=df['area'].quantile(0.99))
```

---

## 🛠️ 2. Feature Engineering

Additional features were created to improve model learning:

```python
df['price_per_sqft'] = df['price'] / df['area']
df['room_ratio'] = df['bedrooms'] / df['bathrooms'].replace(0, 0.1)  # Prevent division by zero
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
```

These features helped capture more meaningful relationships in the dataset.

---

## 🤖 3. Models Used

Seven regression models were trained and compared:

1. **Linear Regression**
2. **Ridge Regression**
3. **Lasso Regression**
4. **Decision Tree Regressor**
5. **Random Forest Regressor**
6. **XGBoost Regressor**
7. **Gradient Boosting Regressor** ✅ **(Best Performer)**

---

## 🏆 4. Best Model: Gradient Boosting Regressor

* **R² Score**: `0.983`
* **Mean Absolute Error (MAE)**: Low
* **Mean Squared Error (MSE)**: Low

The Gradient Boosting model captured complex patterns and outperformed simpler models in terms of variance explained and generalization.

---

## 📈 5. Evaluation Metrics

Models were evaluated using:

* **R² Score** — How much variance is explained
* **MAE** — Average prediction error
* **MSE / RMSE** — Penalizes large errors more

---

## 📊 6. Visualizations (optional if available)

* Correlation heatmap
* Price vs Area scatter plots
* Residual plots
* Actual vs Predicted plot

---

## 🧠 7. Key Insights

* Feature engineering significantly improved model performance.
* Gradient Boosting handled the data’s non-linearity effectively.
* Ordinal encoding for `furnishingstatus` retained the inherent order, which helped models learn better.

---

## 🚀 Next Steps / Improvements

* Deploy the model using a web framework like Streamlit or Flask.
* Use cross-validation with time-based splitting (if dataset is time-aware).
* Perform SHAP or permutation-based feature importance analysis.
* Consider stacking models for better ensemble performance.

 
