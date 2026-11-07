# 🚗 Car Price Prediction

A machine learning project that predicts the **selling price of used cars** based on key features like brand, age, kilometers driven, fuel type, and transmission. Includes a Streamlit web app for interactive predictions.

---

## 📁 Project Structure

```
3.Car_Price_Prediction/
├── data/
│   └── car_data.csv                  # Dataset (4340 records)
├── images/
│   └── car_bg.jpg                    # Background image for the app
├── models/
│   └── best_car_price_model.pkl      # Saved best ML model (Random Forest)
├── app.py                            # Streamlit web application
├── car_price_prediction.ipynb        # Jupyter Notebook (EDA + Model Training)
└── requirements.txt                  # Python dependencies
```

---

## 📊 Dataset

- **Source:** `data/car_data.csv`
- **Records:** 4,340 (3,577 after removing duplicates)
- **Columns:** `name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`

---

## 🔧 Features Used for Prediction

| Feature        | Description                          |
|----------------|--------------------------------------|
| `brand`        | Car manufacturer (extracted from name) |
| `car_age`      | Age of the car (current year - year) |
| `km_driven`    | Total kilometers driven              |
| `fuel`         | Fuel type (petrol, diesel, cng, etc.)|
| `transmission` | Manual or Automatic                  |

**Target:** `selling_price`

---

## 🧪 ML Pipeline

1. **Data Cleaning** — Removed duplicates, stripped whitespace, normalized casing, dropped invalid rows
2. **Feature Engineering** — Extracted `brand` from car name, computed `car_age`
3. **EDA** — Price distribution, car age vs price scatter, fuel type boxplot, correlation heatmap
4. **Preprocessing** — `ColumnTransformer` with `SimpleImputer` + `OneHotEncoder` for categoricals, `SimpleImputer` for numericals
5. **Models Trained:**
   - Linear Regression
   - Random Forest Regressor (200 estimators)
6. **Best Model:** Random Forest (lowest MAE)

### Model Performance

| Model             | MAE        | RMSE       | R² Score |
|-------------------|------------|------------|----------|
| Linear Regression | ~208,837   | ~434,711   | 0.413    |
| **Random Forest** | **~160,023** | **~368,479** | **0.579** |

---

## 🌐 Streamlit Web App

The app (`app.py`) allows users to input car details and get an estimated selling price.

### Input Fields
- Car Brand (dropdown)
- Car Age (years)
- Kilometers Driven
- Fuel Type
- Transmission Type

### Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
joblib
```

---

## 🚀 How to Use

1. Clone or download the project
2. Install dependencies: `pip install -r requirements.txt`
3. (Optional) Retrain the model by running `car_price_prediction.ipynb`
4. Launch the app: `streamlit run app.py`
5. Enter car details and click **Predict Car Price**

---

## 📌 Notes

- The model was trained on Indian used car market data
- Prices are in Indian Rupees (₹)
- Prediction accuracy depends on how well the input matches the training distribution
