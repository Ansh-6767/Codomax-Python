# ============================================================
# HOUSE PRICE PREDICTION
# END-TO-END DATA SCIENCE PROJECT
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE_NAME = "house_price_prediction_dataset.csv"

print("=" * 65)
print("        HOUSE PRICE PREDICTION - DATA SCIENCE PROJECT")
print("=" * 65)

if os.path.exists(FILE_NAME):

    df = pd.read_csv(FILE_NAME)

    print("\nDataset loaded successfully.")

else:

    print("\nCSV file not found.")
    print("Creating an educational dataset automatically...")

    np.random.seed(42)

    n = 1000

    area = np.random.randint(600, 4000, n)

    bedrooms = np.clip(
        np.round(
            area / 650 +
            np.random.normal(0, 0.8, n)
        ),
        1,
        6
    ).astype(int)

    bathrooms = np.clip(
        np.round(
            bedrooms * 0.7 +
            np.random.normal(0.5, 0.5, n) * 2
        ) / 2,
        1,
        4
    )

    age = np.random.randint(0, 60, n)

    garage_cars = np.clip(
        np.round(
            area / 1200 +
            np.random.normal(0, 0.5, n)
        ),
        0,
        3
    ).astype(int)

    stories = np.random.choice(
        [1, 1.5, 2, 2.5],
        n,
        p=[0.35, 0.10, 0.50, 0.05]
    )

    overall_quality = np.random.randint(
        3,
        11,
        n
    )

    neighborhood = np.random.choice(
        [
            "Downtown",
            "Suburban",
            "Rural",
            "Urban",
            "Premium"
        ],
        n,
        p=[
            0.18,
            0.32,
            0.15,
            0.25,
            0.10
        ]
    )

    has_basement = np.random.choice(
        ["Yes", "No"],
        n,
        p=[0.65, 0.35]
    )

    has_pool = np.random.choice(
        ["Yes", "No"],
        n,
        p=[0.12, 0.88]
    )

    garage_area = (
        garage_cars *
        np.random.randint(
            180,
            350,
            n
        )
        +
        np.random.randint(
            0,
            80,
            n
        )
    )

    lot_area = (
        area *
        np.random.uniform(
            1.2,
            3.5,
            n
        )
    )

    location_factor = {
        "Downtown": 1.20,
        "Suburban": 1.05,
        "Rural": 0.82,
        "Urban": 0.98,
        "Premium": 1.40
    }

    factor = np.array([
        location_factor[x]
        for x in neighborhood
    ])

    price = (
        45000
        + area * 145
        + overall_quality * 22000
        + bedrooms * 9000
        + bathrooms * 14000
        + garage_cars * 18000
        + garage_area * 35
        + lot_area * 8
        - age * 1200
        + np.where(
            has_basement == "Yes",
            18000,
            0
        )
        + np.where(
            has_pool == "Yes",
            30000,
            0
        )
    )

    price = (
        price *
        factor
    )

    price += np.random.normal(
        0,
        35000,
        n
    )

    price = np.maximum(
        price,
        50000
    ).round(0).astype(int)

    df = pd.DataFrame({

        "Area_sqft": area,

        "Bedrooms": bedrooms,

        "Bathrooms": bathrooms,

        "House_Age": age,

        "Garage_Cars": garage_cars,

        "Garage_Area_sqft": garage_area,

        "Lot_Area_sqft": lot_area.round(0),

        "Stories": stories,

        "Overall_Quality": overall_quality,

        "Neighborhood": neighborhood,

        "Has_Basement": has_basement,

        "Has_Pool": has_pool,

        "Sale_Price": price
    })


# ============================================================
# 2. DATA UNDERSTANDING
# ============================================================

print("\n" + "=" * 65)
print("1. DATA UNDERSTANDING")
print("=" * 65)

print(
    "\nRows:",
    df.shape[0]
)

print(
    "Columns:",
    df.shape[1]
)

print("\nColumns:")

for column in df.columns:
    print("-", column)


print("\nFirst 5 rows:")

print(
    df.head().to_string(index=False)
)


# ============================================================
# 3. DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 65)
print("2. DATA QUALITY CHECK")
print("=" * 65)

print(
    "\nDuplicate rows:",
    df.duplicated().sum()
)

print("\nMissing values:")

missing = df.isnull().sum()

for column, value in missing.items():

    if value > 0:

        print(
            f"{column}: {value}"
        )


# Remove duplicates

df = df.drop_duplicates()


# ============================================================
# 4. BASIC STATISTICS
# ============================================================

print("\n" + "=" * 65)
print("3. BASIC STATISTICS")
print("=" * 65)

print(
    df.describe().round(2).to_string()
)


# ============================================================
# 5. TARGET ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("4. HOUSE PRICE ANALYSIS")
print("=" * 65)

print(
    f"\nAverage Price: "
    f"${df['Sale_Price'].mean():,.2f}"
)

print(
    f"Median Price: "
    f"${df['Sale_Price'].median():,.2f}"
)

print(
    f"Minimum Price: "
    f"${df['Sale_Price'].min():,.2f}"
)

print(
    f"Maximum Price: "
    f"${df['Sale_Price'].max():,.2f}"
)


# ============================================================
# 6. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("5. CORRELATION WITH HOUSE PRICE")
print("=" * 65)

numeric_columns = df.select_dtypes(
    include=np.number
).columns

correlation = (
    df[numeric_columns]
    .corr()["Sale_Price"]
    .sort_values(
        ascending=False
    )
)

print(
    correlation.round(3).to_string()
)


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 65)
print("6. FEATURE ENGINEERING")
print("=" * 65)

# IMPORTANT:
# Price_per_sqft is NOT used as a model feature
# because it directly uses Sale_Price.

df["Garage_Available"] = (
    df["Garage_Cars"] > 0
).astype(int)

df["Is_New_House"] = (
    df["House_Age"] <= 5
).astype(int)

df["Total_Amenities"] = (
    (df["Has_Basement"] == "Yes").astype(int)
    +
    (df["Has_Pool"] == "Yes").astype(int)
    +
    df["Garage_Available"]
)

print(
    "Created features:"
)

print(
    "- Garage_Available"
)

print(
    "- Is_New_House"
)

print(
    "- Total_Amenities"
)


# ============================================================
# 8. PREPARE X AND Y
# ============================================================

X = df.drop(
    columns=["Sale_Price"]
)

y = df["Sale_Price"]


# ============================================================
# 9. TRAIN TEST SPLIT
# ============================================================

print("\n" + "=" * 65)
print("7. TRAIN / TEST SPLIT")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 10. PREPROCESSING
# ============================================================

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()


numeric_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),

        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 11. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_actual,
    y_prediction
):

    mae = mean_absolute_error(
        y_actual,
        y_prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_actual,
            y_prediction
        )
    )

    r2 = r2_score(
        y_actual,
        y_prediction
    )

    return {

        "Model": model_name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2
    }


results = []


# ============================================================
# 12. LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 65)
print("8. TRAINING LINEAR REGRESSION")
print("=" * 65)

linear_model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            LinearRegression()
        )
    ]
)


linear_model.fit(
    X_train,
    y_train
)


linear_predictions = (
    linear_model.predict(
        X_test
    )
)


results.append(
    evaluate_model(
        "Linear Regression",
        y_test,
        linear_predictions
    )
)


# ============================================================
# 13. RANDOM FOREST
# ============================================================

print("\n" + "=" * 65)
print("9. TRAINING RANDOM FOREST")
print("=" * 65)

random_forest = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


random_forest.fit(
    X_train,
    y_train
)


rf_predictions = (
    random_forest.predict(
        X_test
    )
)


results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        rf_predictions
    )
)


# ============================================================
# 14. GRADIENT BOOSTING
# ============================================================

print("\n" + "=" * 65)
print("10. TRAINING GRADIENT BOOSTING")
print("=" * 65)

gradient_boosting = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        )
    ]
)


gradient_boosting.fit(
    X_train,
    y_train
)


gb_predictions = (
    gradient_boosting.predict(
        X_test
    )
)


results.append(
    evaluate_model(
        "Gradient Boosting",
        y_test,
        gb_predictions
    )
)


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 65)
print("11. MODEL COMPARISON")
print("=" * 65)

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "RMSE"
)


print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:,.4f}"
    )
)


# ============================================================
# 16. BEST MODEL
# ============================================================

best_model_name = (
    results_df.iloc[0]["Model"]
)


print("\n" + "=" * 65)

print(
    "BEST MODEL:",
    best_model_name
)

print("=" * 65)


# Choose predictions according to best model

if best_model_name == "Linear Regression":

    final_model = linear_model

    final_predictions = linear_predictions

elif best_model_name == "Random Forest":

    final_model = random_forest

    final_predictions = rf_predictions

else:

    final_model = gradient_boosting

    final_predictions = gb_predictions


best_mae = mean_absolute_error(
    y_test,
    final_predictions
)

best_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        final_predictions
    )
)

best_r2 = r2_score(
    y_test,
    final_predictions
)


print(
    f"\nMean Absolute Error: "
    f"${best_mae:,.2f}"
)

print(
    f"Root Mean Squared Error: "
    f"${best_rmse:,.2f}"
)

print(
    f"R² Score: "
    f"{best_r2:.4f}"
)


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 65)
print("12. FEATURE IMPORTANCE")
print("=" * 65)

if hasattr(
    final_model.named_steps["model"],
    "feature_importances_"
):

    fitted_preprocessor = (
        final_model.named_steps[
            "preprocessor"
        ]
    )

    trained_model = (
        final_model.named_steps[
            "model"
        ]
    )

    feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
    )

    importances = (
        trained_model
        .feature_importances_
    )

    importance_df = pd.DataFrame({

        "Feature":
            feature_names,

        "Importance":
            importances
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
    )

    print(
        importance_df
        .head(15)
        .to_string(
            index=False
        )
    )

else:

    print(
        "Feature importance is not available "
        "for this model."
    )


# ============================================================
# 18. ACTUAL VS PREDICTED
# ============================================================

comparison = pd.DataFrame({

    "Actual_Price":
        y_test.values,

    "Predicted_Price":
        final_predictions.round(0)

})


comparison["Error"] = (
    comparison["Actual_Price"]
    -
    comparison["Predicted_Price"]
)


print("\n" + "=" * 65)
print("13. SAMPLE PREDICTIONS")
print("=" * 65)

print(
    comparison.head(10).to_string(
        index=False
    )
)


# ============================================================
# 19. CUSTOM HOUSE PRICE PREDICTOR
# ============================================================

def predict_house_price():

    print("\n")
    print("=" * 65)
    print("        HOUSE PRICE PREDICTOR")
    print("=" * 65)

    try:

        area = float(
            input(
                "Area (sq ft): "
            )
        )

        bedrooms = int(
            input(
                "Number of bedrooms: "
            )
        )

        bathrooms = float(
            input(
                "Number of bathrooms: "
            )
        )

        house_age = int(
            input(
                "House age (years): "
            )
        )

        garage_cars = int(
            input(
                "Garage capacity (cars): "
            )
        )

        garage_area = float(
            input(
                "Garage area (sq ft): "
            )
        )

        lot_area = float(
            input(
                "Lot area (sq ft): "
            )
        )

        stories = float(
            input(
                "Number of stories: "
            )
        )

        quality = int(
            input(
                "Overall quality (3-10): "
            )
        )

        print(
            "\nNeighborhood options:"
        )

        print(
            "Downtown"
        )

        print(
            "Suburban"
        )

        print(
            "Rural"
        )

        print(
            "Urban"
        )

        print(
            "Premium"
        )

        neighborhood = input(
            "Neighborhood: "
        ).strip()

        basement = input(
            "Has basement? (Yes/No): "
        ).strip().title()

        pool = input(
            "Has pool? (Yes/No): "
        ).strip().title()


        garage_available = int(
            garage_cars > 0
        )

        is_new_house = int(
            house_age <= 5
        )

        total_amenities = (

            int(
                basement == "Yes"
            )

            +

            int(
                pool == "Yes"
            )

            +

            garage_available
        )


        house = pd.DataFrame({

            "Area_sqft": [area],

            "Bedrooms": [bedrooms],

            "Bathrooms": [bathrooms],

            "House_Age": [house_age],

            "Garage_Cars": [garage_cars],

            "Garage_Area_sqft": [
                garage_area
            ],

            "Lot_Area_sqft": [
                lot_area
            ],

            "Stories": [stories],

            "Overall_Quality": [
                quality
            ],

            "Neighborhood": [
                neighborhood
            ],

            "Has_Basement": [
                basement
            ],

            "Has_Pool": [
                pool
            ],

            "Garage_Available": [
                garage_available
            ],

            "Is_New_House": [
                is_new_house
            ],

            "Total_Amenities": [
                total_amenities
            ]
        })


        prediction = (
            final_model.predict(
                house
            )[0]
        )


        print("\n")
        print("=" * 65)
        print("                PREDICTION")
        print("=" * 65)

        print(
            f"\nEstimated House Price:"
        )

        print(
            f"${prediction:,.0f}"
        )

        print(
            "\nNote: This is an ML estimate "
            "based on the training dataset."
        )

    except Exception as error:

        print(
            "\nInvalid input."
        )

        print(
            "Error:",
            error
        )


# ============================================================
# 20. PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("PROJECT SUMMARY")
print("=" * 65)

print("""
This project completed an end-to-end Data Science workflow:

✓ Data loading
✓ Data understanding
✓ Missing-value analysis
✓ Duplicate checking
✓ Statistical analysis
✓ Correlation analysis
✓ Feature engineering
✓ Data preprocessing
✓ Train/test splitting
✓ Linear Regression
✓ Random Forest Regression
✓ Gradient Boosting Regression
✓ Model comparison
✓ MAE / RMSE / R² evaluation
✓ Feature importance
✓ House price prediction

The project demonstrates how machine learning can be used
to estimate house prices from property characteristics.
""")


# ============================================================
# 21. RUN CUSTOM PREDICTOR
# ============================================================

choice = input(
    "\nDo you want to predict the price of a custom house? "
    "(yes/no): "
).strip().lower()


if choice == "yes":

    predict_house_price()


print("\n" + "=" * 65)
print("              PROJECT COMPLETED")
print("=" * 65)
