import pandas as pd
import pickle

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --------------------------------------
# Load Dataset
# --------------------------------------

df=pd.read_csv(
    "data/insurance.csv"
)

# --------------------------------------
# Encode categorical columns
# --------------------------------------

encoders={}

for col in df.select_dtypes(
    include='object'
).columns:

    le=LabelEncoder()

    df[col]=le.fit_transform(
        df[col]
    )

    encoders[col]=le


# --------------------------------------
# Features and target
# --------------------------------------

X=df.drop(
    "charges",
    axis=1
)

y=df["charges"]

# --------------------------------------
# Split
# --------------------------------------

X_train,X_test,y_train,y_test=(
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

# --------------------------------------
# Hyperparameter tuning
# --------------------------------------

rf=RandomForestRegressor()

params={

    "n_estimators":[50,100,200,300],

    "max_depth":[
        None,
        5,
        10,
        15,
        20
    ],

    "min_samples_split":[
        2,
        5,
        10
    ],

    "min_samples_leaf":[
        1,
        2,
        4
    ],

    "max_features":[
        "sqrt",
        "log2"
    ]
}

search=RandomizedSearchCV(

    estimator=rf,

    param_distributions=params,

    n_iter=10,

    cv=5,

    scoring="r2",

    random_state=42,

    n_jobs=-1

)

search.fit(
    X_train,
    y_train
)

best_model=search.best_estimator_

# --------------------------------------
# Evaluate
# --------------------------------------

pred=best_model.predict(
    X_test
)

score=r2_score(
    y_test,
    pred
)

print(
    "R2 Score:",
    score
)

print(
    "Best Parameters:"
)

print(
    search.best_params_
)

# --------------------------------------
# Save files
# --------------------------------------

pickle.dump(
    best_model,
    open(
        "models/random_forest_model.pkl",
        "wb"
    )
)

pickle.dump(
    encoders,
    open(
        "models/encoder.pkl",
        "wb"
    )
)

pickle.dump(
    search.best_params_,
    open(
        "models/best_params.pkl",
        "wb"
    )
)

print(
    "Model Saved"
)