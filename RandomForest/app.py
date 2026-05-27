import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Insurance Cost Prediction",
    layout="wide"
)

# --------------------------------
# Load files
# --------------------------------

model=pickle.load(
    open(
        "models/random_forest_model.pkl",
        "rb"
    )
)

encoders=pickle.load(
    open(
        "models/encoder.pkl",
        "rb"
    )
)

params=pickle.load(
    open(
        "models/best_params.pkl",
        "rb"
    )
)

df=pd.read_csv(
    "data/insurance.csv"
)

X=df.drop(
    "charges",
    axis=1
)

# --------------------------------
# Title
# --------------------------------

st.title(
    "🏥 Insurance Cost Prediction"
)

st.write(
    "Random Forest Regressor"
)

# --------------------------------
# Hyperparameters display
# --------------------------------

st.subheader(
    "Best Hyperparameters"
)

st.json(
    params
)

# --------------------------------
# User Input
# --------------------------------

st.subheader(
    "Enter Details"
)

input_data={}

for col in X.columns:

    if col in encoders:

        values=list(
            encoders[col].classes_
        )

        selected=st.selectbox(
            col,
            values
        )

        encoded=(
            encoders[col]
            .transform(
                [selected]
            )[0]
        )

        input_data[col]=encoded

    else:

        value=st.number_input(

            col,

            value=float(
                X[col].mean()
            )
        )

        input_data[col]=value


# --------------------------------
# Prediction
# --------------------------------

if st.button(
    "Predict"
):

    input_df=pd.DataFrame(
        [input_data]
    )

    prediction=model.predict(
        input_df
    )

    st.success(

        f"Estimated Insurance Cost: ₹ {prediction[0]:,.2f}"

    )