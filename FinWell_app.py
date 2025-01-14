import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Apply custom styles
st.markdown(
    """
    <style>
    /* Set background color to black */
    .stApp {
        background-color: black;
        color: white;
    }

    /* Set the style of text inputs and buttons */
    input, button {
        color: black;
        font-size: 16px;
    }

    /* Customize DataFrame display */
    .dataframe {
        background-color: black;
        color: white;
    }

    /* Add a border around all containers */
    div[data-testid="stVerticalBlock"] {
        border: 1px solid white;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and introduction
st.title("Random Forest Model with FinWell Score")
st.write(
    """
    Welcome to the FinWell Score Predictor app! Upload your validation data and calculate
    FinWell Scores and Risk Categories with our Random Forest model.
    """
)

# File uploader
uploaded_file = st.file_uploader("Upload Validation CSV", type=["csv"])

if uploaded_file is not None:
    # Load the model and uploaded data
    model = joblib.load("optimized_random_forest_model_main.pkl")
    df_val = pd.read_csv(uploaded_file)

    st.write("Data Preview:")
    st.dataframe(df_val.head())  # Show first few rows

    # Feature alignment
    model_features = model.feature_names_in_
    X_val = df_val.filter(items=model_features)

    # Check for missing features in the validation dataset
    missing_features = set(model_features) - set(X_val.columns)
    if missing_features:
        st.error(f"Missing features in validation data: {missing_features}")
    else:
        # Predict probabilities
        val_probs = model.predict_proba(X_val)[:, 1]
        df_val["predicted_probability"] = val_probs

        # Feature normalization
        onus_cols = [col for col in df_val.columns if col.startswith("onus_attributes")]
        transaction_cols = [col for col in df_val.columns if col.startswith("transaction_attribute")]
        bureau_cols = [col for col in df_val.columns if col.startswith("bureau")]
        bureau_enquiry_cols = [col for col in df_val.columns if col.startswith("bureau_enquiry")]

        scaler = MinMaxScaler()
        if onus_cols:
            df_val[onus_cols] = scaler.fit_transform(df_val[onus_cols])
        if transaction_cols:
            df_val[transaction_cols] = scaler.fit_transform(df_val[transaction_cols])
        if bureau_cols:
            df_val[bureau_cols] = scaler.fit_transform(df_val[bureau_cols])
        if bureau_enquiry_cols:
            df_val[bureau_enquiry_cols] = scaler.fit_transform(df_val[bureau_enquiry_cols])

        # FinWell score calculation
        def calculate_finwell_score(row):
            weight_prob = 0.5
            weight_onus = 0.2
            weight_transaction = 0.2
            weight_bureau = 0.05
            weight_enquiry = 0.05

            onus_score = row[onus_cols].mean() if onus_cols else 0
            transaction_score = row[transaction_cols].mean() if transaction_cols else 0
            bureau_score = row[bureau_cols].mean() if bureau_cols else 0
            enquiry_score = row[bureau_enquiry_cols].mean() if bureau_enquiry_cols else 0

            inverse_prob = 1 - row["predicted_probability"]

            raw_score = (
                weight_prob * inverse_prob +
                weight_onus * onus_score +
                weight_transaction * transaction_score +
                weight_bureau * bureau_score +
                weight_enquiry * enquiry_score
            )
            raw_score = max(0, min(raw_score, 1))
            scaled_score = 300 + (raw_score * 600)
            return scaled_score

        df_val["finwell_score"] = df_val.apply(calculate_finwell_score, axis=1)

        # Risk categorization
        def assign_risk_category(score):
            if score < 500:
                return "High Risk"
            elif 500 <= score < 700:
                return "Moderate Risk"
            else:
                return "Low Risk"

        df_val["risk_category"] = df_val["finwell_score"].apply(assign_risk_category)

        # Display results
        st.write("Results:")
        st.dataframe(df_val[["account_number", "predicted_probability", "finwell_score", "risk_category"]].head())

        # Save results option
        if st.button("Save Results"):
            output_file = "finwell_predictions_validation_new.csv"
            df_val[["account_number", "predicted_probability", "finwell_score", "risk_category"]].to_csv(output_file, index=False)
            st.success(f"Results saved to {output_file}")
