import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

from src.cleaner import DataCleaner
from src.features import FeatureProcessor
from src.model import CustomBGDRegressor
from src.recommender import AssetRecommender


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="OptiAsset | Enterprise Marketing Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title(
    "🚀 OptiAsset: Predictive Marketing & Asset Intelligence Engine"
)

st.markdown(
    "An end-to-end Machine Learning system powered by "
    "custom linear algebra, dimensionality reduction, "
    "and spatial asset clustering."
)


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------

st.sidebar.header("Navigation & Controls")

app_mode = st.sidebar.selectbox(
    "Choose Pipeline View",
    [
        "1. Data Overview & Cleaning",
        "2. PCA & Feature Engineering",
        "3. Conversion Prediction & Recommendations"
    ]
)


# ---------------------------------------------------------
# ABSOLUTE / PROJECT PATH
# ---------------------------------------------------------

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "raw_marketing_data.csv"
)


# ---------------------------------------------------------
# LOAD + CLEAN DATA
# ---------------------------------------------------------

@st.cache_data
def load_and_clean_data(filepath):

    cleaner = DataCleaner(filepath)

    cleaner.load_data()
    cleaner.clean_missing_values()
    cleaner.encode_categorical_data()

    df = cleaner.get_cleaned_data()

    # Return both dataframe and encoders
    return df, cleaner.encoders


# ---------------------------------------------------------
# INITIALIZE ML PIPELINE
# ---------------------------------------------------------

@st.cache_resource
def initialize_pipeline(df, target_col):

    # All columns except target
    feature_cols = [
        col for col in df.columns
        if col != target_col
    ]

    processor = FeatureProcessor(
        df,
        target_col=target_col
    )

    # Scaling
    X_scaled, y = (
        processor.extract_and_scale_features(
            feature_cols
        )
    )

    # PCA
    X_pca = processor.apply_pca(
        X_scaled,
        n_components=3
    )

    # Custom BGD Regression
    regressor = CustomBGDRegressor(
        learning_rate=0.01,
        epochs=500
    )

    regressor.fit(
        X_pca,
        y
    )

    # KNN Recommender
    recommender = AssetRecommender(
        n_neighbors=4
    )

    recommender.fit(X_pca)

    return (
        processor,
        regressor,
        recommender,
        X_pca,
        y,
        feature_cols
    )


# ---------------------------------------------------------
# CHECK DATASET
# ---------------------------------------------------------

if not os.path.exists(DEFAULT_DATA_PATH):

    st.error(
        f"Dataset not found at: {DEFAULT_DATA_PATH}. "
        "Place 'raw_marketing_data.csv' inside 'data/' folder."
    )

else:

    # IMPORTANT:
    # Now we receive both dataframe and encoders
    df_clean, encoders = load_and_clean_data(
        DEFAULT_DATA_PATH
    )


    # =====================================================
    # 1. DATA OVERVIEW & CLEANING
    # =====================================================

    if app_mode == "1. Data Overview & Cleaning":

        st.header(
            "Step 1: Raw Data Ingestion & Automated Cleaning"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Records",
            df_clean.shape[0]
        )

        col2.metric(
            "Cleaned Features",
            df_clean.shape[1]
        )

        col3.metric(
            "Noise & Leakage",
            "Removed"
        )

        st.subheader(
            "Cleaned Dataset Preview"
        )

        st.dataframe(
            df_clean.head(10),
            use_container_width=True
        )

        st.subheader(
            "Feature Correlation Matrix"
        )

        corr = (
            df_clean
            .select_dtypes(include=[np.number])
            .corr()
        )

        st.dataframe(
            corr.style.format("{:.2f}"),
            use_container_width=True
        )

        # Show encoding mappings
        if encoders:

            st.subheader(
                "Categorical Feature Encodings"
            )

            for col, encoder in encoders.items():

                mapping = dict(
                    zip(
                        encoder.classes_,
                        encoder.transform(
                            encoder.classes_
                        )
                    )
                )

                st.write(
                    f"**{col}:** {mapping}"
                )


    # =====================================================
    # 2. PCA & FEATURE ENGINEERING
    # =====================================================

    elif app_mode == "2. PCA & Feature Engineering":

        st.header(
            "Step 2: Dimensionality Reduction "
            "via Principal Component Analysis (PCA)"
        )

        n_components = st.slider(
            "Select Number of PCA Components",
            min_value=2,
            max_value=5,
            value=3
        )

        target_candidates = [
            c
            for c in df_clean.columns
            if (
                "conversion" in c.lower()
                or "target" in c.lower()
                or "label" in c.lower()
            )
        ]

        default_target = (
            target_candidates[0]
            if target_candidates
            else df_clean.columns[-1]
        )

        target_col = st.selectbox(
            "Select Target Variable (Y)",
            df_clean.columns,
            index=list(
                df_clean.columns
            ).index(default_target)
        )

        feature_cols = [
            col
            for col in df_clean.columns
            if col != target_col
        ]

        processor = FeatureProcessor(
            df_clean,
            target_col=target_col
        )

        X_scaled, y = (
            processor.extract_and_scale_features(
                feature_cols
            )
        )

        X_pca = processor.apply_pca(
            X_scaled,
            n_components=n_components
        )

        st.subheader(
            "3D Spatial Distribution of Marketing Assets"
        )

        pca_df = pd.DataFrame(
            X_pca,
            columns=[
                f"PC{i+1}"
                for i in range(n_components)
            ]
        )

        pca_df["Target"] = y

        if n_components >= 3:

            fig = px.scatter_3d(
                pca_df,
                x="PC1",
                y="PC2",
                z="PC3",
                color="Target",
                opacity=0.7,
                title="PCA Component Space"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            fig = px.scatter(
                pca_df,
                x="PC1",
                y="PC2",
                color="Target",
                opacity=0.7,
                title="PCA Component Space"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # =====================================================
    # 3. PREDICTION & RECOMMENDATIONS
    # =====================================================

    elif app_mode == "3. Conversion Prediction & Recommendations":

        st.header(
            "Step 3: Custom Gradient Descent Prediction "
            "& KNN Recommender"
        )

        target_candidates = [
            c
            for c in df_clean.columns
            if (
                "conversion" in c.lower()
                or "target" in c.lower()
            )
        ]

        target_col = (
            target_candidates[0]
            if target_candidates
            else df_clean.columns[-1]
        )

        # Initialize ML pipeline
        with st.spinner(
            "Initializing ML Pipeline & "
            "Training NumPy Gradient Descent..."
        ):

            (
                processor,
                regressor,
                recommender,
                X_pca,
                y,
                feature_cols
            ) = initialize_pipeline(
                df_clean,
                target_col
            )

        st.divider()

        st.subheader(
            "Simulate a New Marketing Asset"
        )

        st.info(
            "Select categorical values from the dropdowns. "
            "They will automatically be converted to the "
            "same numerical encoding used during training."
        )

        user_inputs = {}

        cols = st.columns(3)

        display_cols = [
            col
            for col in feature_cols
            if "id" not in col.lower()
        ][:6]


        # -------------------------------------------------
        # USER INPUTS
        # -------------------------------------------------

        for idx, col_name in enumerate(display_cols):

            with cols[idx % 3]:

                # CATEGORICAL FEATURE
                if col_name in encoders:

                    categories = (
                        encoders[col_name].classes_
                    )

                    selected_value = st.selectbox(
                        col_name,
                        categories
                    )

                    # Convert category to encoded number
                    encoded_value = (
                        encoders[col_name]
                        .transform(
                            [selected_value]
                        )[0]
                    )

                    user_inputs[col_name] = (
                        encoded_value
                    )

                # NUMERICAL FEATURE
                else:

                    min_val = float(
                        df_clean[col_name].min()
                    )

                    max_val = float(
                        df_clean[col_name].max()
                    )

                    mean_val = float(
                        df_clean[col_name].mean()
                    )

                    user_inputs[col_name] = (
                        st.slider(
                            col_name,
                            min_val,
                            max_val,
                            mean_val
                        )
                    )


        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        if st.button(
            "Evaluate Asset Performance "
            "& Find Matches",
            type="primary"
        ):

            # Start with first row as a template
            dummy_row = (
                df_clean.iloc[0].copy()
            )

            # Replace selected fields
            for k, v in user_inputs.items():

                dummy_row[k] = v


            # Convert user input into DataFrame
            user_df = pd.DataFrame(
                [dummy_row[feature_cols]]
            )


            # Same scaling used during training
            user_scaled = (
                processor.scaler.transform(
                    user_df
                )
            )


            # Same PCA model used during training
            user_pca = (
                processor.pca_model.transform(
                    user_scaled
                )
            )


            # BGD prediction
            predicted_score = (
                regressor
                .predict(user_pca)[0]
            )


            st.success(
                f"🎯 **Predicted Asset "
                f"Conversion / Engagement Score:** "
                f"`{predicted_score:.4f}`"
            )


            # -------------------------------------------------
            # KNN SIMILAR ASSETS
            # -------------------------------------------------

            distances, indices = (
                recommender.find_similar_assets(
                    user_pca
                )
            )

            st.subheader(
                "🔗 Top Similar Historical Assets"
            )

            match_cols = st.columns(3)

            for i in range(1, 4):

                match_idx = indices[i]

                with match_cols[i - 1]:

                    st.markdown(
                        f"**Match #{i} "
                        f"(Row {match_idx})**"
                    )

                    st.text(
                        f"Euclidean Distance: "
                        f"{distances[i]:.4f}"
                    )

                    st.dataframe(
                        df_clean.iloc[
                            match_idx
                        ][display_cols[:3]],
                        use_container_width=True
                    )