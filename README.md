# OptiAsset 🚀
## Predictive Marketing & Asset Intelligence Engine

OptiAsset is an end-to-end machine learning system designed to help marketers analyze historical campaign performance, reduce high-dimensional feature space, predict engagement scores using custom-built math, and discover structurally similar high-performing historical assets via spatial clustering.

The system bridges foundational linear algebra with a production-grade interactive web dashboard, avoiding "black-box" libraries for core regression modeling by implementing algorithms entirely from scratch.

---

## Core Pipeline

Raw Dataset Ingestion 
→ Automated Data Cleaning & Leakage Prevention (`DataCleaner`) 
→ Feature Standardization (`StandardScaler`) 
→ Dimensionality Reduction (`PCA`) 
→ Custom Batch Gradient Descent Regression (`CustomBGDRegressor`) & Spatial Asset Clustering (`AssetRecommender`) 
→ Real-Time Interactive Analytics & Simulation Dashboard (`Streamlit` & `Plotly`)

---

## Key Features

1. **Automated Data Hygiene:** Programmatically drops noise identifiers (`CustomerID`) and prevents target leakage by stripping alternate success metrics (`ConversionRate`, `ClickThroughRate`) prior to model training.
2. **Custom Math From Scratch:** Implements a Batch Gradient Descent (BGD) regression algorithm in **pure NumPy**, optimizing model weights through Mean Squared Error (MSE) loss minimization and Ordinary Least Squares (OLS) logic.
3. **Dimensionality Reduction & Multicollinearity Control:** Utilizes Principal Component Analysis (PCA) to project high-dimensional marketing variables into orthogonal latent components, tracking variance retention.
4. **Spatial Similarity Clustering:** Employs a K-Nearest Neighbors (KNN) model using Euclidean distance metrics over the PCA feature space to instantly surface top-performing historical campaign matches.
5. **Interactive Streamlit Dashboard:** Features a multi-view production UI equipped with dynamic simulation sliders, correlation heatmaps, and 3D spatial scatter plots rendered via Plotly.

---

## Tech Stack

- **Core Language:** Python
- **Math & Data Processing:** NumPy, Pandas, Scikit-learn (Preprocessing & PCA)
- **Machine Learning Architecture:** Custom NumPy Batch Gradient Descent, Scikit-learn KNN
- **UI & Visualization:** Streamlit, Plotly
- **Deployment:** Docker, Git

---

## Project Structure

```text
OptiAsset/
│
├── data/
│   └── raw_marketing_data.csv       # Kaggle digital marketing dataset
│
├── src/
│   ├── __init__.py                  # Package initializer
│   ├── cleaner.py                   # Data ingestion, cleaning, and encoding pipeline
│   ├── features.py                  # StandardScaler and PCA dimensionality reduction
│   ├── model.py                     # Custom Batch Gradient Descent Regressor (NumPy)
│   └── recommender.py               # KNN spatial asset similarity engine
│
├── app.py                           # Multi-view Streamlit interactive web dashboard
├── requirements.txt                 # Project dependencies
└── Dockerfile                       # Production container configuration