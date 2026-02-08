
# CB7 – SmartWasteNet: A Deep Learning Framework to Transition from Take-Make-Waste to
Rethink-Redesign-Reuse for Circular Economy under SDG 12


## Team Info
- 22471A05J3 — Shaik Rasheed ( [LinkedIn](https://www.linkedin.com/in/shaik-rasheed-222274276/) )
_Work Done: Data preprocessing, feature engineering, model implementation (MLP classifiers), training & evaluation, SHAP explainability, GitHub documentation,Literature survey, model comparison, report writing, result interpretation.

- 22471A05K5 — Vattikuti Hemanth ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: Dataset collection, SDG indicator integration, exploratory data analysis (EDA), visualization.

- 22471A05I5 —  Kumbha Chandra Sekar (( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: Autoencoder-based clustering, label assignment (Rethink/Redesign/Reuse), results analysis.



---

## Abstract
Rapid urbanization has significantly increased municipal solid waste (MSW), making traditional “Take-Make-Waste” models unsustainable. SmartWasteNet proposes a deep learning-based decision support framework aligned with Circular Economy principles (Rethink, Redesign, Reuse) under UN SDG-12.
Using data from 59 Indian cities, the system integrates MSW statistics, recycling efficiency, population density, and SDG indicators. An autoencoder is used for city-level clustering, followed by MLP-based classifiers to recommend optimal circular economy actions. The Dropout-enhanced MLP achieved 97.86% accuracy, and SHAP explainability ensures transparent decision-making for policymakers.

---

## Paper Reference (Inspiration)
👉 [A Comparative Analysis of Forecasting Algorithms for Predicting Municipal Solid Waste Generation in Chittagong City.](Paper URL here)
Original conference/IEEE paper used as inspiration for the model.

---

## Our Improvement Over Existing Paper
1. Implemented end-to-end reproducible ML pipeline (EDA → preprocessing → clustering → classification → explainability).

2. Enhanced model comparison using multiple MLP variants.

3. Integrated SHAP explainability for transparent feature importance analysis.

4. Structured the framework for real-world deployment and scalability.

5. Provided GitHub-ready implementation for academic and industrial reuse.

---

## About the Project

What the project does :

SmartWasteNet analyzes city-level waste and sustainability indicators to recommend circular economy actions:

Rethink (high waste, low recycling)

Redesign (moderate efficiency)

Reuse (high recycling & sustainability)

Why it is useful :

Supports data-driven urban waste policy decisions

Aligns waste management with SDG-12

Provides interpretable AI outputs instead of black-box predictions

Workflow

Input Data → Preprocessing → Autoencoder Clustering → MLP Classification → SHAP Explainability → CE Action Recommendation
---

## Dataset Used
👉 **[Waste Management and Recycling in Indian Cities](https://www.kaggle.com/datasets/krishnayadav456wrsty/waste-management-and-recycling-in-indian-cities)**

**Dataset Details:**
1. 59 Indian cities

2. Municipal Solid Waste (tons/day)

3. Recycling & composting rates

4. Population density & city area

5. SDG-11 & SDG-12 scores (NITI Aayog)

6. Municipal efficiency indicators

Data sources include World Bank (What a Waste 2.0), CPCB India, and SDG India Index.

---

## Dependencies Used

Python 3.x

NumPy

Pandas

Scikit-learn

TensorFlow / Keras

Matplotlib

Seaborn

SHAP

---

## EDA & Preprocessing
Missing value handling using mean/median imputation

Label encoding for categorical variables

Z-score normalization for numerical features

Correlation analysis & feature selection

Autoencoder-based dimensionality reduction

Visualization using heatmaps, scatter plots, and trend graphs

---

## Model Training Info
Autoencoder for unsupervised clustering

K-Means on latent space for CE label assignment

MLP architectures tested:

1-Layer MLP

2-Layer MLP

Dropout MLP

BatchNorm MLP

Wide-Deep MLP

Optimizer: Adam

Loss: Categorical Cross-Entropy

Validation: 5-Fold Cross-Validation

Regularization: Dropout & Early Stopping

---

## Model Testing / Evaluation

Evaluation metrics:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

SHAP Feature Importance

Best model: Dropout MLP

---

## Results

Highest Accuracy: 97.86%

Perfect classification across CE action categories

Key influencing features:

SDG-12 Score

Recycling Rate

Population Density

The model demonstrates strong generalization and interpretability, making it suitable for real-world decision support.

---

## Limitations & Future Work

Limitations :

Static (non real-time) dataset

Limited to Indian urban cities

No IoT sensor integration

Future Work :

Integration with real-time IoT waste data

Extension to rural and global datasets

Time-series forecasting with LSTM

Interactive dashboards for policymakers

GIS-based waste analytics

---

## Deployment Info
xxxxxxxxxx

---
