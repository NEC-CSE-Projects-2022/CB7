
# CB7 – SmartWasteNet: A Deep Learning Framework to Transition from Take-Make-Waste to Rethink-Redesign-Reuse for Circular Economy under SDG 12


## Team Info
- 22471A05J3 — Shaik Rasheed ( [LinkedIn](https://www.linkedin.com/in/shaik-rasheed-222274276/) )
_Work Done: Data preprocessing, feature engineering, model implementation (MLP classifiers), training & evaluation, SHAP explainability, GitHub documentation,Literature survey, model comparison, report writing, result interpretation.

- 22471A05K5 — Vattikuti Hemanth ( [LinkedIn](https://www.linkedin.com/in/hemanth-vattikuti-65a11a341?/) )
_Work Done: Dataset collection, SDG indicator integration, exploratory data analysis (EDA), visualization.

- 22471A05I5 —  Kumbha Chandra Sekar (( [LinkedIn](https://www.linkedin.com/in/chandra-6b0648276?/) )
_Work Done: Autoencoder-based clustering, label assignment (Rethink/Redesign/Reuse), results analysis.



---

## Abstract
Rapid urbanization has significantly increased municipal solid waste (MSW), making traditional “Take-Make-Waste” models unsustainable. SmartWasteNet proposes a deep learning-based decision support framework aligned with Circular Economy principles (Rethink, Redesign, Reuse) under UN SDG-12.
Using data from 59 Indian cities, the system integrates MSW statistics, recycling efficiency, population density, and SDG indicators. An autoencoder is used for city-level clustering, followed by MLP-based classifiers to recommend optimal circular economy actions. The Dropout-enhanced MLP achieved 97.86% accuracy, and SHAP explainability ensures transparent decision-making for policymakers.

---

## Paper Reference (Inspiration)
👉 [A Comparative Analysis of Forecasting Algorithms for Predicting Municipal Solid Waste Generation in Chittagong City.](https://link.springer.com/article/10.1007/s13762-025-06488-0)
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

1. SmartWasteNet analyzes city-level waste and sustainability indicators to recommend circular economy actions:

2. Rethink (high waste, low recycling)

3. Redesign (moderate efficiency)

4. Reuse (high recycling & sustainability)

Why it is useful :

1. Supports data-driven urban waste policy decisions

2. Aligns waste management with SDG-12

3. Provides interpretable AI outputs instead of black-box predictions

Workflow:

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

1. Python 3.x

2. NumPy

3. Pandas

4. Scikit-learn

5. TensorFlow / Keras

6. Matplotlib

7. Seaborn

8. SHAP

---

## EDA & Preprocessing
1. Missing value handling using mean/median imputation

2. Label encoding for categorical variables

3. Z-score normalization for numerical features

4. Correlation analysis & feature selection

5. Autoencoder-based dimensionality reduction

6. Visualization using heatmaps, scatter plots, and trend graphs

---

## Model Training Info
Autoencoder for unsupervised clustering

K-Means on latent space for CE label assignment

MLP architectures tested:

1. 1-Layer MLP

2. 2-Layer MLP

3. Dropout MLP

4. BatchNorm MLP

5. Wide-Deep MLP

6. Optimizer: Adam

7. Loss: Categorical Cross-Entropy

8. Validation: 5-Fold Cross-Validation

9. Regularization: Dropout & Early Stopping

---

## Model Testing / Evaluation

Evaluation metrics:

1. Accuracy

2. Precision

3. Recall

4. F1-Score

5. Confusion Matrix

6. SHAP Feature Importance

7. Best model: Dropout MLP

---

## Results

1. Highest Accuracy: 97.86%

2. Perfect classification across CE action categories

Key influencing features:

1. SDG-12 Score

2. Recycling Rate

3. Population Density

The model demonstrates strong generalization and interpretability, making it suitable for real-world decision support.

---

## Limitations & Future Work

Limitations :

1. Static (non real-time) dataset

2. Limited to Indian urban cities

3. No IoT sensor integration

Future Work :

1. Integration with real-time IoT waste data

2. Extension to rural and global datasets

3. Time-series forecasting with LSTM

4. Interactive dashboards for policymakers

5. GIS-based waste analytics

---

## Deployment Info
The SmartWasteNet project is designed to be easily deployable on a local machine or cloud-based platforms for analysis and decision support.

Deployment Type: Local Machine / Cloud-Based (Optional)

Environment: Python-based ML environment using Anaconda

Platform:

Local Laptop (Windows 10/11, 64-bit)

Google Colab (for cloud-based execution and experimentation)

Model Serving (Optional):

Flask / Streamlit used to deploy the trained model as a web application

Input:

City-level waste and sustainability indicators (CSV format)

Output:

Predicted Circular Economy Action (Rethink / Redesign / Reuse)

Feature importance explanations using SHAP

Execution Flow:
Dataset upload → Preprocessing → Model inference → CE action recommendation → Visualization

Scalability:
The system can be extended to handle larger datasets and real-time inputs with minimal changes.

---
