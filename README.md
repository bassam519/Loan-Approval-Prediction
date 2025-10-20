# Loan-Approval-Prediction 🏦

## Overview
This project focuses on building and comparing machine learning models to predict the **loan approval status** of applicants. The primary goal is to leverage data science to assist financial institutions in automating and improving their loan application decision process by identifying key predictors and selecting the most accurate predictive model.

The complete workflow, from data ingestion to model evaluation, is documented within the **`Predicting_Loan_Approvals_With_ML..ipynb`** Jupyter Notebook.

---

## Key Technologies and Topics
This project implements a standard machine learning pipeline using Python and its scientific computing ecosystem.

### Main Topics
* **Supervised Machine Learning**: Specifically, binary classification to predict one of two outcomes (Approved/Rejected).
* **Exploratory Data Analysis (EDA)**: Initial data cleaning, visualization, and identification of feature distributions.
* **Data Preprocessing**: Preparing raw data for model consumption.
* **Model Evaluation**: Using key metrics and visualizations (Confusion Matrices) to assess performance.

### Python Libraries Used
* `pandas` & `numpy`: Data manipulation and numerical operations.
* `matplotlib` & `seaborn`: Data visualization (for EDA and Confusion Matrix plotting).
* `scikit-learn`: The core library for machine learning tasks, including:
    * `LogisticRegression`
    * `DecisionTreeClassifier`
    * `StandardScaler` & `OrdinalEncoder` (for preprocessing)
    * `train_test_split` (for cross-validation)
    * `accuracy_score`, `f1_score`, `confusion_matrix`

---

## Dataset
The core data is provided in the **`loan_approval_dataset.csv`** file, containing over 4,200 records. Key features include:
* **Creditworthiness**: `cibil_score` (a critical factor).
* **Financials**: `income_annum`, `loan_amount`.
* **Assets**: Values for `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, and `bank_asset_value`.
* **Personal Info**: `no_of_dependents`, `education`, and `self_employed` status.
* **Target Variable**: `loan_status` (Approved/Rejected).

---

## Methodology
The methodology involved rigorous data preparation and model comparison:

1.  **Preprocessing**:
    * **Categorical Encoding**: Variables like `education` and `self_employed` were converted to numerical format using **Ordinal Encoding**.
    * **Feature Scaling**: All numerical features were scaled using **StandardScaler** to prevent features with larger magnitudes from dominating the models.
2.  **Model Training**: The data was split (80/20) for training and testing. Two models were trained:
    * **Logistic Regression**: Simple, baseline classification model.
    * **Decision Tree Classifier**: Non-linear model to potentially capture complex interactions.
3.  **Hyperparameter Tuning (Implied)**: While not explicitly stated, models were trained to maximize generalization and performance.

---

## Results and Evaluation
The Decision Tree model demonstrated superior performance over the Logistic Regression model.

| Model | Accuracy Score | F1-Score |
| :--- | :--- | :--- |
| **Logistic Regression** | **94.03%** | **0.9247** |
| **Decision Tree Classifier** | **98.01%** | **0.9731** |

### Confusion Matrix for Logistic Regression (LR)
The confusion matrix for the Logistic Regression model:

![Logistic Regression Confusion Matrix](https://github.com/bassam519/Loan-Approval-Prediction/blob/main/lr%20cm.png?raw=true)

### Confusion Matrix for Decision Tree Classifier (DT)
The Decision Tree matrix highlights its lower misclassification rate:

![Decision Tree Confusion Matrix](https://github.com/bassam519/Loan-Approval-Prediction/blob/main/Dt%20cm.png?raw=true)

---

## Conclusion
The **Decision Tree Classifier** achieved a high accuracy and F1-score, making it the most reliable model for accurately predicting the loan approval status based on the provided dataset.

---

## Repository Contents
| File Name | Description |
| :--- | :--- |
| `Predicting_Loan_Approvals_With_ML..ipynb` | The executable Jupyter Notebook with all the code, analysis, and model results. |
| `loan_approval_dataset.csv` | The raw dataset used for training and testing the models. |
| `README.md` | This comprehensive documentation file. |

---

## Setup and Usage
To run this project locally, you will need Python and the necessary machine learning libraries.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/bassam519/Loan-Approval-Prediction.git](https://github.com/bassam519/Loan-Approval-Prediction.git)
    cd Loan-Approval-Prediction
    ```
2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```
3.  **Run the Notebook:**
    ```bash
    jupyter notebook
    ```
