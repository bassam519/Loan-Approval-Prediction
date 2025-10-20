# Loan-Approval-Prediction 🏦

## Overview
This project focuses on building and comparing machine learning models to predict the **loan approval status** of applicants. The goal is to assist financial institutions in automating and improving their decision-making process for loan applications by identifying key predictors and selecting the best-performing model.

The full analysis, including data loading, cleaning, feature engineering, model training, and evaluation, is documented within the **`Predicting_Loan_Approvals_With_ML..ipynb`** Jupyter Notebook.

---

## Dataset
The core data is provided in the **`loan_approval_dataset.csv`** file, containing over 4,200 records. Key features include:
* **Income and Loan Details**: `income_annum`, `loan_amount`, `loan_term`.
* **Creditworthiness**: `cibil_score` (a critical factor in lending).
* **Applicant Assets**: Values for `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, and `bank_asset_value`.
* **Personal Info**: `no_of_dependents`, `education`, and `self_employed` status.
* **Target Variable**: `loan_status` (Approved/Rejected).

---

## Methodology
The notebook follows a robust machine learning pipeline:
1.  **Data Preprocessing**: Handling categorical variables using **Ordinal Encoding** and scaling numerical features using **StandardScaler** to ensure all features contribute equally to model training.
2.  **Model Training**: The processed data was split into training and testing sets, and two distinct classification algorithms were trained:
    * **Logistic Regression**: A simple, highly interpretable linear model.
    * **Decision Tree Classifier**: A non-linear model capable of capturing complex decision boundaries.
3.  **Model Evaluation**: Performance was assessed using **Accuracy Score** (overall correctness) and **F1-Score** (a balanced measure of precision and recall).

---

## Results and Evaluation
The Decision Tree model significantly outperformed the Logistic Regression model, suggesting that non-linear relationships in the data are important for accurate prediction.

| Model | Accuracy Score | F1-Score |
| :--- | :--- | :--- |
| **Logistic Regression** | **94.03%** | **0.9247** |
| **Decision Tree Classifier** | **98.01%** | **0.9731** |

### Confusion Matrix for Logistic Regression (LR)
The confusion matrix for the Logistic Regression model provides a breakdown of correct and incorrect predictions:

![Logistic Regression Confusion Matrix](https://github.com/bassam519/Loan-Approval-Prediction/blob/main/lr%20cm.png?raw=true)

### Confusion Matrix for Decision Tree Classifier (DT)
The Decision Tree matrix shows much higher true positive and true negative counts, confirming its superior performance:

![Decision Tree Confusion Matrix](https://github.com/bassam519/Loan-Approval-Prediction/blob/main/Dt%20cm.png?raw=true)

---

## Conclusion
The **Decision Tree Classifier** achieved an impressive **98.01% accuracy**, making it the recommended model for deployment in a production environment to predict loan approval status.

---

## Repository Contents
| File Name | Description |
| :--- | :--- |
| `Predicting_Loan_Approvals_With_ML..ipynb` | The executable Jupyter Notebook with all the code, analysis, and model results. |
| `loan_approval_dataset.csv` | The raw dataset used for training and testing the models. |
| `README.md` | This documentation file. |

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
