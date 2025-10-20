# ML-Loan-Approval-Prediction

## Overview
This project focuses on building and comparing machine learning models to predict the **loan approval status** of applicants. The goal is to leverage data science to help financial institutions automate and streamline their decision-making process for loan applications.

The full analysis, including data loading, cleaning, feature engineering, and model evaluation, is contained within the **`Predicting_Loan_Approvals_With_ML..ipynb`** Jupyter Notebook.

---

## Dataset
The core data is provided in the **`loan_approval_dataset.csv`** file. It includes key financial and personal attributes of loan applicants, such as:
* `income_annum`
* `loan_amount`
* `cibil_score`
* `residential_assets_value`
* `education`
* `loan_status` (The target variable: Approved/Rejected)

---

## Methodology
The notebook follows a standard machine learning pipeline:
1.  **Exploratory Data Analysis (EDA)** and Data Preprocessing.
2.  **Feature Scaling** and Encoding of Categorical Variables.
3.  **Model Training and Comparison** using two popular classification algorithms:
    * **Logistic Regression**
    * **Decision Tree Classifier**

---

## Results and Evaluation
Model performance was rigorously evaluated using metrics such as Accuracy, F1-Score, and **Confusion Matrices**, which are vital for understanding the type of errors (False Positives and False Negatives) the models make.

### Confusion Matrix for Logistic Regression (LR)
The confusion matrix for the Logistic Regression model:

![Logistic Regression Confusion Matrix](https://github.com/bassam519/Loan-Approval-Prediction/blob/main/lr%20cm.png?raw=true)

### Confusion Matrix for Decision Tree Classifier (DT)
The confusion matrix for the Decision Tree Classifier model:

![Decision Tree Confusion Matrix](https://github.com/bassam519/Loan-Approval-Prediction/blob/main/Dt%20cm.png?raw=true)

---

## Repository Contents
| File Name | Description |
| :--- | :--- |
| `Predicting_Loan_Approvals_With_ML..ipynb` | The executable Jupyter Notebook with all the code, analysis, and model results. |
| `loan_approval_dataset.csv` | The raw dataset used for training and testing the models. |
| `README.md` | This documentation file. |

---

## Setup and Usage
To reproduce the analysis, you will need Python and the necessary libraries.

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
