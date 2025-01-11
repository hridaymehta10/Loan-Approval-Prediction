# Loan-Approval-Prediction

Loan Approval Prediction using Machine Learning

Project Overview
This project focuses on predicting whether a loan will be approved based on various applicant details using machine learning techniques. The dataset used contains information about applicants such as their income, credit history, loan amount, etc. The goal is to build a model that accurately classifies loan approval status, helping financial institutions make faster and more reliable decisions.


Table of Contents
•	Installation
•	Dataset
•	Data Preprocessing
•	Model Building
•	Model Evaluation
•	Usage
•	Conclusion
•	License


Installation
Prerequisites
Ensure you have Python 3.x installed, along with the following libraries:
•	pandas
•	numpy
•	matplotlib
•	seaborn
•	scikit-learn
You can install the required dependencies using the following:
bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn

Dataset
The dataset used in this project is publicly available and contains information related to applicants and their loan approval status. Some of the key columns include:
•	Gender: Applicant's gender
•	Married: Marital status
•	Dependents: Number of dependents
•	Education: Level of education
•	Self_Employed: Whether the applicant is self-employed
•	ApplicantIncome: Income of the applicant
•	CoapplicantIncome: Income of the co-applicant (if any)
•	LoanAmount: The requested loan amount
•	Loan_Amount_Term: Duration of the loan
•	Credit_History: Credit history of the applicant (1: good, 0: poor)
•	Property_Area: The area where the applicant resides
•	Loan_Status: The loan approval status (Target variable: 1 for approved, 0 for not approved)


Data Preprocessing
The dataset was preprocessed to prepare it for machine learning models:
1.	Handling Missing Values: Missing values were filled using appropriate techniques like the mean for numerical columns and the mode for categorical columns.
2.	Encoding Categorical Variables: Categorical variables (like Gender, Married, Education, etc.) were encoded using Label Encoding to convert them into numeric format.
3.	Data Visualization: Various visualizations were created using matplotlib and seaborn to understand relationships between variables and check for correlations.
4.	Feature Engineering: New features were created where applicable, such as combining ApplicantIncome and CoapplicantIncome into a TotalIncome feature.


Model Building
Several machine learning models were trained to predict loan approval:
1.	Random Forest Classifier
2.	K-Nearest Neighbors (KNN)
3.	Support Vector Machine (SVM)
4.	Logistic Regression
Each model was trained on the dataset, and hyperparameters were tuned to achieve the best performance.


Model Evaluation
The models were evaluated based on their accuracy on the test set. The following results were observed:
•	Random Forest Classifier: Achieved the highest accuracy of 82% on the test set.
•	Other models showed varying levels of performance, with Random Forest emerging as the best.
The Random Forest model was selected as the final model for deployment due to its superior performance in predicting loan approval.


Usage
To run the project:
1.	Clone the repository:
bash
Copy code
git clone https://github.com/hridaymehta10/loan-approval-prediction.git
2.	Navigate to the project directory:
bash
Copy code
cd loan-approval-prediction
3.	Run the script to train the model and evaluate it:
bash
Copy code
python loan_approval.py
You can modify the loan_approval.py script to input your own data or tweak the preprocessing steps to suit your needs.


Conclusion
This project demonstrates how machine learning models can be used to predict loan approval status based on a variety of applicant features. By utilizing algorithms like Random Forest, we can accurately classify loan approval and assist financial institutions in their decision-making process.
Future improvements can include:
•	Hyperparameter tuning for better performance
•	Cross-validation to improve the robustness of the models
•	Deploying the model for real-time loan approval predictions


License
This project is licensed under the MIT License - see the LICENSE file for details.

