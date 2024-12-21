# README.md

## **Employee Retention Prediction Project**

Welcome to the **Employee Retention Prediction** project! This repository contains a comprehensive analysis and predictive modeling aimed at identifying factors that contribute to employee retention. By leveraging data science techniques, we aim to help organizations understand and enhance their employee retention strategies.

---

## **Table of Contents**

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation Instructions](#installation-instructions)
- [Data Description](#data-description)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## **Project Overview**

In today's competitive job market, retaining talented employees is crucial for organizational success. This project utilizes machine learning algorithms to predict employee retention based on various features such as demographic information, job characteristics, and company attributes. The insights gained can assist HR departments in making informed decisions.

---

## **Technologies Used**

This project employs the following technologies:

- **Python**: Programming language for data analysis and modeling.
- **Pandas**: Data manipulation and analysis library.
- **NumPy**: Library for numerical computations.
- **Matplotlib & Seaborn**: Libraries for data visualization.
- **Scikit-learn**: Machine learning library for model building.
- **XGBoost & LightGBM**: Advanced gradient boosting frameworks.
- **Optuna**: Hyperparameter optimization framework.

---

## **Installation Instructions**

To set up this project locally, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/employee-retention-prediction.git
   cd employee-retention-prediction
   ```

2. **Install Required Libraries**
   Ensure you have Python installed, then run:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn xgboost lightgbm optuna
   ```

3. **Download the Dataset**
   Place the dataset `aug_train.csv` in the root directory of the project.

---

## **Data Description**

The dataset consists of various features related to employees, including:

| Feature                     | Description                                      |
|-----------------------------|--------------------------------------------------|
| enrollee_id                 | Unique identifier for each employee              |
| city                        | City of employment                                |
| city_development_index      | Index indicating city development level          |
| gender                      | Gender of the employee                            |
| relevant_experience         | Relevant work experience                          |
| enrolled_university         | Enrollment status in university                   |
| education_level             | Highest education level attained                  |
| major_discipline            | Major discipline studied                          |
| experience                  | Years of work experience                          |
| company_size                | Size of the company                              |
| company_type                | Type of company (e.g., private, funded startup)  |
| last_new_job                | Duration since last job                           |
| training_hours              | Total training hours completed                    |
| target                      | Target variable indicating retention (1) or not (0) |

---

## **Usage**

To run the prediction model, execute the following script:
```bash
employeeRetentionPredection.ipynb
```
This will load the data, preprocess it, train the model, and output predictions regarding employee retention.

## **Preprocessing, Hyperparameter Tuning, and Model Pickling**

In this section, we discuss the essential steps taken in the **Employee Retention Prediction** project, focusing on data preprocessing, hyperparameter tuning for the base model, and the process of pickling the trained model for future use.

### **Data Preprocessing**

Data preprocessing is a critical step in preparing the dataset for modeling. The following steps were performed:

1. **Loading Data**: The dataset was loaded using Pandas:
   ```python
   import pandas as pd
   df = pd.read_csv("aug_train.csv")
   ```

2. **Handling Missing Values**: We checked for missing values and decided to fill or drop them based on their significance:
   ```python
   df.fillna(method='ffill', inplace=True)  # Forward fill for simplicity
   ```

3. **Encoding Categorical Variables**: Categorical features were encoded using techniques such as one-hot encoding or label encoding to convert them into numerical format:
   ```python
   df = pd.get_dummies(df, columns=['gender', 'relevant_experience', 'enrolled_university', 'education_level', 'major_discipline', 'company_size', 'company_type', 'last_new_job'])
   ```

4. **Feature Scaling**: Numerical features were standardized or normalized to ensure uniformity in scale:
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   df[['training_hours']] = scaler.fit_transform(df[['training_hours']])
   ```

5. **Splitting the Data**: The dataset was split into training and testing sets to evaluate model performance:
   ```python
   from sklearn.model_selection import train_test_split
   X = df.drop('target', axis=1)
   y = df['target']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

### **Base Model Hyperparameter Tuning**

To improve model performance, hyperparameter tuning was conducted using techniques such as Grid Search or Random Search. For instance, we utilized `Optuna` for efficient hyperparameter optimization:

1. **Defining the Objective Function**: We defined an objective function that takes hyperparameters as input and returns model performance metrics:
   ```python
   import optuna

   def objective(trial):
       model = XGBClassifier(
           max_depth=trial.suggest_int('max_depth', 3, 10),
           learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
           n_estimators=trial.suggest_int('n_estimators', 50, 300)
       )
       model.fit(X_train, y_train)
       preds = model.predict(X_test)
       return accuracy_score(y_test, preds)
   ```

2. **Running the Optimization**: The optimization process was executed to find the best hyperparameters:
   ```python
   study = optuna.create_study(direction='maximize')
   study.optimize(objective, n_trials=100)
   ```

3. **Best Parameters**: After running the optimization, the best parameters were retrieved:
   ```python
   best_params = study.best_params
   print("Best Hyperparameters:", best_params)
   ```

### **Pickling the Model**

Once the model was trained and optimized, it was essential to save it for future use without needing to retrain it each time:

1. **Model Training**: Train the final model using the best parameters obtained from hyperparameter tuning:
   ```python
   final_model = XGBClassifier(**best_params)
   final_model.fit(X_train, y_train)
   ```

2. **Pickling the Model**: The trained model was saved using Python's `pickle` module:
   ```python
   import pickle

   with open('employee_retention_model.pkl', 'wb') as file:
       pickle.dump(final_model, file)
   ```

3. **Loading the Model**: To use the saved model later for predictions:
   ```python
   with open('employee_retention_model.pkl', 'rb') as file:
       loaded_model = pickle.load(file)
   
   predictions = loaded_model.predict(X_test)
   ```


By following these steps in preprocessing, hyperparameter tuning, and pickling the model, we ensure that our employee retention prediction system is robust and ready for deployment or further web app developmenat.

---

## **Contributing**

We welcome contributions! If you have suggestions or improvements, please fork the repository and submit a pull request.

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Thank you for checking out this repository! We hope you find it useful in understanding and predicting employee retention. For any questions or feedback, feel free to reach out!

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/12626707/acb250c1-ca4d-4134-a91e-6b6bfd423f75/employeeRetentionPredection.ipynb
