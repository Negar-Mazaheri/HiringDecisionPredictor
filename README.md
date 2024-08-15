 # Predicting Hiring Decisions in Recruitment Data
This project aims to predict hiring decisions based on a variety of candidate attributes using machine learning models. The data and code are structured to facilitate exploratory data analysis, model training, and evaluation.
## Dataset Description
The [dataset](https://github.com/Negar-Mazaheri/HiringDecisionPredictor/blob/main/recruitment_data.csv) includes candidate information such as demographics, education level, experience, skill scores, personality scores, and recruitment strategies. The target variable is the HiringDecision, indicating the outcome of the hiring process. Detailed descriptions of each feature can be found in the project [notebook](https://github.com/Negar-Mazaheri/HiringDecisionPredictor/blob/main/hiring_prediction%20.ipynb) or in [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data/data).
## Project Structure
The project consists of the following sections:
1. __Data Preprocessing:__ Handling missing values, encoding categorical variables, and normalizing the data.
2. __Exploratory Data Analysis (EDA):__ Visualizing the data to identify patterns and correlations between different features and the hiring decision.
3. __Model Training:__ Building and training machine learning models to predict the hiring decision.
   - Logistic Regression
   - Random Forest Classifier
4. __Model Evaluation:__ Evaluating the performance of the models using metrics such as accuracy, precision, recall, and F1-score.
5. __Feature Importance Analysis:__ Identifying the most important features that influence the hiring decision.
## Prerequisites
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
## Results
The Logistic Regression model achieved an accuracy score of 87.78%, while the Random Forest Classifier achieved an accuracy score of 93.11%. The Random Forest model showed better performance and was able to capture the relationships between features more effectively.
