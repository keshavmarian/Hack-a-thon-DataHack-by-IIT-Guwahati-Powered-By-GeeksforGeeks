# Hack-a-thon-DataHack-by-IIT-Guwahati-Powered-By-GeeksforGeeks
Vaccine Prediction This project aims to predict the likelihood of individuals receiving XYZ and seasonal flu vaccines using machine learning techniques. The repository includes data preprocessing, model training, evaluation, and prediction steps. The model is evaluated using ROC AUC, accuracy, precision, recall, and F1 score metrics.
# Overview of  Machine Learning Project: Vaccine Uptake Prediction

## 1. Import Libraries
First, we imported necessary libraries for data manipulation, model training, and evaluation. This included libraries like pandas for data handling, numpy for numerical operations, and scikit-learn for building and evaluating machine learning models.

## 2. Load the Datasets
We loaded the training and test datasets, as well as the submission format. The training set contains features and labels, while the test set contains only features. The submission format is used to structure our prediction results.

## 3. Define Feature Columns
Next, we identified which columns in our dataset were categorical (e.g., age group, education) and which were numeric (e.g., concern about XYZ, knowledge about XYZ). This distinction is crucial for appropriate preprocessing steps.

## 4. Preprocessing Pipelines
We created separate preprocessing pipelines for numeric and categorical features:

- **Numeric features:** We handled missing values by imputing them with the median and scaled the features to ensure they have a mean of 0 and a standard deviation of 1.
- **Categorical features:** We handled missing values by imputing them with a constant value ('missing') and applied one-hot encoding to convert categorical values into a numerical format suitable for our model.

We then combined these preprocessing steps using a `ColumnTransformer`.

## 5. Define the Model
We constructed a machine learning pipeline that first preprocesses the data and then applies a `MultiOutputClassifier` with a `RandomForestClassifier`. This model can handle multiple target variables (i.e., predicting both XYZ and seasonal vaccine uptake simultaneously).

## 6. Split Data into Train and Validation Sets
To evaluate our model's performance, we split the training data into training and validation sets. This allows us to train the model on one portion of the data and validate it on another, ensuring that our model generalizes well to unseen data.

## 7. Fit the Model
We trained the model using the training set. This step involved fitting the model to the data, allowing it to learn the patterns associated with vaccine uptake.

## 8. Predict Probabilities for Validation Set
After training, we predicted the probabilities of each class (i.e., likelihood of taking XYZ vaccine and seasonal vaccine) on the validation set. This helps us understand how confident the model is about its predictions.

## 9. Evaluate the Model
We evaluated the model using various metrics:

- **ROC AUC:** This metric measures the model's ability to distinguish between classes, with a higher score indicating better performance.
- **Accuracy:** The ratio of correctly predicted instances to the total instances.
- **Precision:** The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall:** The ratio of correctly predicted positive observations to all the observations in the actual class.
- **F1 Score:** The weighted average of precision and recall, providing a balance between the two.

By calculating these metrics, we assessed the model's performance comprehensively.

## 10. Predict Probabilities for Test Set
With the model trained and validated, we applied it to the test set to predict the probabilities for each class. This step involved processing the test features and generating predictions.

## 11. Preparatiobn of Submission File
Finally, we prepared a submission file. This file included the `respondent_id`, along with the predicted probabilities for both the XYZ and seasonal vaccines. We structured it according to the submission format and saved it as `submission.csv`.

## Conclusion
This project demonstrated a systematic approach to preprocessing data, training a machine learning model, evaluating its performance, and making predictions. The use of various metrics ensured a thorough evaluation, and the final predictions were formatted for submission.
