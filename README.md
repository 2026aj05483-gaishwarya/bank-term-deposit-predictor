# Bank Term Deposit Predictor

Machine Learning Assignment 2: predict whether a client subscribes to a bank term deposit after a direct marketing phone campaign.

## a. Problem statement

The data comes from direct marketing campaigns (phone calls) of a banking institution. Campaigns were based on phone calls, and the same client was often contacted more than once to determine whether the product (a bank term deposit) was subscribed (`yes`) or not (`no`).

This is a binary classification problem. The target column is `y`.

## b. Dataset description

Source (Kaggle input used for this project):

https://www.kaggle.com/code/imkushwaha/term-deposit-prediction/input

Related notebook:

https://www.kaggle.com/code/imkushwaha/term-deposit-prediction

The file used here is `data/bank-additional-full.csv` (UCI Bank Marketing, additional full set), semicolon-separated.

- Instances: 41,188
- Features: 20 input columns plus the target `y`
- Target: `y` — `yes` if the client subscribed to a term deposit, `no` otherwise
- Feature groups:
  - Client: `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`
  - Campaign contact: `contact`, `month`, `day_of_week`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`
  - Macroeconomic context: `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`

Several categorical fields use the value `unknown`. The class distribution is imbalanced, with `no` as the majority class.

## c. GitHub repository link

https://github.com/2026aj05483-gaishwarya/bank-term-deposit-predictor

## d. Models used

All five classifiers were trained on the same stratified split and scored on the validation set. About 11.3% of clients subscribed, so accuracy near 0.89 is close to always predicting `no`. AUC, F1, and MCC separate the models.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9124 | 0.9363 | 0.6886 | 0.4066 | 0.5113 | 0.4863 |
| Decision Tree | 0.8877 | 0.7310 | 0.5014 | 0.5287 | 0.5147 | 0.4514 |
| KNN | 0.9032 | 0.8659 | 0.6047 | 0.4066 | 0.4863 | 0.4455 |
| Naive Bayes | 0.8309 | 0.8261 | 0.3505 | 0.5876 | 0.4391 | 0.3630 |
| Support Vector Machine | 0.9116 | 0.9363 | 0.7301 | 0.3420 | 0.4658 | 0.4609 |

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Highest accuracy (0.9124) and highest MCC (0.4863). AUC is tied with the SVM at 0.9363. Precision is 0.6886, so most predicted subscribers are real subscribers, while recall of 0.4066 still misses many `yes` cases. |
| Decision Tree | Highest recall among the stronger models (0.5287) and a slightly higher F1 (0.5147) than logistic regression, but accuracy 0.8877 is only about the majority-class rate and AUC 0.7310 is the weakest ranking quality. |
| KNN | Accuracy 0.9032 and AUC 0.8659 sit in the middle. Precision 0.6047 and recall 0.4066 match logistic regression on recall but with more false positives, so F1 (0.4863) and MCC (0.4455) are lower. |
| Naive Bayes | Highest recall (0.5876) and the lowest accuracy (0.8309), precision (0.3505), F1 (0.4391), and MCC (0.3630). It finds more subscribers by marking many non-subscribers as `yes`. |
| Support Vector Machine | Accuracy 0.9116 and AUC 0.9363 match logistic regression. Precision is the best of the five (0.7301), but recall is the lowest (0.3420), so F1 falls to 0.4658. |
| Overall, winner for this dataset? | Logistic Regression. It ties the best AUC, leads MCC and accuracy, and keeps a better balance of precision and recall than the SVM. The decision tree's F1 is only marginally higher and its AUC is much worse. |
