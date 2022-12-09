# Customer Churn Logistic Regression

PySpark with logistic regression predicting if customers will exit a bank service.

---

## Problem Overview

Being able to predict if a customer is going to end his account on a service is crucial for companies that need to build customer loyalty and maintain their revenue. Keeping customers is also cheaper than acquiring new ones. [Churn] prediction has been a successful endeavor for machine learning models of different levels of complexity.

One common issue related with churn prediction is the lack of available data for customers exiting the service. This leads to a natural class imbalance: datasets have much more instances about customers that did not exit than about customers that did exit. [The dataset analyzed in this repository] also has this issue. It is composed of 10,000 instances with 10 independent variables about customers from a bank and 1 binary dependent variable ("Exited") which indicates if the customer left the service or not. The next table provides a summary on each attribute in this dataset.

| Attribute | Summary |
|:---------:|:-------:|
| CreditScore | Credit score calculated by the bank. |
| Geography | Country where the customer resides: Germany, France or Spain. |
| Gender | Gender between Female and Male. |
| Age | Customer's age. |
| Tenure | Amount of years the customer has been with the bank. |
| Balance | Current balance. |
| NumOfProducts | Number of bank products used. |
| HasCrCard | Credit card status (0: does not have; 1: has credit card). |
| IsActiveMember | Active membership status (0: not active; 1: active). |
| EstimatedSalary | Customer's estimated salary. |
| Exited | Customer status (0: has not left the bank; 1: has left the bank). |

## Analysis Introduction

Logistic regression models are relatively easy to train when there are few independent variables to learn from, especially considering modern day computing power. It is also usually easier to interpret regression models than other machine learning models, such as neural networks. Besides being cheap to train and interpretable, logistic regressors can readily be adapted to deal with datasets having imbalanced classes. These 3 reasons make logistic regressors good candidates for predicting churn and, therefore, are the chosen ones for this project.

We perform two distinct analyses using two logistic regressors. In the first scenario, we ignore class imbalance and check how well the model learns to represent the data. In the second scenario, we compensate for class imbalance by performing a weighted logistic regression. In this case, the minority class (positive for churn) is given more weight. The analysis is entirely performed using only [PySpark].

## Brief Result Analysis

The class imbalance level happens at a factor of 4, meaning that the majority class has 4 times more instances in the dataset than the minority class. This considerable amount of class imbalance is the main issue when dealing with this dataset. The testing set results confirm that, with the weighted logistic regressor achieving nearly 100% performance across all metrics evaluated: [weighted precision, weighted recall, accuracy, F1-score and others]. Meanwhile, the original model, which does not account for class imbalance, achieves around 75% performance across all metrics. Not a bad result, but far from perfect.

Note that our analysis is tailored to get better results at recall-related metrics. This is because identifying all customer churn (with a low number of false negatives) is more important for churn prediction than making precise predictions (low number of false positives). This way, the company can work its best at trying to keep the customer that is predicted to leave, even if unnecessarily spending some resources at false positives that do not intend to leave.

[//]: #

[the dataset analyzed in this repository]: <https://www.kaggle.com/code/mathchi/churn-problem-for-bank-customer>
[churn]: <https://www.paddle.com/resources/customer-churn>
[PySpark]: <https://spark.apache.org/docs/latest/api/python/>
[weighted precision, weighted recall, accuracy, F1-score and others]: <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.evaluation.MulticlassMetrics.html>
