# Supervised Machine learning and Credit Risk

_"Machine learning is the use of statistical algorithms to perform tasks such as learning from data patterns and making predictions"_

_"In supervised learning, first a model is initiated, or a template for the algorithm is created. Then it will analyze the data and attempt to learn patterns, which is also called fitting and training. After the data has been fit and trained, it will then make predictions."_

_"In supervised learning, the input data already has a paired outcome, which is plugged in to train the model to predict outcomes in new datasets. For example, we want to build a model that, when given unfamiliar data, can accurately predict the outcomes"_

![machine-learning-separator-1](https://github.com/kossakova/Credit_Risk_Analysis/blob/main/PNG/machine-learning-separator-1.jpg)

# Overview of the analysis
In 2019, more than 19 million Americans had at least one unsecured personal loan. That's a record-breaking number! Personal lending is growing faster than credit card, auto, mortgage, and even student debt. With such incredible growth, FinTech firms are storming ahead of traditional loan processes. By using the latest machine learning techniques, these FinTech firms can continuously analyze large amounts of data and predict trends to optimize lending.
In this project, we used Python to build and evaluate several machine learning models to predict credit card risk. Being able to predict credit risk with machine learning algorithms can help banks and financial institutions predict anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud.

### Resources:
- Python
- Scikit-learn Library 
- Imbalanced-learn  Library

# Algorithms

Using our knowledge of the imbalanced-learn and scikit-learn libraries w were able to evaluate six machine learning models by using resampling to determine which is better at predicting credit risk. 

- RandomOverSampler
- SMOTE Oversampling
- ClusterCentroids Undersampler 
- SMOTEENN Combination (Over and Under) Sampling
- BalancedRandomForestClassifier
- EasyEnsembleClassifier

Process of building and evaluating models are pretty much the same. First, prepare data by we cleaning it, than we split the data into training and testing datasets using ```train_test_split``` method imported from ```sklearn.model_selection```. Than we choose sampling method it can be RandomOverSampler, ClusterCentroids, SMOTEENN etc. 
```
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
Counter(y_resampled)
```
Next, use the resampled data to train a logistic regression model.
```
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled) 
```
Next, we display confusing matrix on a predicted data. 
```
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)
```
Calculate the balanced accuracy score from ```sklearn.metrics```
```
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(y_test, y_pred)
```
Last step we generate and display classification report 
```
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
```
# Results

For our first part of project we compare two oversampling algorithms RandomOverSampler and SMOTE.
Using these algorithms, we resampled the dataset, viewed the count of the target classes, trained a logistic regression classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.

### **RandomOverSampler** 
- 0.6573009382322703 balanced accuracy 
- 1% high risk precision score, and a 100% low risk precision
- 71% high risk recall score, and 60% low risk recall 

![RandomOverSampler](https://github.com/kossakova/Credit_Risk_Analysis/blob/main/PNG/RandomOverSampler.png)

### **SMOTE**
- 0.6622479600626106 balanced accuracy score
- same as RandomOverSampler 1% high risk precision score, and a 100% low risk precision
- 63% high risk recall score, and 69% low risk recall  

![SMOTE](https://github.com/kossakova/Credit_Risk_Analysis/blob/main/PNG/SMOTE.png)

### **ClusterCentroids** 
Next, we tested an undersampling ClusterCentroids algorithms to determine which algorithm results in the best performance compared to the oversampling algorithms above. 
- lower balanced accuracy score of 0.5447339051023905
- same as over- and under-sampling 1% high risk precision score, and a 100% low risk precision
- 72% high risk recall score, and low risk recall of 57% closest to RandomOverSampler

![ClusterCentroids](https://github.com/kossakova/Credit_Risk_Analysis/blob/main/PNG/ClusterCentroids.png)

### **SMOTEENN** 
In the second part of the project, we tested combinatorial approach of over- and under sampling with the SMOTEENN algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from methods above. Using the SMOTEENN algorithm, we resampled the dataset, viewed the count of the target classes, trained a logistic regression classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.
- not a significant balanced accuracy difference, 0.644711676499736
- same as RandomOverSampler and SMOTE 1% high risk precision score, and a 100% low risk precision
- 69% high risk recall score, and lower low risk recall of 40%

![SMOTEENN](https://github.com/kossakova/Credit_Risk_Analysis/blob/main/PNG/SMOTEENN.png)

### **BalancedRandomForestClassifier**  
In the final part of the project we trained and compared two different ensemble classifiers, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk and to determine which algorithm results in the best performance . Using both algorithms, we resampled the dataset, viewed the count of the target classes, trained the ensemble classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.
- 0.7885466545953005 balanced accuracy score
- slightly higher high risk precision score of 3%, and same 100% low risk precision
- 70% high risk recall score, and higher low risk recall of 87%

![BalancedRandomForestClassifier](https://github.com/kossakova/Credit_Risk_Analysis/blob/main/PNG/BalancedRandomForestClassifier.png)

### **EasyEnsembleClassifier**  
- the most significant balanced accuracy score 0.9316600714093861 
- higher high risk precision score of 9%, and same 100% low risk precision
- overall highest high risk recall score of 92%, and higher low risk recall of 94%

![EasyEnsembleClassifier](https://github.com/kossakova/Credit_Risk_Analysis/blob/main/PNG/EasyEnsembleClassifier.png)

# Summary
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning. the models, justify your reasoning.
In this module we and evaluated several machine learning models on the credit card credit dataset. We oversampled the data using the RandomOverSampler and SMOTE algorithms, and undersampled the data using the ClusterCentroids algorithm. Then, we used a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Lastly, we compared two machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. 

Our analysis showed us that BalancedRandomForestClassifier and EasyEnsembleClassifier had over all best performance, other 4 models had almost the same results. EasyEnsembleClassifier gave us the most significant balanced accuracy score of 0.9316600714093861 over other models with a score around 0.70- 0.50. Also highest high risk precision score of 9% over other models with score of 1-3%, same for the sensitivity scores of 92 and 94 percent.  To see how our scores change we could also perform machine learning with Random Forest and Decision Trees models. 
