# Supervised Machine learning and Credit Risk

_Machine learning is the use of statistical algorithms to perform tasks such as learning from data patterns and making predictions_

# Overview of the analysis
In 2019, more than 19 million Americans had at least one unsecured personal loan. That's a record-breaking number! Personal lending is growing faster than credit card, auto, mortgage, and even student debt. With such incredible growth, FinTech firms are storming ahead of traditional loan processes. By using the latest machine learning techniques, these FinTech firms can continuously analyze large amounts of data and predict trends to optimize lending.
In this project, we used Python to build and evaluate several machine learning models to predict credit card risk. Being able to predict credit risk with machine learning algorithms can help banks and financial institutions predict anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud.

### Resources:
- Python
- Scikit-learn Library 
- imbalanced-learn  Library

# Results

Using our knowledge of the imbalanced-learn and scikit-learn libraries w were able to evaluate six machine learning models by using resampling to determine which is better at predicting credit risk. 

- RandomOverSampler
- SMOTE Oversampling
- ClusterCentroids Undersampler 
- SMOTEENN Combination (Over and Under) Sampling
- BalancedRandomForestClassifier
- EasyEnsembleClassifier

Process of building and evaluating modeals are pretty muchch the same, First, prepare data by we cleaning it, than we split the data into training and testing datasets using train_test_split method imported from sklearn.model_selection. Than  we choose sampling method it can be RandomOverSampler, ClusterCentroids, SMOTEENN etc. 
```
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
Counter(y_resampled)
```
```
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(random_state=1, sampling_strategy='auto').fit_resample(X_train, y_train)
Counter(y_resampled)
```
```
from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=1)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
Counter(y_resampled)
```
After that we train out resampled data. 
```
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled) 
```
Than we display confusing matrix on a predicted data. 
```
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)
```
Next step is where we calculate the balanced accuracy score
```
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(y_test, y_pred)
```
Last step we display classification report
```
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
```

For our  first part of project we compare two oversampling algorithms  RandomOverSampler and SMOTE.
Using these algorithms, we resampled the dataset, viewed the count of the target classes, trained a logistic regression classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.

### **RandomOverSampler** 
- 0.6573009382322703 balanced accuracy 
- 1% high risk precision score, and a 100% low risk prescision
- recall 

![RandomOverSampler](https://github.com/kossakova/Credit_Risk_Analysis/blob/main/PNG/RandomOverSampler.png)

### **SMOTE**

![SMOTE](https://github.com/kossakova/Credit_Risk_Analysis/blob/main/PNG/SMOTE.png)


Next, we tested an undersampling ClusterCentroids algorithms to determine which algorithm results in the best performance compared to the oversampling algorithms above. 
### **ClusterCentroids** 


![ClusterCentroids](https://github.com/kossakova/Credit_Risk_Analysis/blob/main/PNG/ClusterCentroids.png)

### **SMOTEENN** 
In the second part of the project we twsted combinatorial approach of over- and undersampling with the SMOTEENN algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms frompart tho above. Using the SMOTEENN algorithm, we resampled the dataset, viewed the count of the target classes, trained a logistic regression classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.


![SMOTEENN](https://github.com/kossakova/Credit_Risk_Analysis/blob/main/PNG/SMOTEENN.png)


In the final part of the project we trained and compared two different ensemble classifiers, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk and  to determine which algorithm results in the best performance . Using both algorithms, we resampled the dataset, viewed the count of the target classes, trained the ensemble classifier, calculateed the balanced accuracy score, generated a confusion matrix, and generated a classification report.
### **BalancedRandomForestClassifier**  

![BalancedRandomForestClassifier](https://github.com/kossakova/Credit_Risk_Analysis/blob/main/PNG/BalancedRandomForestClassifier.png)

### **EasyEnsembleClassifier**  


![EasyEnsembleClassifier](https://github.com/kossakova/Credit_Risk_Analysis/blob/main/PNG/EasyEnsembleClassifier.png)



Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.
# Summary
3.	
4.	Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.
