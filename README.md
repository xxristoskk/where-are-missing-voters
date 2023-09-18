# Where Are The Voters?
 Building a classification model to help determine whether or not a county will have a voter turnout less than 38%

### Introduction
Tynan Challenor from Stanford University published their finding in a paper called "Predicting Votes From Census Data". They gathered data from a survey given by Census that took place after the 2016 election, and used Logistic Regression, Support Vector Machine, and Naive Bayes models to predict whether or not a person is likely to vote. This inspired me to see if I could do the same, but rather than predicting an individual's likeliness to vote, I want to predict whether or not a county will have a low voter turnout.

### Exploring The Data
After merging the sources together and creating a definition for the target variable, I performed some exploratory data analysis by comparing a county's average voter turnout with various demographics.

![pairplot](https://github.com/xxristoskk/where-are-missing-voters/blob/master/visuals/pairplot.png)

![pairplot](https://github.com/xxristoskk/where-are-missing-voters/blob/master/visuals/pairplot2.png)

The above indicates there is some correlation with the turnout percentage and education and demographic indicators.

Doing some principal component analysis showed a distinct difference between the Gini Coefficient and the size rank of each county.
![PCA scatter plot](https://github.com/xxristoskk/where-are-missing-voters/blob/master/visuals/scatter_pca.png)

### Selecting The Model
Based on the work from [Tynan Challenor](https://cs229.stanford.edu/proj2017/final-reports/5232542.pdf), I choose to start testing models with Logistic Regression.

### Model performance
I used a grid search to find the best parameters for the two models that had the better initial test performance. To get a quick understanding of what these scores mean, I made a reference of the summary.

* **Accuracy**: overall accuracy
* **Precision**: percent of positive predictions out of overall true-positives
* **Recall**: percent of true low low-turnouts that were predicted as low-turnouts
* **F1**: weighted accuracy between the precision and recall scores
* **ROC AUC score**: optimal percent of true-positive rate vs. false positive rate

#### Logistic Regression
  * Precision: 80%
  * F1: 74%
  * Accuracy: 81%
  * Recall: 68%
  * ROC_AUC: 79%

#### Random Forest
  * Accuracy: 82%
  * Precision: 83%
  * Recall: 70%
  * F1: 73%
  * ROC AUC score: %80

#### XGBoost
  * Accuracy: **83%**
  * Precision: 83%
  * Recall: **73%**
  * F1: **78%**
  * ROC AUC score: **%82**

### Visualized ROC
![img](https://github.com/xxristoskk/where-are-missing-voters/blob/master/visuals/roc_curve.png)

### Most important features considered for XGBoost model
![img](https://github.com/xxristoskk/where-are-missing-voters/blob/master/visuals/important_feats.png)

## Conclusions and Final Thoughts
 * The best results from the XGBoost are promising for future work
 * The overall socioeconomic status of a county's residence has a strong correlation with voter turnout
* Suggestions for better results
* When predicting the 2018 midterm turnout on its own, it had a 93% recall score and 89% overall accuracy
* When predicting the presidential election turnout on its own, it had a 93% recall score, 86% overall accuracy, and a 91% F1 score
