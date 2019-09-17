import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc, roc_auc_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

## Improvements to make:
## Put labels on all the states to make a multiclass variable

##Define target and labels
target = one_hot['low_turnout']
one_hot.drop('low_turnout',axis=1,inplace=True)
X = one_hot
## Set up training data
xTrain,xTest,yTrain,yTest = train_test_split(X,target,test_size=.3)
## Scale the training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaledTrained = scaler.fit_transform(xTrain)
scaledTest = scaler.fit_transform(xTest)
scaled_df_train = pd.DataFrame(scaledTrained,columns=one_hot.columns)
# scaled_df_train.head()

## Run models function to calculate the scores from Random Forest, SVM, Decision Tree, and Logistic Regression
run_models(one_hot,one_hot,target)


##################################################################
def create_dict(df,col):
    """creates a list of df's"""
    unique_col = df[col].unique()
    df_list = []
    for val in unique_col:
        df_list.append(df[df[col]==val])
    return df_list
####################################################################
###############################################################
def run_models(df,X,y):
    ## Set up training data
    xTrain,xTest,yTrain,yTest = train_test_split(X,y,test_size=.25)
    ## Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X=xTrain,y=yTrain)
    dt_pred = dt.predict(xTest)
    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(xTrain,yTrain)
    lr_pred = lr.predict(xTest)
    # svm
    svm = Pipeline([
                      ("scaler", StandardScaler()),
                      ("linear_svc", LinearSVC(C=1, loss="hinge")),
                      ])

    svm.fit(xTrain,yTrain)
    svm_p = svm.predict(xTest)
    rf = RandomForestClassifier()
    rf.fit(xTrain,yTrain)
    rf_y = rf.predict(xTest)
    # Calculate scores
    dt_s = calc_scores(':Decision Tree',yTest,dt_pred)
    lr_s = calc_scores(':Logistic Regression',yTest,lr_pred)
    svm_s = calc_scores(':SVM',yTest,svm_p)
    rf_s = calc_scores(':Random Forest',yTest,rf_y)
    scores = [dt_s,lr_s,svm_s,rf_s]
    for j in scores:
        print(j)

## Function that determines which model performs best
## Needs more work
def find_best(lst):
    best_precision = [0]
    best_f1 = [0]
    best_accuracy = [0]
    best_recall = [0]
    best_roc_auc_score = [0]
    for n in lst:
        i = 0
        while i < 5:
            if n[0] > best_precision[0]:
                best_precision[0] == n[0]
                best_precision.append(n[5])
            i+=1
            if n[1] > best_f1[0]:
                best_f1[0] == n[1]
                best_f1.append(n[5])
            i+=1
            if n[2] > best_accuracy[0]:
                best_accuracy[0] == n[2]
                best_accuracy.append(n[5])
            i+=1
            if n[3] > best_recall[0]:
                best_recall[0] == n[3]
                best_recall.append(n[5])
            i+=1
            if n[4] > best_roc_auc_score[0]:
                best_roc_auc_score[0] == n[4]
                best_roc_auc_score.append(n[5])
            i+=1
        return [best_precision,best_f1,best_accuracy,best_recall,best_roc_auc_score]



## Calculates all the scores and returns them in a list that includes the model name
## Needs more work
def calc_scores(model,yTest,yPred):
    p = precision_score(yTest,yPred)
    f1 = f1_score(yTest,yPred)
    a = accuracy_score(yTest,yPred)
    r = recall_score(yTest,yPred)
    roc = roc_auc_score(yTest,yPred)
    return [p,f1,a,r,roc,model]



## Function that will run models on every county
## Needs more work
def run_models(all):
    for d in all:
        best_precision = 0
        best_f1 = 0
        best_accuracy = 0
        best_recall = 0

        target = d['low_turnout']
        d = d.drop('low_turnout',axis=1)
        state = list(d['state'])[0]
        d = d[['gini_coefficient',
                              'child_poverty_living_in_families_below_the_poverty_line','uninsured','production_transportation_and_material_moving_occupations',
                              'poverty_rate_below_federal_poverty_threshold','construction_extraction_maintenance_and_repair_occupations','total_population',
                              'less_than_high_school','at_least_bachelor_s_degree','adults_65_and_older_living_in_poverty','unemployment','graduate_degree']]
        X = d
        y = target
        xTrain,xTest,yTrain,yTest = train_test_split(X,y,test_size=.25)
        dt = DecisionTreeClassifier()
        dt.fit(X=xTrain,y=yTrain)
        dt_pred = dt.predict(xTest)
        scaler = StandardScaler()
        scaledTrained = scaler.fit_transform(xTrain)
        scaledTest = scaler.fit_transform(xTest)
        scaled_df_train = pd.DataFrame(scaledTrained,columns=d.columns)
        knn = KNeighborsClassifier()
        knn.fit(scaled_df_train,yTrain)
        test_preds = knn.predict(scaledTest)
        print(f'{state}:')
        print_metrics(yTest, test_preds)
