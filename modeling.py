import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, classification_report,recall_score, accuracy_score, roc_curve, auc, roc_auc_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

## Improvements to make:
## Put labels on all the states to make a multiclass variable

idk = GridSearchCV(RandomForestClassifier(), param_grid={'n_estimators': [10, 100, 1000]})
idk.fit(xTrain,yTrain)
pd.DataFrame(idk.cv_results_)

idk.score(xTest,yTest)

q = idk.predict(xTest)
calc_scores('boosted',yTest,q)

##Define target and labels
target = big_df['low_turnout']
big_df.drop('low_turnout',axis=1,inplace=True)
X = big_df
## Set up training data
xTrain,xTest,yTrain,yTest = train_test_split(X,target,test_size=.25)
## Scale the training data
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaledTrained = scaler.fit_transform(xTrain)
# scaledTest = scaler.fit_transform(xTest)
# scaled_df_train = pd.DataFrame(scaledTrained,columns=big_df.columns)
# scaled_df_train.head()


rf = RandomForestClassifier(max_features=7,max_depth=3,min_samples_split=.3,min_samples_leaf=.3)
rf.fit(xTrain,yTrain)
rf_y = rf.predict(xTest)

## Run models function to calculate the scores from Random Forest, SVM, Decision Tree, and Logistic Regression
run_models(big_df,big_df,target)

################################################################
## Training the Classifier
# Identify the optimal tree depth for given data
depth = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for n in depth:
    dt = DecisionTreeClassifier(criterion='entropy',max_depth=n)
    dt.fit(xTrain,yTrain)
    #train
    train_pred = dt.predict(xTrain)
    fp, tp, thresholds = roc_curve(yTrain,train_pred)
    roc_auc = auc(fp,tp)
    train_results.append(roc_auc)
    #test
    test_pred = dt.predict(xTest)
    fp,tp,thresholds = roc_curve(yTest,test_pred)
    roc_auc = auc(fp,tp)
    test_results.append(roc_auc)

plt.figure(figsize=(12,6))
plt.plot(depth, train_results, 'b', label='Train AUC')
plt.plot(depth, test_results, 'r', label='Test AUC')
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.legend()
plt.show()

# Identify the optimal min-samples-split for given data
# Identify the optimal min-samples-split for given data
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
   dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=min_samples_split)
   dt.fit(xTrain, yTrain)
   train_pred = dt.predict(xTrain)
   false_positive_rate, true_positive_rate, thresholds =    roc_curve(yTrain, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = dt.predict(xTest)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(yTest, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
plt.figure(figsize=(12,6))
plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
plt.xlabel('Min. Sample splits')
plt.legend()
plt.show()

min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
   dt = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=min_samples_leaf)
   dt.fit(xTrain, yTrain)
   train_pred = dt.predict(xTrain)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(yTrain, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = dt.predict(xTest)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(yTest, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)


plt.figure(figsize=(12,6))
plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')
plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')
plt.ylabel('AUC score')
plt.xlabel('Min. Sample Leafs')
plt.legend()
plt.show()

# Find the best value for optimal maximum feature size
max_features = list(range(1,xTrain.shape[1]))
train_results = []
test_results = []
for max_feature in max_features:
   dt = DecisionTreeClassifier(criterion='entropy', max_features=max_feature)
   dt.fit(xTrain, yTrain)
   train_pred = dt.predict(xTrain)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(yTrain, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = dt.predict(xTest)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(yTest, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)


plt.figure(figsize=(12,6))
plt.plot(max_features, train_results, 'b', label='Train AUC')
plt.plot(max_features, test_results, 'r', label='Test AUC')
plt.ylabel('AUC score')
plt.xlabel('max features')
plt.legend()
plt.show()
##################################################################
def create_dict(df,col):
    """creates a list of df's"""
    unique_col = df[col].unique()
    df_list = []
    for val in unique_col:
        df_list.append(df[df[col]==val])
    return df_list
####################################################################
run_models(big_df,X,target)
###############################################################
def run_models(df,X,y):
    ## Set up training data
    xTrain,xTest,yTrain,yTest = train_test_split(X,y,test_size=.2)
    ## Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X=xTrain,y=yTrain)
    dt_pred = dt.predict(xTest)
    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(xTrain,yTrain)
    lr_pred = lr.predict(xTest)
    # svm
    svm = LinearSVC(C=1, loss="hinge",random_state=10)
    svm.fit(xTrain,yTrain)
    svm_p = svm.predict(xTest)
    rf = RandomForestClassifier(n_estimators=100)
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
def rm(all):
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
