from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, classification_report,recall_score, accuracy_score, roc_curve, auc, roc_auc_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier





# XGboot !!

xg = XGBClassifier(n_estimators=1000,max_depth=5,min_samples_split=.3,max_features=6)
xg.fit(xTrain,yTrain)
xg_pred = xg.predict(xTest)
confusion_matrix(yTest,xg_pred)
calc_scores(yTest,xg_pred)
xg_big = xg.predict(valid_df)
calc_scores(target,xg_big)
confusion_matrix(target,xg_big)
valid_df['predictions'] = xg_big
valid_df.head()
## GredSearch for random forest
idk = GridSearchCV(RandomForestClassifier(), param_grid={'n_estimators': [10, 100, 1000]})
idk.fit(xTrain,yTrain)
pd.DataFrame(idk.cv_results_)

idk.score(xTest,yTest)

q = idk.predict(xTest)
calc_scores('boosted',yTest,q)

## Set up training data
target = valid_df['low_turnout']
valid_df.drop('low_turnout',axis=1,inplace=True)
X = valid_df
xTrain,xTest,yTrain,yTest = train_test_split(X,target,test_size=.25)

## Random Forest Model
rf = RandomForestClassifier(n_estimators=1000,max_depth=5,min_samples_split=.3,max_features=6)
rf.fit(xTrain,yTrain)
rf_y_test = rf.predict(xTest)
calc_scores(yTest,rf_y_test)
rf_y = rf.predict(valid_df)
confusion_matrix(target,rf_y)
calc_scores(target,rf_y)

## Run models function to calculate the scores from Random Forest, SVM, Decision Tree, and Logistic Regression
run_models(valid_df,valid_df,target)

################################################################
## Training the Classifier
# Identify the optimal tree depth for given data
depth = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for n in depth:
    dt = RandomForestClassifier(criterion='entropy',max_depth=n)
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
   dt = RandomForestClassifier(criterion='entropy', min_samples_split=min_samples_split)
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
   dt = RandomForestClassifier(criterion='entropy', min_samples_leaf=min_samples_leaf)
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
   dt = RandomForestClassifier(criterion='entropy', max_features=max_feature)
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
run_models(valid_df,valid_df,target)
run_models(valid_df,xTest,yTest)
###############################################################
def run_models(df,X,y):
    ## Set up training data
    xTrain,xTest,yTrain,yTest = train_test_split(X,y,test_size=.2)

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(xTrain,yTrain)
    lr_pred = lr.predict(xTest)
    # svm
    svm = LinearSVC(C=20, loss="hinge",random_state=10)
    svm.fit(xTrain,yTrain)
    svm_p = svm.predict(xTest)
    #Random Forest
    rf = RandomForestClassifier(n_estimators=1000,max_depth=5,min_samples_split=.3,max_features=6)
    rf.fit(xTrain,yTrain)
    rf_y = rf.predict(xTest)
    #xgboost
    xg = XGBClassifier(n_estimators=1000,max_depth=5,min_samples_split=.3,max_features=6)
    xg.fit(xTrain,yTrain)
    xg_y = xg.predict(xTest)
    # Calculate scores
    lr_s = calc_scores(yTest,lr_pred)
    svm_s = calc_scores(yTest,svm_p)
    rf_s = calc_scores(yTest,rf_y)
    xg_s = calc_scores(yTest,xg_y)
    scores = [lr_s,svm_s,rf_s,xg_s]
    for j in scores:
        print(f'Precision: {j[0]}, F1: {j[1]}, Accuracy: {j[2]}, Recall: {j[3]}, ROC_AUC: {j[4]}')



## Calculates all the scores and returns them in a list that includes the model name
## Needs more work
def calc_scores(yTest,yPred):
    p = precision_score(yTest,yPred)
    f1 = f1_score(yTest,yPred)
    a = accuracy_score(yTest,yPred)
    r = recall_score(yTest,yPred)
    roc = roc_auc_score(yTest,yPred)
    return [p,f1,a,r,roc]
