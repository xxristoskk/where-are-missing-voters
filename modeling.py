from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import f1_score, precision_score, classification_report,recall_score, accuracy_score, roc_curve, auc, roc_auc_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

## Set up training data
X = features
xTrain,xTest,yTrain,yTest = train_test_split(X,target,test_size=.35)

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

## Calculates all the scores of a model and prints them
def calc_scores(yTest,yPred):
    print('Precision: ', precision_score(yTest,yPred))
    print('F1: ', f1_score(yTest,yPred))
    print('Accuracy: ', accuracy_score(yTest,yPred))
    print('Recall: ', recall_score(yTest,yPred))
    falseP,trueP,thresholds = roc_curve(yTest,yPred)
    roc_auc = auc(falseP,trueP)
    print('ROC_AUC: ', roc_auc)
    print(confusion_matrix(yTest,yPred))

### Running the models
## Logistic regression model
lr = LogisticRegression()
lr.fit(xTrain,yTrain)
lr_y = lr.predict(xTest)
calc_scores(yTest,lr_y)

## Random Forest Model
# Grid search function for Random Forest model
def grid_search(xTrain,xTest,yTrain,yTest):
    gs = GridSearchCV(estimator=RandomForestClassifier(),
                     param_grid={'max_depth': [3,8],
                                 'n_estimators': (25,50,75,100,500,1000),
                                 'max_features': (4,6,8)},
                     cv=4,n_jobs=-1,scoring='balanced_accuracy')
    model = gs.fit(xTrain,yTrain)
    print(f'Best score: {model.best_score_}')
    print(f'Best parms: {model.best_params_}')
rf = RandomForestClassifier(n_estimators=1000,max_depth=5,min_samples_split=.5,max_features=8)
rf.fit(xTrain,yTrain)
rf_y_test = rf.predict(xTest)
calc_scores(yTest,rf_y_test)

## Plot important features for Random Forest
def plot_feature_importances(model,X_train):
    n_features = X_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns.values)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

plot_feature_importances(rf,xTrain)

# XGboot Model
# Grid search function for XGBoost model
def boost_search(xTrain,xTest,yTrain,yTest):
    gs = GridSearchCV(estimator=XGBClassifier(),
                     param_grid={'max_depth': [3,8],
                                 'n_estimators': (25,50,75,100,500,1000),
                                 'max_features': (4,6,8),
                                 'eta': (.1,.07,.05,.03,.01),
                                 'min_child_weight': (1,2),
                                 'sub_sample': (.3,.5,.7),
                                 'min_samples_split': (.4,.5,.6,.7)},
                     cv=4,n_jobs=-1,scoring='balanced_accuracy')
    model = gs.fit(xTrain,yTrain)
    print(f'Best score: {model.best_score_}')
    print(f'Best parms: {model.best_params_}')

boost_search(xTrain,xTest,yTrain,yTest)

## Preparing the final model
xg = XGBClassifier(silent=0,eta=.1,min_child_weight=2,sub_sample=.7,eval_metric='auc',n_estimators=1000,max_depth=5,min_samples_split=.7,max_features=9)
xg.fit(xTrain,yTrain)
xg_pred = xg.predict(xTest)

## Model summary
confusion_matrix(yTest,xg_pred)
calc_scores(yTest,xg_pred)

## Important features
plot_feature_importances(rf,xTrain)
