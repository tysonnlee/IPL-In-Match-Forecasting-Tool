import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import joblib
from scipy.stats import mode

df = pd.read_excel('#Enter path to dataset here')
print(df['chased_successfully'].value_counts(normalize=True))


#Isolate the 2025 IPL season for testing
df_test = df[df['year'] == 2025]
columns_to_drop = ['match_id', 'date', 'year']
X_test = df_test.drop(columns='chased_successfully')
X_test = X_test.drop(columns=columns_to_drop, errors='ignore')
y_test = df_test['chased_successfully']

#Create the dataframe for model training
df_main = df[df['year'] != 2025]
X = df_main.drop(columns='chased_successfully')
X = X.drop(columns=columns_to_drop, errors='ignore')
y = df_main['chased_successfully']


#Create a baseline model using log loss
p_class1 = np.mean(y)
p_class0 = 1 - p_class1
y_proba_baseline = np.tile([p_class0, p_class1], (len(y), 1))
baseline_log_loss = log_loss(y, y_proba_baseline)
print(f"Baseline log loss: {baseline_log_loss:.4f}")

mode_result = mode(y, keepdims=True)
mode_class = mode_result.mode[0]
y_pred_baseline = np.full_like(y, fill_value=mode_class)

#Assess accuracy and F1 score of a baseline model
baseline_accuracy = accuracy_score(y, y_pred_baseline)
baseline_f1 = f1_score(y, y_pred_baseline)
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
print(f"Baseline F1 Score: {baseline_f1:.4f}")

#Run a gridsearch cross-validation on the training dataset
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=33)
hyperparams_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 5, 10]
}

scoring_metrics = {
    'log_loss': 'neg_log_loss',
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score)
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=33),
    param_grid=hyperparams_grid,
    cv=cv,
    scoring=scoring_metrics,
    refit='log_loss',
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(X, y)

#Extract CV results
results_df = pd.DataFrame(grid_search.cv_results_)

#Flip sign for log loss since sklearn stores it as negative
results_df['mean_train_log_loss'] = -results_df['mean_train_log_loss']
results_df['mean_test_log_loss'] = -results_df['mean_test_log_loss']

results_df = results_df[[
    'param_n_estimators', 
    'param_max_depth', 
    'param_min_samples_leaf',
    'mean_train_log_loss', 'mean_test_log_loss',
    'mean_train_accuracy', 'mean_test_accuracy',
    'mean_train_f1', 'mean_test_f1'
]]

results_df.to_excel('#Enter path to output results for analysis')

rf_best_model = grid_search.fit(X, y)
print("Best hyperparameters:", grid_search.best_params_)

#Assess accuracy and F1 on testing dataset
y_test_proba = rf_best_model.predict_proba(X_test)
test_log_loss = log_loss(y_test, y_test_proba)
print(f"Test Log Loss: {test_log_loss:.4f}")

y_test_pred = rf_best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")


#Assess feature importance
importances = rf_best_model.best_estimator_.feature_importances_
feature_names = X.columns

feat_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feat_importances)

#Create ROC Curve
y_test_probs = rf_best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_probs)

#Calculate Area Under Curve (AUC)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



joblib.dump(rf_best_model, '#Enter path to output model/rf_model.pkl')
