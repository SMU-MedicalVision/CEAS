import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.svm import SVC
use_CNN = True
centers = ['TAHSYSU' ,'TAHSMURET_test','NHH', 'ZJH',]
# centers = ['TAHSMUPRO_nonrefined']
refine = False
refine_center = 'TAHSMUPRO_refined'

def metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob > threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    return accuracy, auc, sensitivity, specificity, precision, f1

# load data
train_df = pd.read_excel(rf"./imputation_TAHSMURET_train_2024_02_22.xlsx", sheet_name='Sheet1')
X_train = train_df[['Gender', 'Age', 'HLA-B27'] if not use_CNN else ['Gender', 'Age', 'HLA-B27', 'CNN']]
Y_train = train_df['label(axSpA:1;nonaxSpA:0)']
if refine:
    refine_df = pd.read_excel(rf"./test_TAHSMUPRO_refined_2024_02_22.xlsx", sheet_name='refined')
    X_refine = refine_df[['Gender', 'Age', 'HLA-B27'] if not use_CNN else ['Gender', 'Age', 'HLA-B27', 'CNN']]
    X_refine = pd.concat([X_train, X_refine], axis=0)
    Y_refine = refine_df['label(axSpA:1;nonaxSpA:0)']
    Y_refine = pd.concat([Y_train, Y_refine], axis=0)

# ML models
classifiers = {'Logistic Regression': LogisticRegression(),
               'Decision Tree': DecisionTreeClassifier(criterion='gini',
                                                       min_samples_leaf=10,
                                                       min_samples_split=20,
                                                       max_leaf_nodes=None,
                                                       max_depth=10),
               'Naive Bayes': GaussianNB(),
               'Random Forest': RandomForestClassifier(criterion="gini",
                                                       min_samples_leaf=10,
                                                       min_samples_split=20,
                                                       max_leaf_nodes=None,
                                                       max_depth=10),
               'KNN-3': KNeighborsClassifier(n_neighbors=3),
               'KNN-5': KNeighborsClassifier(n_neighbors=5),
               # 'KNN-7': KNeighborsClassifier(n_neighbors=7),
               # 'KNN-9': KNeighborsClassifier(n_neighbors=9),
               # 'KNN-11': KNeighborsClassifier(n_neighbors=11),
               # 'KNN-13': KNeighborsClassifier(n_neighbors=13),
               # 'KNN-15': KNeighborsClassifier(n_neighbors=15),
               'SVM': SVC(probability=True)}

result_dict = {'Center': [], 'Model': [], 'Accuracy': [], 'AUC': [], 'Sensitivity': [], 'Specificity': [],
               'Precision': [], 'F1': []}

for center in centers:
    test_df = pd.read_excel(rf"./imputation_{center}_2024_02_22.xlsx", sheet_name='Sheet1')
    X_test = test_df[['Gender', 'Age', 'HLA-B27'] if not use_CNN else ['Gender', 'Age', 'HLA-B27', 'CNN']]
    Y_test = test_df['label(axSpA:1;nonaxSpA:0)']
    if refine:
        result_path = rf"./result_ML{'_CNN' if use_CNN else ''}_refined.xlsx"
        patient_wise_result_path = rf"./patient_wise_result_ML_{refine_center}{'_CNN' if use_CNN else ''}.xlsx"
    else:
        if len(centers) == 1 and centers[0] == 'TAHSMUPRO_nonrefined':
            result_path = rf"./result_ML{'_CNN' if use_CNN else ''}_nonrefine.xlsx"
            patient_wise_result_path = rf"./patient_wise_result_ML_{center}{'_CNN' if use_CNN else ''}.xlsx"
        else:
            result_path = rf"./result_ML{'_CNN' if use_CNN else ''}.xlsx"
            patient_wise_result_path = rf"./patient_wise_result_ML_{center}{'_CNN' if use_CNN else ''}.xlsx"
    patient_wise_result_dict = {'patient': [], 'label': []}
    patient_wise_result_dict.update(zip(classifiers.keys(), [[] for _ in range(len(classifiers))]))
    patient_wise_result_dict['patient'].extend(test_df['Number'])
    patient_wise_result_dict['label'].extend(Y_test)
    for clf_name in classifiers:
        clf = classifiers[clf_name]
        clf.fit(X_train, Y_train)
        if refine:
            clf.fit(X_refine, Y_refine)
        Y_pred = clf.predict(X_test)
        Y_pred_proba = clf.predict_proba(X_test)[:, 1]
        accuracy, auc, sensitivity, specificity, precision, f1 = metrics(Y_test, Y_pred_proba)
        print(f"{center} center, {clf_name} model:")
        print(
            f"Accuracy: {accuracy:.3f} AUC: {auc:.3f} Sensitivity: {sensitivity:.3f} Specificity: {specificity:.3f} Precision: {precision:.3f} F1: {f1:.3f}")
        result_dict['Center'].append(center)
        result_dict['Model'].append(clf_name)
        result_dict['Accuracy'].append(accuracy)
        result_dict['AUC'].append(auc)
        result_dict['Sensitivity'].append(sensitivity)
        result_dict['Specificity'].append(specificity)
        result_dict['Precision'].append(precision)
        result_dict['F1'].append(f1)

        patient_wise_result_dict[clf_name].extend(Y_pred_proba)

    patient_wise_result_df = pd.DataFrame(patient_wise_result_dict)
    patient_wise_result_df.to_excel(patient_wise_result_path, index=False)

result_df = pd.DataFrame(result_dict)
result_df.to_excel(result_path, index=False)
print(f"Results saved to {result_path}")