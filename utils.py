import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle

def save_file_as_pickle(path, data_object):
    pickle_out = open(path, "wb")
    pickle.dump(data_object, pickle_out)
    pickle_out.close()

def calculate_accuracy_per_class(df, labels_col='labels'):
    if 'new_labs' in df.columns:
        labels_col = 'new_labs'
    y_true = df[labels_col]
    labels = y_true.unique().tolist()
    accuracy_dict = {}
    for label in labels:
        true_labs = df[df[labels_col] == label][labels_col]
        pred_labs = df[df[labels_col] == label]['preds']
        lab_size = true_labs.shape[0]
        if lab_size == 0:
            accuracy_dict[label] = 0
        else:
            accuracy_dict[label] = sum(true_labs == pred_labs)/lab_size
    return accuracy_dict

def calculate_average_accuracy_per_budget(queried, n_samp, n_iter, train_set, test_set, minority_list, majority_list, n_trees):
        # calculate accuracy per original class and minority/majority classes
    mean_accuracy_per_budget = {}
    mean_accuracy_per_budget_original_classes = {}
    confusion_list_test = [0] * len(n_samp)
    classification_info_list_test = [0] * len(n_samp)
    for i in range(len(n_samp)):
        if i == 0:
            mean_accuracy_per_budget[i] = 0
            mean_accuracy_per_budget_original_classes[i] = 0
        else:
            total_budget_df = pd.DataFrame()
            total_budget_df_original_classes = pd.DataFrame()
            temp_df = pd.DataFrame()
            confusion_matrix_dict_test = {}
            classification_info_dict_test = {}
            for j in range(n_iter):
                trained_RF_mod = RandomForestClassifier(n_estimators=n_trees, class_weight='balanced')
                trained_RF_mod.fit(X=queried[i][j].drop(['labels', 'new_labs'], axis=1), y=queried[i][j]['new_labs'])
                preds = trained_RF_mod.predict(test_set.drop(['labels', 'new_labs'], axis=1))
                temp_df = pd.concat([test_set.reset_index(drop=True), pd.DataFrame(preds)], axis=1).rename(columns={0: 'preds'})
                temp = pd.DataFrame.from_dict(calculate_accuracy_per_class(temp_df), orient='index').rename(columns={0: j})
                total_budget_df = pd.concat([total_budget_df, temp], axis=1)

                # check accuracy per original class in the assigning to super class
                b = {}
                for k in test_set['labels'].tolist():
                    a = temp_df[temp_df['labels'] == k]
                    b[k] = (a['new_labs'] == a['preds']).sum() / a.shape[0]
                temp_b = pd.DataFrame.from_dict(b, orient='index').rename(columns={0: j})
                total_budget_df_original_classes = pd.concat([total_budget_df_original_classes, temp_b], axis=1)
                confusion_matrix_dict_test[j] = confusion_matrix(y_true=np.array(test_set['new_labs']), y_pred=np.array(preds))
                classification_info_dict_test[j] = classification_report(y_true=np.array(test_set['new_labs']), y_pred=np.array(preds), output_dict=True)
            classification_info_list_test[i] = classification_info_dict_test
            mean_accuracy_per_budget[i] = total_budget_df.apply(lambda x: np.mean(x), axis=1)
            mean_accuracy_per_budget_original_classes[i] = total_budget_df_original_classes.apply(lambda x: np.mean(x), axis=1)
            confusion_list_test[i] = confusion_matrix_dict_test
        print(i)

    accuracy_df = pd.DataFrame.from_dict(mean_accuracy_per_budget)
    original_classes_accuracy_df = pd.DataFrame.from_dict(mean_accuracy_per_budget_original_classes)

    props = train_set.groupby(by='labels').size()
    minority_classes = props.loc[minority_list]
    minority_props = minority_classes / minority_classes.sum()
    minority_df=original_classes_accuracy_df.loc[minority_list]

    majority_classes = props.loc[majority_list]
    majority_props = majority_classes / majority_classes.sum()
    majority_df=original_classes_accuracy_df.loc[majority_list]

    minority_average_list = []
    majority_average_list = []
    for i in range(len(n_samp)):
        minority_average_list.append((minority_df[i]*minority_props).sum())
        majority_average_list.append((majority_df[i] * majority_props).sum())


    accuracy_df_full = pd.concat([accuracy_df, pd.DataFrame(minority_average_list, columns=['minority']).T, \
                                pd.DataFrame(majority_average_list, columns=['majority']).T])
    
    original_classes_accuracy_df_full = pd.concat([original_classes_accuracy_df, pd.DataFrame(minority_average_list, \
                                        columns=['minority']).T, pd.DataFrame(majority_average_list, columns=['majority']).T])

    return accuracy_df_full, original_classes_accuracy_df_full

