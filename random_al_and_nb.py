import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from utils import calculate_average_accuracy_per_budget, save_file_as_pickle, calculate_accuracy_per_class

def main():
    base_path = "C:\\Users\\Noa Gruber\\Desktop\\Thesis Stuff\\ThesisImplementations\\results\\scrambeled\\"

    ######### Hand-written digits MNIST
    ######### 3,7 minority
    output_path = f'{base_path}digits\\original_class_eval_37'
    file_name = 'unbalanced_mnist'
    majority_list = [0,1,2,4,5,6,8,9]
    minority_list = [3,7]

    train_set = pd.read_csv(f'{output_path}\\{file_name}_train.csv')
    test_set = pd.read_csv(f'{output_path}\\{file_name}_test.csv')
    X = train_set.drop(['labels', 'new_labs'], axis=1)
    X_labels = train_set['new_labs']
    N=train_set.shape[0]

    # initial model
    n_trees = 50
    RF_mod = RandomForestClassifier(n_estimators=n_trees)
    # create a pool of unlabeled instances without the initial data (without the closest obs)
    N=X.shape[0]
    n_samp = np.arange(0, 0.26, 0.01) * N

    #  Random Sampling
    full_pool_data = train_set.copy()
    queried_r = {}
    non_queried_r = {}
    budget = n_samp[1:26]
    n_iter = 10
    pools_datas_random = {}
    for i, k in enumerate(budget):
        trained_mod_r = RF_mod
        full_pool_data = train_set.copy()
        pools_data_random = {}
        queried_datas = {}
        for j in range(n_iter):
            full_pool_data = train_set.copy()
            trained_mod_r = RF_mod
            curr_chosen_idx = np.random.choice(list(full_pool_data.index), replace=False, size=np.int(k))
            curr_chosen_instances = full_pool_data.loc[curr_chosen_idx]
            curr_chosen_X = curr_chosen_instances.drop(['labels', 'new_labs'], axis=1)
            curr_chosen_labels = curr_chosen_instances['new_labs']
            trained_mod_r = trained_mod_r.fit(X=curr_chosen_X, y=np.ravel(curr_chosen_labels, order='C'))
            full_pool_data = full_pool_data.drop(curr_chosen_instances.index).reset_index(drop=True)
            queried_datas[j] = curr_chosen_instances
            non_queried_r[j] = full_pool_data
            pool_preds = trained_mod_r.predict(full_pool_data.drop(['labels', 'new_labs'], axis=1))
            pools_data_random[j] = pd.concat([full_pool_data, pd.DataFrame(pool_preds)], axis=1).rename(columns={0: "preds"})
        queried_r[i+1] = queried_datas
        pools_datas_random[i+1] = pools_data_random
        print(k)

    save_file_as_pickle(path=f"{output_path}\\randomAL_queried.pickle", data_object=queried_r)
    accuracy_df_full_r, original_classes_accuracy_df_full_r = calculate_average_accuracy_per_budget(queried_r, n_samp, n_iter, train_set, test_set, minority_list, majority_list, n_trees=100)

    with pd.ExcelWriter(f"{output_path}\\randomAL_scrambled01.xlsx") as writer:
        original_classes_accuracy_df_full_r.to_excel(writer, sheet_name='orig_class_accuracy')
        accuracy_df_full_r.to_excel(writer, sheet_name='01_accuracy')

    # Naive Bayes
    n_clts = 5
    kmeans_clustering = KMeans(n_clusters=n_clts).fit(X)
    # clt_cnts = kmeans_clustering.cluster_centers_
    clt_labs_sc = kmeans_clustering.labels_

    # pred_data_sc = pd.concat([X, X_labels, pd.DataFrame(np.ravel(clt_labs_sc, order='C'), columns=['cluster'])], axis=1)
    pred_data_sc = pd.concat([X, X_labels, train_set['labels'], pd.DataFrame(np.ravel(clt_labs_sc, order='C'), columns=['cluster'])], axis=1)
    N=X.shape[0]
    n_samp = np.arange(0, 0.26, 0.01) * N
    n_iter=10
    mnb = MultinomialNB()
    budget=n_samp[1:26]
    accuracy_NB = {}
    queried_NB = {}
    mean_accuracy_per_budget_original_classes_NB = {}
    accuracy_NB[0] = 0
    mean_accuracy_per_budget_original_classes_NB[0] = 0
    budget_dict = {}
    for i, k in enumerate(budget):
        total_budget_df = pd.DataFrame()
        total_budget_df_original_classes = pd.DataFrame()
        temp_df = pd.DataFrame()
        experiment_dict = {}
        queried_datas = {}
        for j in range(n_iter):
            curr_data = pred_data_sc.groupby(by="cluster", group_keys=False).apply(lambda x: x.sample(n=np.int(k/n_clts)))
            queried_datas[j] = curr_data
            x_train = curr_data.drop(['labels', 'cluster', 'new_labs'], axis=1)
            x_labels = curr_data['new_labs']
            y_pred = mnb.fit(x_train, x_labels).predict(test_set.drop(['labels', 'new_labs'], axis=1))
            temp_df = pd.concat([test_set.reset_index(drop=True), pd.DataFrame(y_pred)], axis=1).rename(columns={0: 'preds'})
            temp = pd.DataFrame.from_dict(calculate_accuracy_per_class(temp_df), orient='index').rename(columns={0: j})
            total_budget_df = pd.concat([total_budget_df, temp], axis=1)
            # check accuracy per original class in the assigning to super class
            b = {}
            for l in test_set['labels'].tolist():
                a = temp_df[temp_df['labels'] == l]
                b[l] = (a['new_labs'] == a['preds']).sum() / a.shape[0]
            experiment_dict[f'experiment_{j}'] = b
            temp_b = pd.DataFrame.from_dict(b, orient='index').rename(columns={0: j})
            total_budget_df_original_classes = pd.concat([total_budget_df_original_classes, temp_b], axis=1)
        accuracy_NB[i + 1] = total_budget_df.apply(lambda x: np.mean(x), axis=1)
        mean_accuracy_per_budget_original_classes_NB[i + 1] = total_budget_df_original_classes.apply(lambda x: np.mean(x), axis=1)
        queried_NB[i + 1] = queried_datas
        budget_dict[f'budget_size_{k}'] = experiment_dict
        print(k)

    save_file_as_pickle(path=f"{output_path}\\budget_dict_NB.pickle", data_object=budget_dict)
    save_file_as_pickle(path=f"{output_path}\\NB_queried.pickle", data_object=queried_NB)

    accuracy_df_full_NB, original_classes_accuracy_df_full_NB = calculate_average_accuracy_per_budget(queried_r, n_samp, n_iter, train_set, test_set, minority_list, majority_list, n_trees=100)

    with pd.ExcelWriter(f"{output_path}\\NB_scrambled01.xlsx") as writer:
        original_classes_accuracy_df_full_NB.to_excel(writer, sheet_name='orig_class_accuracy')
        accuracy_df_full_NB.to_excel(writer, sheet_name='01_accuracy')

if __name__ == "main":
    main()