import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from scipy.stats import entropy
from utils import save_file_as_pickle, calculate_average_accuracy_per_budget

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

    n_clts = 5
    batch = 25
    n_init = 200
    
    kmeans_clustering = KMeans(n_clusters=n_clts).fit(X)
    clt_labs_sc = kmeans_clustering.labels_

    # the whole data with x1|x2|labels|cluster
    pred_data_sc = pd.concat([X, X_labels, train_set['labels'], pd.DataFrame(np.ravel(clt_labs_sc, order='C'), columns=['cluster'])], axis=1)

    majority_idx = pred_data_sc["labels"].isin(majority_list)
    majority_pred_data_sc = pred_data_sc[majority_idx]
    temp_init_full_data_sc = majority_pred_data_sc.groupby(by="cluster", group_keys=False).apply(lambda x: x.sample(n=int(n_init/n_clts)))

    init_labeled_data_sc = temp_init_full_data_sc.drop(['labels', 'cluster', 'new_labs'], axis=1)
    init_cluster_labels = temp_init_full_data_sc['cluster']

    # initial model
    n_trees = 50
    RF_mod = RandomForestClassifier(n_estimators=n_trees)
    init_mod_sc = RF_mod.fit(X=init_labeled_data_sc, y=np.ravel(init_cluster_labels, order='C'))
    N=X.shape[0]
    n_samp = np.arange(0, 0.26, 0.01) * N

    queried = {}
    non_queried = {}
    total_queried_samps = {}
    n_iter=10
    for k in range(len(n_samp)):
        quer_dats = {}
        queried_samps = {}
        for j in range(n_iter):
            quer_dat = pd.DataFrame()
            pools_full_data = pred_data_sc.copy()
            trained_RF_mod=init_mod_sc
            curr_quer_dat = pd.DataFrame()
            t=1
            for i in range(np.int(n_samp[k]/batch)):
                t=curr_quer_dat.shape[0] + 1
                prob_mat = trained_RF_mod.predict_proba(pools_full_data.drop(['labels', 'cluster', 'new_labs'], axis=1))
                entropies_list = (pd.DataFrame(prob_mat).apply(entropy, axis=1)).tolist()
                entropies_list = (pd.DataFrame(prob_mat).apply(entropy, axis=1)).tolist()
                sorted_entropies = np.argpartition(entropies_list, kth=-batch)
                n_highest_entropies_idx = sorted_entropies[-batch:]
                n_highest_entropies_df = pools_full_data.iloc[n_highest_entropies_idx]
                queried_samps[f"{j}_{t}"] = n_highest_entropies_df
                # currently queried data
                curr_quer_dat = pd.concat([curr_quer_dat, n_highest_entropies_df])
                # the clusters in the chosen instances so far (should also contain instances from previous batches of this budget)
                temp_df = pd.DataFrame()
                curr_queried = n_highest_entropies_df.drop('cluster', axis=1)
                quer_dat = pd.concat([quer_dat, curr_queried])
                curr_train_set = quer_dat
                trained_RF_mod = trained_RF_mod.fit(X=curr_train_set.drop(columns=['labels', 'new_labs']), y=curr_train_set['new_labs'])
                pools_full_data = pools_full_data.drop(n_highest_entropies_df.index).reset_index(drop=True)
            quer_dats[j]=quer_dat
        queried[k] = quer_dats
        total_queried_samps[k] = queried_samps
        non_queried[k] = pools_full_data
        print(k)

        # Save queried instances
        save_file_as_pickle(path=f"{output_path}\\nclt5_ninit200_nconf02_queried_scrambled01_no_explore.pickle", data_object=queried)
        
        accuracy_df_full, original_classes_accuracy_df_full = calculate_average_accuracy_per_budget(queried, n_samp, n_iter, train_set, test_set, minority_list, majority_list, n_trees=100)

        with pd.ExcelWriter(f"{output_path}\\nclt5_ninit200_nconf02_scrambled01_no_explore.xlsx") as writer:
            original_classes_accuracy_df_full.to_excel(writer, sheet_name='orig_class_accuracy')
            accuracy_df_full.to_excel(writer, sheet_name='01_accuracy')

if __name__ == "main":
    main()