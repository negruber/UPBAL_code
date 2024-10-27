import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from utils import calculate_average_accuracy_per_budget, save_file_as_pickle

def main():
    base_path = "C:\\Users\\Noa Gruber\\Desktop\\Thesis Stuff\\ThesisImplementations\\results\\scrambeled\\"

    ######### Hand-written digits MNIST
    ######### 3,7 minority
    output_path = f'{base_path}digits\\original_class_eval_37'
    file_name = 'unbalanced_mnist'
    majority_list = [0,1,2,4,5,6,8,9]
    minority_list = [3,7]

    train_set = pd.read_csv(f'{output_path}\\{file_name}_train.csv')#.drop('Unnamed: 0', axis=1)
    test_set = pd.read_csv(f'{output_path}\\{file_name}_test.csv')#.drop('Unnamed: 0', axis=1)
    X = train_set.drop(['labels', 'new_labs'], axis=1)
    # X_labels = train_set['labels']
    X_labels = train_set['new_labs']
    N=train_set.shape[0]
    n_init=200
    n_clts = 5
    kmeans_clustering = KMeans(n_clusters=n_clts).fit(X)
    clt_labs_sc = kmeans_clustering.labels_

    batch=25
    n_samp = np.arange(0, 0.26, 0.01) * N
    pred_data_sc = pd.concat([X, X_labels, train_set['labels'], pd.DataFrame(np.ravel(clt_labs_sc, order='C'), columns=['cluster'])], axis=1)

    # Cluster based active learning
    queried = {}
    non_queried = {}
    total_queried_samps = {}
    n_iter=10
    n_clts = 2
    for k in range(len(n_samp)):
        quer_dats = {}
        queried_samps = {}
        for j in range(n_iter):
            quer_dat = pd.DataFrame()
            pools_full_data = pred_data_sc.copy()
            curr_quer_dat = pd.DataFrame()
            t=1
            for i in range(np.int(n_samp[k]/batch)):
                t=curr_quer_dat.shape[0] + 1
                pool_cluster = KMeans(n_clusters=n_clts).fit(pools_full_data.drop(['labels', 'cluster', 'new_labs'], axis=1))
                pool_cluster_labels = pool_cluster.labels_
                cluster_center = pd.DataFrame(pool_cluster.cluster_centers_)
                pools_full_data_dists = pd.DataFrame()
                for i in np.unique(pool_cluster_labels):
                    temp_df = (pools_full_data[pools_full_data['cluster'] == i]).drop(['cluster', 'labels', 'new_labs'], axis=1)
                    temp_pool_dists = (temp_df.apply(lambda x: x.to_numpy()-(cluster_center.loc[i]).to_numpy(), axis=1)).apply(lambda y: np.linalg.norm(y))
                    temp_df_dists = pd.concat([temp_df, temp_pool_dists], axis=1).rename(columns={0: 'dists'})
                    pools_full_data_dists = pd.concat([pools_full_data_dists, temp_df_dists])
                pools_full_data_dists['similarity'] = pools_full_data_dists['dists']
                similarity=pools_full_data_dists['similarity'].tolist()
                sorted_max_similarity = np.argpartition(similarity, kth=batch)
                n_highest_similarity_idx = sorted_max_similarity[0:batch]
                n_highest_similarity_df = pools_full_data.iloc[n_highest_similarity_idx]
                queried_samps[f"{j}_{t}"] = n_highest_similarity_df
                # currently queried data
                curr_quer_dat = pd.concat([curr_quer_dat, n_highest_similarity_df])
                curr_queried = n_highest_similarity_df.drop('cluster', axis=1)
                quer_dat = pd.concat([quer_dat, curr_queried])
                pools_full_data = pools_full_data.drop(n_highest_similarity_df.index).reset_index(drop=True)
            quer_dats[j]=quer_dat
        queried[k] = quer_dats
        total_queried_samps[k] = queried_samps
        non_queried[k] = pools_full_data
        print(k)

        save_file_as_pickle(path=f"{output_path}\\nclt{n_clts}_ninit{n_init}_batch{batch}_total_instances.pickle", data_object=queried)

        accuracy_df_full, original_classes_accuracy_df_full = calculate_average_accuracy_per_budget(queried, n_samp, n_iter, train_set, test_set, minority_list, majority_list, n_trees=100)

        with pd.ExcelWriter(f"{output_path}\\cluster_AL.xlsx") as writer:
            original_classes_accuracy_df_full.to_excel(writer, sheet_name='orig_class_accuracy')
            accuracy_df_full.to_excel(writer, sheet_name='01_accuracy')


if __name__ == "main":
    main()
