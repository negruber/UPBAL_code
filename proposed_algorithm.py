import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from scipy.stats import entropy
import pickle
from scipy.spatial.distance import pdist, squareform
from utils import calculate_average_accuracy_per_budget, save_file_as_pickle

# data is a slice of the full dataframe
def get_majority_class(class_labels):
    unique_class_labels = np.unique(class_labels, return_counts=True)
    majority_class_idx = pd.DataFrame(unique_class_labels[1]).idxmax()[0]
    majority_class = unique_class_labels[0][majority_class_idx]
    return majority_class


def sample(df, n_sample, key="pred_class"):
    class_size_df = df.groupby(by=key).size()
    if (class_size_df < n_sample).any():
        temp_list = []
        for class_name in class_size_df.index:
            # print(class_name)
            k_samp = min(n_sample, class_size_df[class_name])
            temp_df = df[df[key] == class_name].sample(n=k_samp)
            # print(df[df[key] == class_name])
            temp_list.append(temp_df)
        final_df = pd.concat(temp_list)
    else:
        final_df = df.groupby(by=key, group_keys=False).apply(lambda x: x.sample(n=n_sample))
    return final_df


def get_average_f1_per_class(dict_list, n_labs):
    average_f1_list = []
    for budget_dict in dict_list:
        # print(budget_dict[0])
        f1_budget_dict = {}
        mean_f1_scores = []
        for iteration_dict_key in budget_dict:
            class_names = list(budget_dict[iteration_dict_key].keys())[0:n_labs]
            curr_iteration_f1_scores = []
            for class_name in class_names:
                curr_iteration_f1_scores.append(budget_dict[iteration_dict_key][class_name]["f1-score"])
            f1_budget_dict[iteration_dict_key] = curr_iteration_f1_scores
        average_f1_per_budget = pd.DataFrame(f1_budget_dict).apply(np.mean, axis=1)
        average_f1_list.append(average_f1_per_budget)
    return average_f1_list


def get_weight(t, w):
    if t:
        weight=w
    else:
        weight=0
    return weight

# Stratified sampling functions
def calc_stratum_var(df, cluster):
    strt_df = df[df['cluster'] == cluster]
    if 'new_labs' in strt_df.columns:
        strt_df = strt_df.drop(['labels', 'cluster', 'new_labs'], axis=1)
    else:
        strt_df = strt_df.drop(['labels', 'cluster'], axis=1)
    y_h = strt_df.mean()
    S_h = ((strt_df-y_h)**2).sum(axis=1)/(strt_df.shape[0]-1)
    return S_h


def calc_s_w_h(df, cluster):
    w_h = df[df['cluster'] == cluster].shape[0]/df.shape[0]
    S_h = calc_stratum_var(df, cluster)
    return w_h*S_h


def calc_n_h(df, cluster, cluster_lab_list):
    temp_dict = {}
    for lab in cluster_lab_list:
        temp_dict[lab]=np.linalg.norm(calc_s_w_h(df, lab))
    s_w_df = pd.DataFrame.from_dict(temp_dict, orient='index')
    d = s_w_df.sum()
    n_h = np.linalg.norm(calc_s_w_h(df, cluster))/d
    return n_h.values[0]

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
    n_labs = X_labels.nunique()

    train_set.shape[0]+test_set.shape[0]

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
    # create a pool of unlabeled instances without the initial data (without the closest obs)

    N=X.shape[0]
    n_samp = np.arange(0, 0.26, 0.01) * N
    ####

    distances = pdist(X.values, metric='euclidean')
    dist_matrix = squareform(distances)

    queried = {}
    non_queried = {}
    total_exploration_samps = {}
    total_lowest_entropies_dict = {}
    AL_queried_samps = {}

    n_iter=10
    count=0
    alpha = 1

    for k in range(len(n_samp)):
        quer_dats = {}
        exploration_samps = {}
        lowest_entropies_dict = {}
        queried_samps = {}
        for j in range(n_iter):
            curr_dist_df = pd.DataFrame(dist_matrix.copy())
            quer_dat = pd.DataFrame()
            pools_full_data = pred_data_sc.copy()
            trained_RF_mod=init_mod_sc
            curr_quer_dat = pd.DataFrame()
            t=1
            for i in range(int(n_samp[k]/batch)):
                t=curr_quer_dat.shape[0] + 1
                prob_mat = trained_RF_mod.predict_proba(pools_full_data.drop(['labels', 'cluster', 'new_labs'], axis=1))
                eps = np.random.rand()
                n=quer_dat.shape[0]+1
                if eps < 1/(np.log(t) + 1) and curr_quer_dat.shape[0] > 0:
                    count+=1
                    entropies_df = (pd.DataFrame(prob_mat).apply(entropy, axis=1))
                    entropies_list = entropies_df.tolist()
                    id_x = entropies_df * (((1 / pools_full_data.shape[0]) * ((1 / (curr_dist_df + 1)).sum(axis=1))) ** alpha)
                    sorted_max_similarity = np.argpartition(id_x, kth=batch)
                    n_lowest_similarity_idx = sorted_max_similarity[0:batch]
                    n_lowest_similarity_df = pools_full_data.iloc[n_lowest_similarity_idx]

                    n_highest_entropies_idx = n_lowest_similarity_df.index.tolist()
                    lowest_entropies_dict[f"{j}_{t}"] = n_lowest_similarity_idx
                    # These are the instances queried in the exploratory step
                    curr_similarity_queried = pools_full_data.loc[n_highest_entropies_idx]
                    exploration_samps[f"{j}_{t}"] = curr_similarity_queried
                else:
                    # calculate estimated probabilities
                    entropies_df = pd.DataFrame(prob_mat).apply(entropy, axis=1)
                    entropies_list = (pd.DataFrame(prob_mat).apply(entropy, axis=1)).tolist()

                    sorted_max_similarity = np.argpartition(entropies_list, kth=-batch)
                    n_highest_similarity_idx = sorted_max_similarity[-batch:]
                    # These are the instances which are queried in the AL step
                    curr_similarity_queried = pools_full_data.iloc[n_highest_similarity_idx]
                    queried_samps[f"{j}_{t}"] = curr_similarity_queried
                # currently queried data
                curr_quer_dat = pd.concat([curr_quer_dat, curr_similarity_queried])
                # the clusters in the chosen instances so far (should also contain instances from previous batches of this budget)
                temp_df = pd.DataFrame()
                curr_queried = curr_similarity_queried.drop('cluster', axis=1)
                quer_dat = pd.concat([quer_dat, curr_queried])
                curr_train_set = quer_dat
                trained_RF_mod = trained_RF_mod.fit(X=curr_train_set.drop(columns=['labels', 'new_labs']), y=curr_train_set['new_labs'])
                pools_full_data = pools_full_data.drop(curr_similarity_queried.index).reset_index(drop=True)
                curr_dist_df = curr_dist_df.drop(curr_similarity_queried.index, axis=1)
                curr_dist_df = curr_dist_df.drop(curr_similarity_queried.index).reset_index(drop=True)
                curr_dist_df.columns = range(curr_dist_df.shape[0])
            quer_dats[j]=quer_dat
        queried[k] = quer_dats
        total_exploration_samps[k] = exploration_samps
        total_lowest_entropies_dict[k] = lowest_entropies_dict
        AL_queried_samps[k] = queried_samps
        non_queried[k] = pools_full_data

    minority_str = "".join([str(x) for x in minority_list])
    results_output_path = f"C:\\Users\\Noa Gruber\\Desktop\\Thesis Stuff\\Paper\\algorithm_refactoring_results\\digits\\minority_{minority_str}"

    save_file_as_pickle(path=f"{results_output_path}\\nclt{n_clts}_ninit{n_init}_batch{batch}_exploration_instances.pickle", data_object=total_exploration_samps)
    save_file_as_pickle(path=f"{results_output_path}\\nclt{n_clts}_ninit{n_init}_batch{batch}_AL_instances.pickle", data_object=AL_queried_samps)
    save_file_as_pickle(path=f"{results_output_path}\\nclt{n_clts}_ninit{n_init}_batch{batch}_total_instances.pickle", data_object=queried)

    accuracy_df_full, original_classes_accuracy_df_full = calculate_average_accuracy_per_budget(queried, n_samp, n_iter, train_set, test_set, minority_list, majority_list, n_trees=100)

    with pd.ExcelWriter(f"{results_output_path}\\nclt{n_clts}_ninit{n_init}_batch{batch}_algorithm_accuracy.xlsx") as writer:
        original_classes_accuracy_df_full.to_excel(writer, sheet_name='orig_class_accuracy')
        accuracy_df_full.to_excel(writer, sheet_name='01_accuracy')


if __name__ == "main":
    main()

        