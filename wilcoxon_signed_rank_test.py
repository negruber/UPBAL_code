import pandas as pd
import numpy as np
import pickle
import scipy.stats

def main():
    output_path = r'C:\Users\Noa Gruber\Desktop\Thesis Stuff\ThesisImplementations\results\scrambeled\digits\original_class_eval_368'
    minority_list = [3,6,8]
    majority_list = [0,1,2,4,5,7,9]
    algorithm_file_names = [f"{output_path}\\total_queried_with_preds_and_mcnemar_val_density.pickle",
                            f"{output_path}\\total_queried_with_preds_and_mcnemar_val_no_explore.pickle",
                            f"{output_path}\\total_queried_with_preds_and_mcnemar_val_cluster.pickle",
                            f"{output_path}\\total_queried_with_preds_and_mcnemar_val_random.pickle",
                            f"{output_path}\\total_queried_with_preds_and_mcnemar_val_NB.pickle"]
    algorithm_names = ['density', 'no_explore', 'cluster', 'random', 'NB']
    total_algorithm_tst = {}
    minority_algorithm_tst = {}
    majority_algorithm_tst = {}
    for i, algorithm_file_name in enumerate(algorithm_file_names):
        with open(algorithm_file_name, 'rb') as queried_file:
            queried = pickle.load(queried_file)

        total_tst_dict = {}
        majority_tst_dict = {}
        minority_tst_dict = {}
        for budget in queried:
            test_df = queried[budget].reset_index()
            test_df["proposed_success"] = test_df.apply(lambda x: 1 if x['new_labs'] == x['preds'] else 0, axis=1)
            test_df[f"{algorithm_names[i]}_success"] = test_df.apply(lambda x: 1 if x['new_labs'] == x[f'{algorithm_names[i]}_preds'] else 0, axis=1)
            successes_df = test_df[["index", "labels", "new_labs","proposed_success", f"{algorithm_names[i]}_success"]].groupby(by=["index", "labels", "new_labs"]).sum().reset_index()
            successes_df["success_diff"] = successes_df.apply(lambda x: x["proposed_success"] - x[f"{algorithm_names[i]}_success"], axis=1)
            successes_df["abs_success_diff"] = successes_df.apply(lambda x: abs(x["proposed_success"] - x[f"{algorithm_names[i]}_success"]), axis=1)
            
            # for total data calculations
            total_successes_df = successes_df.copy()
            total_successes_df_non_zero = total_successes_df[total_successes_df["abs_success_diff"] != 0].copy()
            total_successes_df_sorted = total_successes_df_non_zero.sort_values(by="abs_success_diff")
            total_successes_df_sorted["rank"] = total_successes_df_sorted["abs_success_diff"].rank()
            total_successes_df_sorted["rank_sign"] = total_successes_df_sorted.apply(lambda x: np.sign(x["success_diff"]), axis=1)
            total_successes_df_sorted["signed_rank"] = total_successes_df_sorted.apply(lambda x: x["rank"]*x["rank_sign"], axis=1)
            total_w_minus = total_successes_df_sorted[total_successes_df_sorted["rank_sign"] < 0]["signed_rank"].sum()
            total_w_plus = total_successes_df_sorted[total_successes_df_sorted["rank_sign"] > 0]["signed_rank"].sum()
            total_n = total_successes_df_sorted[total_successes_df_sorted["rank_sign"] != 0].shape[0]
            total_mu = 0
            total_sigma = total_n*(total_n+1)*(2*total_n+1)/6
            
            # to calculate t
            total_t_vec = total_successes_df_sorted.groupby(by="rank").size().reset_index()[0]
            total_t = ((total_t_vec ** 3 - total_t_vec) / 48).sum()
            total_test_stat = ((total_w_plus + total_w_minus) - total_mu)/np.sqrt(total_sigma - total_t)
            
            # for minority
            minority_successes_df = successes_df[successes_df["labels"].isin(minority_list)].copy()
            minority_successes_df_non_zero = minority_successes_df[minority_successes_df["abs_success_diff"] != 0].copy()
            minority_successes_df_sorted = minority_successes_df_non_zero.sort_values(by="abs_success_diff")
            minority_successes_df_sorted["rank"] = minority_successes_df_sorted["abs_success_diff"].rank()
            minority_successes_df_sorted["rank_sign"] = minority_successes_df_sorted.apply(lambda x: np.sign(x["success_diff"]), axis=1)
            minority_successes_df_sorted["signed_rank"] = minority_successes_df_sorted.apply(lambda x: x["rank"] * x["rank_sign"], axis=1)
            minority_w_minus = minority_successes_df_sorted[minority_successes_df_sorted["rank_sign"] < 0]["signed_rank"].sum()
            minority_w_plus = minority_successes_df_sorted[minority_successes_df_sorted["rank_sign"] > 0]["signed_rank"].sum()
            minority_n = minority_successes_df_sorted[minority_successes_df_sorted["rank_sign"] != 0].shape[0]
            minority_mu = 0
            minority_sigma = minority_n * (minority_n + 1) * (2 * minority_n + 1) / 6
            
            # to calculate t
            minority_t_vec = minority_successes_df_sorted.groupby(by="rank").size().reset_index()[0]
            minority_t = ((minority_t_vec ** 3 - minority_t_vec) / 48).sum()
            minority_test_stat = ((minority_w_plus + minority_w_minus) - minority_mu) / np.sqrt(minority_sigma - minority_t)
            
            # for majority
            majority_successes_df = successes_df[successes_df["labels"].isin(majority_list)].copy()
            majority_successes_df_non_zero = majority_successes_df[majority_successes_df["abs_success_diff"] != 0].copy()
            majority_successes_df_sorted = majority_successes_df_non_zero.sort_values(by="abs_success_diff")
            majority_successes_df_sorted["rank"] = majority_successes_df_sorted["abs_success_diff"].rank()
            majority_successes_df_sorted["rank_sign"] = majority_successes_df_sorted.apply(lambda x: np.sign(x["success_diff"]), axis=1)
            majority_successes_df_sorted["signed_rank"] = majority_successes_df_sorted.apply(lambda x: x["rank"] * x["rank_sign"], axis=1)
            majority_w_minus = majority_successes_df_sorted[majority_successes_df_sorted["rank_sign"] < 0]["signed_rank"].sum()
            majority_w_plus = majority_successes_df_sorted[majority_successes_df_sorted["rank_sign"] > 0]["signed_rank"].sum()
            majority_n = majority_successes_df_sorted[majority_successes_df_sorted["rank_sign"] != 0].shape[0]
            majority_mu = 0
            majority_sigma = majority_n * (majority_n + 1) * (2 * majority_n + 1) / 6
            
            # to calculate t
            majority_t_vec = majority_successes_df_sorted.groupby(by="rank").size().reset_index()[0]
            majority_t = ((majority_t_vec ** 3 - majority_t_vec) / 48).sum()
            majority_test_stat = ((majority_w_plus + majority_w_minus) - majority_mu) / np.sqrt(majority_sigma - majority_t)
            
            total_tst_dict[budget] = total_test_stat
            minority_tst_dict[budget] = minority_test_stat
            majority_tst_dict[budget] = majority_test_stat
        total_algorithm_tst[algorithm_names[i]] = total_tst_dict
        minority_algorithm_tst[algorithm_names[i]] = minority_tst_dict
        majority_algorithm_tst[algorithm_names[i]] = majority_tst_dict
        print(algorithm_names[i])

    dict_list = [total_algorithm_tst, minority_algorithm_tst, majority_algorithm_tst]
    sheet_names = ["total", "minority", "majority"]
    with pd.ExcelWriter(f"{output_path}\\wilcoxon_test_statistics.xlsx") as writer:
        for i, dict in enumerate(dict_list):
            curr_tst_df = pd.DataFrame.from_dict(dict,orient='index')
            curr_tst_df.to_excel(writer, sheet_name=f"{sheet_names[i]}_test_stats")
            curr_pval_df = curr_tst_df.applymap(lambda x: np.round((1-scipy.stats.norm.cdf(x)),3))
            curr_pval_df.to_excel(writer, sheet_name=f"{sheet_names[i]}_pval")


if __name__ == "main":
    main()