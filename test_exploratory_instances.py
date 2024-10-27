import pickle
import pandas as pd

def save_file_as_pickle(path, data_object):
    pickle_out = open(path, "wb")
    pickle.dump(data_object, pickle_out)
    pickle_out.close()

def main():
    output_path = r'C:\Users\Noa Gruber\Desktop\Thesis Stuff\ThesisImplementations\results\scrambeled\digits\original_class_eval_37'
    curr_dat_path=f"{output_path}\\nclt5_ninit200_nconf02_scrambled01_exploration_instances.pickle"
    with open(curr_dat_path, 'rb') as f:
        exploratory_queried_idx = pickle.load(f)

    budget_dict = {}
    for budget in exploratory_queried_idx:
        budget_total_count = pd.DataFrame()
        for key in exploratory_queried_idx[budget]:
            curr_df = exploratory_queried_idx[budget][key]
            curr_count = pd.DataFrame(curr_df.groupby(by='labels').size()).reset_index().sort_values(by='labels')
            budget_total_count = pd.concat([budget_total_count, curr_count[0]], axis=1)
        budget_dict[budget] = budget_total_count.fillna(0).sum(axis=1)/10 #len(exploratory_queried_idx[budget].keys())
    save_file_as_pickle(output_path, budget_dict)

if __name__ == "main":
    main()