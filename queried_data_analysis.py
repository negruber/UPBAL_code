import pandas as pd
import pickle
from utils import save_file_as_pickle, calculate_accuracy_per_class

def main():
    ######### 3,7 minority
    output_path = r'C:\Users\Noa Gruber\Desktop\Thesis Stuff\ThesisImplementations\results\scrambeled\digits\original_class_eval_37'
    file_name = 'unbalanced_mnist'
    majority_list = [0,1,2,4,5,6,8,9]
    minority_list = [3,7]

    queried_data_paths = [f"{output_path}\\nclt5_ninit200_nconf02_queried_scrambled01_dense_explore.pickle",
                          f"{output_path}\\corrected_densityAL\\densityAL_queried_instances.pickle",
                          f"{output_path}\\nclt5_ninit200_nconf02_queried_scrambled01_no_explor.pickle",
                          f"{output_path}\\cluster_AL_queried_instances.pickle",
                          f"{output_path}\\randomAL_queried.pickle"]
    algorithms_list = ['proposed_algorithm', 'densityAL', 'no_explor', 'clusterAL', 'randomAL']

    algo_dict = {}
    for i, path in enumerate(queried_data_paths):
        with open(path, 'rb') as queried_file:
            queried_data = pickle.load(queried_file)

        # first key- budget size, second key - number of repeated experiment
        writer = pd.ExcelWriter(f"{output_path}\\queried_analysis\\{algorithms_list[i]}_new.xlsx")
        means_dict = {}
        for budget in queried_data:
            if budget > 0:
                curr_dat = queried_data[budget][0]
                for experiment in queried_data[budget]:
                    if experiment == 0:
                        merged_dat = pd.DataFrame(curr_dat.groupby(by=['labels', 'new_labs']).size().reset_index()).rename(columns={0:f'instance count experiment {experiment}'})
                    elif experiment > 0:
                        curr_data = queried_data[budget][experiment]
                        curr_count = pd.DataFrame(curr_data.groupby(by=['labels']).size().reset_index()).rename(columns={0:f'instance count experiment {experiment}'})
                        merged_dat = pd.merge(merged_dat, curr_count, on='labels', how='outer')

                merged_dat = merged_dat.fillna(0)
                count_cols = [x for x in merged_dat.columns if 'instance' in x]
                merged_dat['mean'] = merged_dat[count_cols].mean(axis=1)
                means_dict[budget] = merged_dat[['labels', 'mean']]
                merged_dat.to_excel(excel_writer=writer, sheet_name=f'queried_{budget}', index=False)
        algo_dict[algorithms_list[i]] = means_dict
        writer.save()
    
if __name__ == "main":
    main()
