import pandas as pd
import matplotlib.pyplot as plt

def main():
    output_path = r'C:\Users\Noa Gruber\Desktop\Thesis Stuff\ThesisImplementations\results\scrambeled\digits\original_class_eval_37'
    
    total_accuracy_df_no_exp = pd.read_excel(f"{output_path}\\accuracy_2021.xlsx", sheet_name='no_explore_accuracy').rename(columns={'Unnamed: 0': 'index'}).set_index('index')
    total_accuracy_df = pd.read_excel(f"{output_path}\\accuracy_2021.xlsx", sheet_name='proposed_method_accuracy').rename(columns={'Unnamed: 0': 'index'}).set_index('index')
    total_accuracy_df_NB = pd.read_excel(f"{output_path}\\accuracy_2021.xlsx", sheet_name='NB_accuracy').rename(columns={'Unnamed: 0': 'index'}).set_index('index')
    total_accuracy_df_plainAL = pd.read_excel(f"{output_path}\\accuracy_2021.xlsx", sheet_name='random_accuracy').rename(columns={'Unnamed: 0': 'index'}).set_index('index')
    total_accuracy_df_densityAL = pd.read_excel(f"{output_path}\\accuracy_2021.xlsx", sheet_name='density_accuracy').rename(columns={'Unnamed: 0': 'index'}).set_index('index')
    total_accuracy_df_cluster_AL = pd.read_excel(f"{output_path}\\accuracy_2021.xlsx", sheet_name='cluster_accuracy').rename(columns={'Unnamed: 0': 'index'}).set_index('index')

    dfs_list = [total_accuracy_df, total_accuracy_df_no_exp, total_accuracy_df_NB, total_accuracy_df_plainAL, total_accuracy_df_densityAL, total_accuracy_df_cluster_AL]
    n_samp = np.arange(0, 0.26, 0.01)
    plot_labels = ['proposed_algorithm', 'pool_based_vote_entropy','naive_bayes', 'random_sampling', 'densityAL', 'cluster_AL']
    c_list_total = ['deeppink', 'lightseagreen', 'teal', 'darkslategray','aquamarine', 'deepskyblue', 'aqua', 'cadetblue', 'steelblue']

    fig, ax = plt.subplots()
    for i, df in enumerate(dfs_list):
        ax.plot(n_samp, df.loc['minority'], label=plot_labels[i], color=c_list_total[i])

    plt.ylim(-0.01,1.01)
    plt.xlabel('budget size (%)')
    plt.ylabel('average total accuracy')
    plt.legend(loc=2, fontsize=7)


    fig, ax = plt.subplots()
    for i, df in enumerate(dfs_list):
        ax.plot(n_samp, df.loc['majority'], label=plot_labels[i], color=c_list_total[i])

    plt.ylim(-0.01,1.01)
    plt.xlabel('budget size (%)')
    plt.ylabel('average total accuracy')
    plt.legend(loc=4, fontsize=8)


    fig, ax = plt.subplots()
    fig.text(0.5, 0.04, 'budget size (%)', ha='center')
    fig.text(0.04, 0.5, 'average total accuracy', va='center', rotation='vertical')
    plt.axis('off')

    for j in range(1,11):
        for i, df in enumerate(dfs_list):
            ax = fig.add_subplot(5,2,j)
            ax.plot(n_samp, df.iloc[j-1,], label=plot_labels[i], color=c_list_total[i])
            plt.ylim(-0.01, 1.01)
            ax.set_title(f'class {j-1}')
    plt.legend(loc=2)

if __name__ == "main":
    main()