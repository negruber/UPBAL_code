import pandas as pd
import matplotlib.pyplot as plt


def main():
    output_path = r'C:\Users\Noa Gruber\Desktop\Thesis Stuff\ThesisImplementations\results\scrambeled\digits\original_class_eval_37'
    minority_classes = [3,7]

    total_accuracy_df_no_exp = pd.read_excel(f"{output_path}\\accuracy_2021.xlsx", sheet_name='no_explore_accuracy').rename(columns={'Unnamed: 0': 'index'}).set_index('index')
    total_accuracy_df = pd.read_excel(f"{output_path}\\accuracy_2021.xlsx", sheet_name='proposed_method_accuracy').rename(columns={'Unnamed: 0': 'index'}).set_index('index')
    total_accuracy_df_NB = pd.read_excel(f"{output_path}\\accuracy_2021.xlsx", sheet_name='NB_accuracy').rename(columns={'Unnamed: 0': 'index'}).set_index('index')
    total_accuracy_df_plainAL = pd.read_excel(f"{output_path}\\accuracy_2021.xlsx", sheet_name='random_accuracy').rename(columns={'Unnamed: 0': 'index'}).set_index('index')
    total_accuracy_df_densityAL = pd.read_excel(f"{output_path}\\accuracy_2021.xlsx", sheet_name='density_accuracy').rename(columns={'Unnamed: 0': 'index'}).set_index('index')
    total_accuracy_df_cluster_AL = pd.read_excel(f"{output_path}\\accuracy_2021.xlsx", sheet_name='cluster_accuracy').rename(columns={'Unnamed: 0': 'index'}).set_index('index')

    queried_df_no_exp = pd.read_excel(f"{output_path}\\queried_analysis\\total_algorithms.xlsx", sheet_name="no_explor")
    queried_df_proposed = pd.read_excel(f"{output_path}\\queried_analysis\\total_algorithms.xlsx", sheet_name="proposed_algorithm")
    queried_df_NB = pd.read_excel(f"{output_path}\\queried_analysis\\total_algorithms.xlsx", sheet_name="NB")
    queried_df_density = pd.read_excel(f"{output_path}\\queried_analysis\\total_algorithms.xlsx", sheet_name="densityAL")
    queried_df_cluster = pd.read_excel(f"{output_path}\\queried_analysis\\total_algorithms.xlsx", sheet_name="clusterAL")
    queried_df_random = pd.read_excel(f"{output_path}\\queried_analysis\\total_algorithms.xlsx", sheet_name="randomAL")


    dfs_list = [total_accuracy_df, total_accuracy_df_no_exp, total_accuracy_df_NB, total_accuracy_df_plainAL, total_accuracy_df_densityAL, total_accuracy_df_cluster_AL]
    queried_dfs_list = [queried_df_proposed, queried_df_no_exp, queried_df_NB, queried_df_random, queried_df_density, queried_df_cluster]
    n_samp = range(0,26)
    plot_labels = ['proposed_algorithm', 'pool_based_vote_entropy','naive_bayes', 'random_sampling', 'densityAL', 'cluster_AL']
    c_list_total = ['deeppink', 'lightseagreen', 'teal', 'darkslategray','aquamarine', 'deepskyblue', 'aqua', 'cadetblue', 'steelblue']
    class_name = 9
    fig, ax = plt.subplots()
    for i, queried_df in enumerate(queried_dfs_list):
        ax.plot(n_samp, queried_df.loc[class_name], label=plot_labels[i], color=c_list_total[i])

    plt.xlabel('budget size (%)')
    plt.ylabel('average number of queried instances')
    plt.legend(loc=2)

    fig, ax = plt.subplots()
    for i, df in enumerate(dfs_list):
        ax.plot(n_samp, df.loc[class_name], label=plot_labels[i], color=c_list_total[i])

    plt.ylim(-0.01,1.01)
    plt.xlabel('budget size (%)')
    plt.ylabel('average accuracy')
    plt.legend(loc=4, fontsize=10)


    barWidth = 0.1
    fig = plt.subplots(figsize=(10, 6))

    class_names = [f"class {x}" for x in range(10)]

    # Set position of bar on X axis
    br1 = np.arange(queried_df_proposed.shape[0])
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    br6 = [x + barWidth for x in br5]
    
    # Make the plot
    budget = 10
    plt.bar(br1, total_accuracy_df.iloc[0:10,budget], color='deeppink', width=barWidth, edgecolor='grey', label='proposed_algorithm')
    plt.bar(br2, total_accuracy_df_no_exp.iloc[0:10,budget], color='lightseagreen', width=barWidth, edgecolor='grey', label='pool_based_vote_entropy')
    plt.bar(br3, total_accuracy_df_NB.iloc[0:10,budget], color='teal', width=barWidth, edgecolor='grey', label='naive_bayes')
    plt.bar(br4, total_accuracy_df_plainAL.iloc[0:10,budget], color='darkslategray', width=barWidth, edgecolor='grey', label='random_sampling')
    plt.bar(br5, total_accuracy_df_densityAL.iloc[0:10,budget], color='aquamarine', width=barWidth, edgecolor='grey', label='densityAL')
    plt.bar(br6, total_accuracy_df_cluster_AL.iloc[0:10,budget], color='deepskyblue', width=barWidth, edgecolor='grey', label='cluster_AL')

    # Adding Xticks
    plt.xlabel('Original Class', fontweight='bold', fontsize=10)
    plt.ylabel('Average Accuracy in Budget 0.1', fontweight='bold', fontsize=10)
    plt.xticks([r + barWidth for r in range(queried_df_proposed.shape[0])], class_names)
    plt.ylim(0.00,1.2)
    plt.legend(loc=9,fontsize=7)
    plt.show()

if __name__ == "main":
    main()