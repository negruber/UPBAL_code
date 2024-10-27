import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import networkx as nx
import matplotlib.pyplot as plt

def create_train_test_dfs(n_te, data):
    sample_num_dict = ((data.groupby(by='labels').size())*n_te).to_dict()
    print(sample_num_dict)
    test_set = pd.DataFrame()
    for class_name in sample_num_dict:
        temp_class_df = data[data['labels'] == class_name].copy()
        test_set = pd.concat([test_set, temp_class_df.sample(n=np.int(sample_num_dict[class_name]))])
    train_set = data.drop(test_set.index).reset_index(drop=True)
    return train_set, test_set

def main():
    ######### fashion data (the same works for digits data)
    data_path = "C:\\Users\\Noa Gruber\\Desktop\\Thesis Stuff\\ThesisImplementations\\results\\fashion_MNIST\\"
    fashion_data_train = pd.read_csv(f'{data_path}fashion-mnist_train.csv').rename(columns={'label': 'labels'})
    fashion_data_test = pd.read_csv(f'{data_path}fashion-mnist_test.csv').rename(columns={'label': 'labels'})

    # set dataset here
    X = fashion_data_train.drop('labels', axis=1)
    X_labels = fashion_data_train['labels']
    test_set = fashion_data_test.copy()
    n_class = X_labels.nunique()

    n_trees = 50
    RF_mod = RandomForestClassifier(n_estimators=n_trees)
    RF_mod.fit(X=X, y=np.ravel(X_labels, order='C'))
    X_preds = RF_mod.predict(test_set.drop('labels', axis=1))

    conf_matrix = confusion_matrix(y_true=np.array(test_set['labels']), y_pred=np.array(X_preds))

    prob_mat = RF_mod.predict_proba(test_set.drop('labels', axis=1))
    max_probs = pd.DataFrame(pd.DataFrame(prob_mat).max(axis=1)).rename(columns={0:'p_hat'})
    prob_df = pd.concat([test_set['labels'].reset_index(drop=True), max_probs, pd.DataFrame(X_preds).rename(columns={0: 'preds'})], axis=1)

    C = np.zeros((n_class,n_class))
    for i in range(n_class):
        curr_label = prob_df[prob_df['labels'] == i]
        for j in range(n_class):
            if j in curr_label['preds'].unique().tolist():
                curr_pred_df = curr_label[curr_label['preds'] == j]
                C[i,j] = curr_pred_df['p_hat'].sum()

    G = nx.Graph()
    for i in range(n_class):
        for j in range(n_class):
            if i > j:
                G.add_edge(f'{i}', f'{j}', weight=C[i,j]+C[j,i])

    T = nx.maximum_spanning_tree(G)
    sorted(T.edges(data=True))
    elarge = [(u, v) for (u, v, d) in T.edges(data=True) if d['weight'] > 0.5]
    esmall = [(u, v) for (u, v, d) in T.edges(data=True) if d['weight'] <= 0.5]
    pos = nx.spring_layout(T)  # positions for all nodes
    # nodes
    nx.draw_networkx_nodes(T, pos, node_size=700)
    # edges
    nx.draw_networkx_edges(T, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(T, pos, edgelist=esmall, width=6, alpha=0.5, edge_color='b', style='dashed')
    # labels
    nx.draw_networkx_labels(T, pos, font_size=20, font_family='sans-serif')
    plt.axis('off')
    plt.show()

    def color_nodes(graph):
        color_map = {}
        # Consider nodes in descending degree
        for node in sorted(graph, key=lambda x: len(graph[x]), reverse=True):
            neighbor_colors = set(color_map.get(neigh) for neigh in graph[node])
            print(node, neighbor_colors)
            color_map[node] = next(
                color for color in range(len(graph)) if color not in neighbor_colors
            )
        return color_map

    color_nodes(T)

if __name__ == "main":
    main()