'''This is the deep learning implementation in Keras for graph classification based on FGSD graph features.'''

import numpy as np
import scipy.io
import networkx as nx
from grakel import datasets
from scipy import sparse
from sklearn.utils import shuffle
from scipy import linalg
import time
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

########### LOAD DATA ###########

def return_dataset(file_name):
    dd = datasets.fetch_dataset(file_name, verbose=True)
    graph_list = []
    for gg in dd.data:
        v = set([i[0] for i in gg[0]]).union(set([i[1] for i in gg[0]]))
        g_ = nx.Graph()
        g_.add_nodes_from(v)
        g_.add_edges_from([(i[0], i[1]) for i in gg[0]])
        graph_list.append(g_)
    y = dd.target
    return graph_list, np.array(y)

########### SET PARAMETERS ###########

def FGSD(graphs,labels):
    feature_matrix=[]
    nbins=200
    range_hist=(0,20)
    for A in graphs:
        L=nx.laplacian_matrix(A)
        ones_vector=np.ones(L.shape[0])
        fL=np.linalg.pinv(L.todense())
        S=np.outer(np.diag(fL),ones_vector)+np.outer(ones_vector,np.diag(fL))-2*fL
        hist, bin_edges = np.histogram(S.flatten(),bins=nbins,range=range_hist)
        feature_matrix.append(hist)
    feature_matrix=np.array(feature_matrix)
    feature_matrix,data_labels_binary=shuffle(feature_matrix, labels)
    return feature_matrix, data_labels_binary


########### TRAIN AND VALIDATE MODEL ###########


data = ["MUTAG","PROTEINS_full","NCI1","NCI109","DD","COLLAB","REDDIT-BINARY","REDDIT-MULTI-5K","IMDB-BINARY","IMDB-MULTI"]
#file = open("fgsd_res.csv",'a',newline='')
#res_writer = csv.writer(file, delimiter = ',', quotechar='|', quoting= csv.QUOTE_MINIMAL)
#header = ["dataset","accuracy","time"]
#res_writer.writerow(header)
for d in data:
    graphs, labels = return_dataset(dataset)
    print(" {} dataset loaded with {} number of graphs".format(d, len(graphs)))
    start = time.time()
    emb,y = FGSD(graphs, labels)
    end = time.time()
    print("total time taken: ", end-start)
    model = RandomForestClassifier(n_estimators = 100)
    res = cross_val_score(model,emb, y, cv = 10, scoring='accuracy')
    print("10 fold cross validation accuracy: {}, for dataset ; {}".format(np.mean(res)*100, d))
    to_write = [d, np.mean(res)*100, end-start]
    print(to_write)
    #res_writer.writerow(to_write)
    #file.flush()
#file.close()
    







