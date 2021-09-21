import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF, TruncatedSVD

from func_methods.eval_func import *


def compute_model_path(model):
    if model == compute_cah:
        return os.getcwd() + '\\output_methods\\cah\\' 
    elif model == compute_kmeans:
        return os.getcwd() + '\\output_methods\\kmeans\\'
    elif model == compute_lsa:
        return os.getcwd() + '\\output_methods\\lsa\\'
    elif model == compute_nmf:
        return os.getcwd() + '\\output_methods\\nmf\\'
    elif model == compute_lda:
        return os.getcwd() + '\\output_methods\\lda\\'
    else:
        return 'Modèle inconnu'

def compute_models(data, model, n_tries):
    ''' cette fonction calcule le modèle pour plusieurs valeurs de n (nb de clusters)
    '''
    models_list = []
    for n in range(2,n_tries):
        models_list.append(model(data,n))
    return models_list

def compute_n_opt(data, data_count, model, n_tries, path):
    ''' fonction qui trace la valeur du score de cohérence
        pour chaque n (nb de clust)
    '''
    models_list = compute_models(data, model, n_tries)
    n = range(2,n_tries)
    coherence = []
    for model_results in models_list:
        coherence.append(compute_cv_coherence(model_results.components_, data_count))
    plt.plot(n, coherence)
    plt.title("Cohérence du modèle selon le nombre de topics")
    plt.savefig(path + '\\coherence_plot.png')
    return plt.show()

def compute_n_opt_cluster(data, model, n_tries, path):
    ''' fonction qui trace la valeur du score de silhouette
        pour chaque n (nb de clust)
    '''
    models_list = compute_models(data, model, n_tries)
    n = range(2,n_tries)
    silhouette = []
    for model_results in models_list:
        silhouette.append(compute_silhouette(data, model_results.fit_predict(data)))
    plt.plot(n, silhouette)
    plt.title("Silhouette du modèle selon le nombre de clusters")
    plt.savefig(path + '\\silhouette_plot.png')
    return plt.show()


def plot_dendo(data):
    ''' plot le dendogramme pour la méthode cah
    '''   
    Z = linkage(data)
    plt.title("CAH")
    dendrogram(dists, orientation='left') #color_threshold=n_clust_opt
    return plt.show()


### models

def compute_cah(data, n_clust):
    cah = AgglomerativeClustering(n_clust)
    cah.fit(data)
    return cah

def compute_kmeans(data, n_clust):
    kmeans = KMeans(n_clusters=n_clust, init ='k-means++', random_state=0)
    kmeans.fit(data)
    return kmeans

def compute_lsa(data, n_clust):
    svd = TruncatedSVD(n_components=n_clust, algorithm='randomized', n_iter=100, random_state=122)
    svd.fit(data)
    return svd

def compute_lda(data, n_clust):
    lda = LDA(n_components=n_clust, n_jobs=-1)
    lda.fit(data)
    return lda

def compute_nmf(data, n_clust):
    nmf = NMF(n_components=n_clust)
    nmf.fit(data)
    return nmf
    