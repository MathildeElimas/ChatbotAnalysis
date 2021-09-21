import pickle
import os

from IPython.display import clear_output

from func_methods.compute_models import *
from func_methods.result_func import *

def print_menu(question, params, response_list = False):
    ''' affiche le menu pour faciliter les choix et renvoie la réponse choisie
    '''
    # affichage du menu
    print(question)
    choices = []
    for key, value in params.items():
        print(str(key)+ ". " + value[1])
    
    # cas où il y a une liste de valeurs à requêter
    if response_list:
        answer = []
        while True:
            ask = get_response(question, params)
            answer.append(ask)
            if ask:
                break
        return answer
    else :
        return get_response(question, params)
            

def get_response(question, params):
    ''' renvoie la valeur du paramètre correspondante à 
        la réponse choisie
    '''
    ask = eval(input())
    if ask not in list(params.keys()):
        print('erreur saisie')
        print_menu(question, params)
    else : 
        return params[ask][0]
        


def ask_params(): 
    ''' fonction qui demande les paramètres à utiliser 
    '''
    data = print_menu('Quel dataframe on utilise ?', {1: ['autres','autres'],
                                               2: ['bbox','bbox'], 
                                               3: ['continuer','continuer'],
                                               4: ['facture','facture'],
                                               5: ['reseau','reseau']})
    clear_output(wait=True)
    
    # regularisation
    regul = print_menu("Quelle type de troncature ?", {1: ['lem','lemmatisation'], 
                                                       2: ['stem','stemming'], 
                                                       3: [None,'aucune']})
    clear_output(wait=True)

    # stopwords
    remove_stopwords = print_menu("On enlève les stopwords ?", {1: [True, 'oui'], 
                                                                2: [False, 'non']})
    clear_output(wait=True)
    stop = []
    if remove_stopwords:
        while True:
            condition = print_menu("Liste de mots supplémentaires à retirer :",{1: [False, 'mot'],
                                                                            2: [True, 'fin']})
            if not condition:
                stop.append(input())
                clear_output(wait=True)

            else : 
                break
    
    clear_output(wait=True)

    # keep
    keep = []
    while True : 
        ask = print_menu("Grammaire à conserver :", {1: ['NOUN','noms communs'],
                                                     2: ['VERB','verbes'],
                                                     3: ['ADJ','adjectifs'], 
                                                     4: [True,'stop']})
        if ask in ['NOUN','VERB','ADJ']:
            keep.append(ask)
            clear_output(wait=True)

        elif ask :
            break
    
    clear_output(wait=True)

    print("Nombre d'occurrences minimal :")
    min_occur = eval(input())
    clear_output(wait=True)

    print("Fréquence d'apparition minimale d'un mot dans les documents:")
    min_freq = eval(input())/100
    clear_output(wait=True)
    
    print("Fréquence d'apparition maximale d'un mot dans les documents:")
    max_freq = eval(input())/100
    clear_output(wait=True)  
    
    embedding = print_menu("Quelle mesure pour l'embedding ?", {1: ['onehot','apparition'],
                                                                2: ['count','fréquence d\'apparition'],
                                                                3: ['tfidf','TF-IDF']})
    clear_output(wait=False)
    
    params = {'data': data,
            'regul': regul, 
            'remove_stopwords': remove_stopwords, 
            'stop': stop, 
            'keep': keep,
            'min_occur': min_occur,
            'min_freq': min_freq,
            'max_freq': max_freq,
            'embedding': embedding}
    print('Paramètres : ', params)
    return params

#########################################
############ TOPIC MODELLING ############
#########################################

def ask_do_model(params):
    ''' demande quel modèle on fait, calcule les différentes possibilités,
        retourne le modèle pour lequel le nombre de cluster est optimal selon mon choix
    '''
    # choix du modèle
    data = pd.read_pickle(os.getcwd() + params['embedding_path'].values[0])
    if params['embedding'].values[0] != 'count':
        data_count = pd.read_pickle(os.getcwd() + params['embedding_path'].values[0]+ '_count')
    else :
        data_count = data
    model = print_menu('Quel modèle on utilise ?', {1: [compute_kmeans, 'KMeans'],
                                                    2: [compute_cah,'CAH'],
                                                    3: [compute_lsa,'LSA'], 
                                                    4: [compute_nmf,'NMF'],
                                                    5: [compute_lda,'LDA']})
    
    # application du modèle
    output_path = compute_model_path(model) + params['id'].values[0] + '\\'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if model == compute_cah or model == compute_kmeans:
        compute_n_opt_cluster(data, model, 20, output_path)
    else:
        compute_n_opt(data, data_count, model, 20, output_path)
    
    # choix du nombre de clusters
    print('Quel nombre de clusters ?')
    n_clust = eval(input())
    
    # apllication et enregistrement du modèle avec le nombre de cluster final
    model_results = model(data, n_clust)
    
    if model == compute_cah or model == compute_kmeans:
        describe_topics_cluster(data, data_count, model_results, output_path)
        wordmap_topic_cluster(data, data_count, model_results, output_path)

    else:
        describe_topics(data, model_results, output_path)
        wordmap_topic(data, model_results, output_path)
    
    with open(output_path+'model.h5', 'wb') as out:
        pickle.dump(model_results, out)
    return None