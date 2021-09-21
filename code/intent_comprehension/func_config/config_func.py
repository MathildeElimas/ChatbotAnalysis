import sys
import os
import pandas as pd
import hashlib
import json

from IPython.display import clear_output

sys.path.insert(1, os.getcwd().replace('func_config', 'func_prepare'))

from func_prepare.data_prepare import *


def exists(path):
    if os.path.exists(path):
        return True
    else : 
        return False

def compute_id_end(json_ids, params):
    '''calcule l'id correspondant aux paramètres
    '''
    with open(json_ids, 'r') as f:
        ids = dict(json.load(f))
    ID = []
    for parameter, value in params.items():
        if parameter in list(ids.keys()):
            ID.append(ids[parameter][str(value)])
        else:
            ID.append(hashlib.md5(str(value).encode('utf-8')).hexdigest()[:2])

    ID = '_'.join(ID)
    return ID

def compute_path(params_id, params, end = 'data'):
    '''calcule le chemin de l'objet end ('data', 'data_clean', 'embedding', 'vocabulary')
       selon les paramètres donnés
    '''
    path = '\\output_data'
    data_value = params['data']
    emb_value = params['embedding']
    if end == 'data':
        return '\\'.join([path, data_value +'.csv'])

    elif end == 'data_clean':
        return '\\'.join([path, data_value, params_id[:11]])
    
    elif end == 'vocabulary':
        return '\\'.join([path, data_value, 'vocab', params_id[:17]])
    
    elif end == 'embedding':
        return '\\'.join([path, data_value, 'vocab', emb_value, params_id])
        

def add_config(params, json_ids):
    ''' ajoute une ligne au dataframe de configuration
    '''
    params_file = pd.read_pickle(os.getcwd()+'\\func_config\\params')
    ID = compute_id_end(json_ids, params)
    
    if not ID in list(params_file['id']):
        
        print("Combinaison de paramètres inconnues")
        print("Création des éléments...")
        
        params_file = params_file.append({'id': ID,
                            'data': params['data'],
                            'clean_regul': params['regul'],
                            'clean_remove_stopwords': params['remove_stopwords'],
                            'clean_stopwords': params['stop'],
                            'clean_keep': params['keep'],                            
                            'vocab_min_freq': params['min_freq'],
                            'vocab_max_freq': params['max_freq'],
                            'embedding': params['embedding'],
                            'data_path': compute_path(ID, params, 'data_clean'), 
                            'vocab_path': compute_path(ID, params, 'vocabulary'), 
                            'embedding_path': compute_path(ID, params, 'embedding')}, ignore_index=True)

        clear_output(wait = True)
        
        if not exists(os.getcwd() +  compute_path(ID, params, 'data_clean')):
            print("Nettoyage de la base de données")
            clean_time = create_topic_data(os.getcwd() +  compute_path(ID, params, 'data'), 
                                           os.getcwd() +  compute_path(ID, params, 'data_clean'), params)
            clear_output(wait = True)

        if not exists(os.getcwd() +  compute_path(ID, params, 'vocabulary')):
            print("Création du vocabulaire")
            vocab_time = create_vocab(os.getcwd() +  compute_path(ID, params, 'data_clean'), 
                                      os.getcwd() +  compute_path(ID, params, 'vocabulary'), params)
            clear_output(wait = True)

        if not exists(os.getcwd() +  compute_path(ID, params, 'embedding')):
            print("Création de la matrice d'embedding")
            embedding_time = embed_data(os.getcwd() +  compute_path(ID, params, 'data_clean'),
                                        os.getcwd() +  compute_path(ID, params, 'vocabulary'), 
                                        os.getcwd() +  compute_path(ID, params, 'embedding'), params)
            clear_output(wait = True)
        
        params_file.to_pickle(os.getcwd()+'\\func_config\\params')
        
    
    return ID


