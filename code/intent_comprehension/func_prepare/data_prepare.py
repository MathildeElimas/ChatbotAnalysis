import sys 
import pandas as pd
import time
import matplotlib.pyplot as plt

from func_prepare.clean_func import *
from func_prepare.vocabulary_func import *
from func_prepare.embedding_func import *



def create_topic_data(import_loc, export_loc, params):
    ''' cette fonction crée et sauvegarde le dataframe utilisé pour l'étude au format csv
    '''
    t0 = time.time()
    dialogues = pd.read_csv(import_loc, index_col = 0)
    
    dialoguesParents = dialogues.groupby('ParentId').head(1)
    # sortie : len(dialogues)
        
    dialoguesParents = clean_sentences(dialoguesParents, params) 
    
    dialoguesParents.to_pickle(export_loc)
    clean_time = time.time()-t0
    return clean_time

def create_vocab(import_loc, export_loc, params):
    ''' cette fonction crée le vocabulaire d'étude 
    '''
    t0 = time.time()
    dialoguesParents = pd.read_pickle(import_loc)
    text = dialoguesParents['CommentClean']
    
    vocabulary = create_vocabulary(text)
    vocabulary = process_vocabulary(vocabulary, 
                                    dialoguesParents, 
                                    params['min_freq'], 
                                    params['max_freq'])
    
    vocabulary.to_pickle(export_loc)
    save_outputs(vocabulary, export_loc+'_vis')
    vocab_time = time.time()-t0
    return vocab_time

def embed_data(data_loc, vocab_loc, export_loc, params):
    ''' cette fonction crée la matrice d'embedding
    à compléter si l'embedding peut avoir 2 types
    '''
    t0 = time.time()
    data = pd.read_pickle(data_loc)
    vocabulary = pd.read_pickle(vocab_loc)
    matrix = document_term(data.CommentClean, list(vocabulary.word), params['embedding'])
    matrix.to_pickle(export_loc)
    if params['embedding'] != 'count':
        data_count = document_term(data.CommentClean, list(vocabulary.word))
        data_count.to_pickle(export_loc+'_count')
    embedding_time = time.time()-t0
    return embedding_time
    