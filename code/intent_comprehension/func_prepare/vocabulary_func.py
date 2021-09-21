import collections
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from nltk import ngrams

from wordcloud import WordCloud

from func_prepare.embedding_func import *

def create_vocabulary(column):
    vocab = column.apply(lambda text : text.split()) 
    vocab = np.sum(list(vocab))
    vocab_dic = dict(collections.Counter(vocab))
    vocab = dict(list(sorted(vocab_dic.items(), key=lambda kv: kv[1], reverse = True)))
    return vocab

def process_vocabulary(vocab, data, min_freq = 0.01, max_freq = 0.9):
    '''
    stop : liste des mots supplémentaire à enlever du vocabulaire
    '''
    # on supprime les mots qui apparaissent - de 2 fois
    vocab = dict([(k,v) for (k, v) in vocab.items() if (int(v) > 2)])

    doc_embedding = document_term(data.CommentClean, list(vocab.keys()), method ='onehot')
    distribution = doc_embedding.apply(sum,axis = 0)
    
    # on ne garde que les mots qui n'aparaissent pas dans + de max_freq% des documents et dans - de min_freq%
    docs_stopwords = list(distribution[(distribution/len(data) > max_freq) | 
                                       (distribution/len(data) < min_freq)].index)    
    
    vocab = pd.DataFrame(vocab.items(), columns = ['word','value'])       
    vocab = vocab[[True if x not in docs_stopwords else False for x in list(vocab.word)]]
    
    vocab['id'] = range(len(vocab)) 
    vocab['freq'] = vocab['value']/len(data)
    return vocab

def save_outputs(vocab, vocab_vis_file):
    ''' infos liées à ce vocabulaire
    '''  
    ### Top 15 mots du vocab ###
    barplot = plt.figure(1)
    plt.subplot(1,1,1)
    plt.bar(list(vocab.word)[:15], list(vocab.value)[:15])
    plt.title("Top 15 words of vocabulary")
    plt.tick_params(axis="x", rotation = 45)
    plt.close(barplot)

    ### Plot WordMap ###
    long_string = ' '.join((vocab.word+' ')*vocab.value)
    wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue', collocations=False)
    wordcloud.generate(long_string)

#     # Visualize the word cloud
#     wordcloud.to_image()

    vis_results = [len(vocab), barplot, wordcloud]
    
    with open(vocab_vis_file, 'wb') as f:
        pickle.dump(vis_results, f)