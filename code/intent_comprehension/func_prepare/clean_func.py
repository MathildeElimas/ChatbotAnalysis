import pandas as pd
import numpy as np
import re
import nltk
import spacy
import os 
import unicodedata

# nltk.download("stopwords")

from spacy_lefff import LefffLemmatizer

from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordPOSTagger

from stop_words import get_stop_words


stfd_path = 'C:/Users/MathildeElimas/Documents/stanford-postagger-full-2020-08-06/' # path for stanford_postagger
jar = stfd_path + 'stanford-postagger.jar'
model = stfd_path + 'models/french-ud.tagger'

java_path = 'C:/java/jdk/bin/java.exe'
os.environ['JAVAHOME'] = java_path

#### TEXT PROCESSING ####

def create_process(params):
    ''' 
    Cette fonction retourne une liste (ordonnée) de fonctions de traitement de texte
    # params : dictionnaire
        regul = None (default) for no regularisation
                'lem' for lemmatization, 
                'stem' for stemming.
        remove_stopwords = True/False
        stop_w = list of other words to delete with stopwords
        keep = list of syntaxical labels to keep (N, A, V ...)
    '''
    process_list = []      
        
    if params['regul'] == 'lem' :
        nlp = spacy.load("fr_core_news_lg")
        lemmatizer = LefffLemmatizer()
        nlp.add_pipe(lemmatizer, name='lefff')
        
        def lemmatize(text):
            return [word._.lefff_lemma for word in nlp(text)]
        
        process_list.append(lemmatize)

    
    elif params['regul'] == 'stem' :
        stemmer = FrenchStemmer()
        
        def stemming(text):
            return stemmer.stem(text).split()
        
        process_list.append(stemming)
        
    if params['remove_stopwords']:
        stop = list(set(stopwords.words('french') +  get_stop_words('fr')))
        if len(params['stop'])>0 : 
            stop = stop + params['stop']
        #         stop = [unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode('utf8') for x in stop]

        def remove_stops(text):
            text = list(set(text.split())-set(stop))
            return text 

        process_list.append(remove_stops)
       
    if len(params['keep']) != 0 : 
        pos_tagger = StanfordPOSTagger(model, jar, encoding='utf-8')
        
        def keep_grammar(text):
            test = pd.DataFrame(pos_tagger.tag(text.split()))
            if len(test)>0:
                return list(test[test[1].isin(params['keep'])][0])
            else:
                return None
        
        process_list.append(keep_grammar)
    
    return process_list

def clean_text(text, process_list):
    ''' Cette fonction renvoie une chaine de caractères nettoyée :
        suppression de la ponctuation, des nombres, des espaces en trop... et
        applique l'ensemble des fonctions de traitement de texte listées dans process_list.
    '''
    clean = re.sub(r'[^\w+]', " ", text).strip()
    clean = re.sub('[0-9]','', clean)
    clean = clean.lower()
    for func in process_list:
        if func(clean) != None:
            clean = ' '.join([words for words in func(clean) if words != None])
        else: 
            clean = ''
            break
    clean = unicodedata.normalize('NFKD', clean).encode('ASCII', 'ignore').decode('utf8')

    return clean
         
def clean_sentences(data, params):
    ''' Cette fonction crée une nouvelle colonne dans un dataframe pandas en appliquant à la 
        variable 'CommentClean' présente dans le dataframe une fonction de nettoyage de texte.
    '''
    cleaning = create_process(params)
    data['CommentClean'] =  data.CommentBody.astype('U').map(lambda text : clean_text(text, cleaning))
    return data[['CommentBody','CommentClean']]
    
