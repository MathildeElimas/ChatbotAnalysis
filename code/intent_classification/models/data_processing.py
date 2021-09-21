import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import re
import nltk
import spacy
import collections

nltk.download('wordnet')
nltk.download("stopwords")
nltk.download("punkt")

# ce ne sont pas des traitements de textes français

from spacy_lefff import LefffLemmatizer


from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams

from stop_words import get_stop_words

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



########################### Data preparation ##########################




def load_data(filename, sep = ',', header = 0, encoding = 'utf-8', names = ["Sentence", "Intent"], index_col = None):
    df = pd.read_csv(filename, encoding = encoding, 
                     names = names, sep = sep, header = header, 
                     index_col=False)
    print(df.head())
    intent = df["Intent"] # .apply(lambda x: x.lower().split('-'))
    unique_intent = list(set(intent))
    sentences = list(df["Sentence"])

    return (intent, unique_intent, sentences)


def clean_sentences(sentences, n_grams = 1, lem = True):
    ''' 
    lem = True, False
        True for lemmatization, False for stemming
    '''
    if lem :
        nlp = spacy.load("fr_core_news_lg")
        lemmatizer = LefffLemmatizer()
        nlp.add_pipe(lemmatizer, name='lefff')
    else :
        stemmer = FrenchStemmer()
        
    #removing stopwords
    stop = list(set(stopwords.words('french') +  get_stop_words('fr'))) + ['cln']
    
    words = []
    for s in sentences:
        clean = re.sub(r'[^\w+]', " ", s).strip()
        clean = re.sub('[0-9]','', clean)
        clean = clean.lower()
        
        if lem :
            #lemmatizing
            clean = ' '.join([word._.lefff_lemma for word in nlp(clean) if word._.lefff_lemma != None])
        else :
            #stemming
            clean = stemmer.stem(clean)
            
        #remove stopowrds
        if n_grams <= 1:
            words.append(' '.join(list(set(clean.split())-set(stop))))
        else :
            words.append(' '.join(x) for x in ngrams(clean.split(), n_grams))
     
    idx = [i for i in range(len(words)) if len(words[i].split())>1]
            
    return idx, list(map(words.__getitem__, idx))


def create_tokenizer(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    token = Tokenizer(filters = filters)
    token.fit_on_texts(words)
    return token


    

########################### Embeddings ##############################


def intent_embedding(intents, encoding = 'one_hot'):
    '''
    encoding = str : 'one_hot', 'label'
    '''
    if encoding == 'one_hot':
        o = OneHotEncoder(sparse = False)
        return o, o.fit_transform(np.array(intents).reshape(-1, 1))
    if encoding == 'label':
        l = LabelEncoder()
        return l, l.fit_transform(intents)

def text_embedding(texts, vocab=None, encoding = 'tfidf', max_length = 50):
    '''
    encoding = str : 'label', 'tfidf', 'count', 'word2vec'
        type of embedding
    '''
    if encoding == 'one_hot':
        output = pd.DataFrame(text_embedding(texts, vocab, encoding = 'count'), columns = vocab)
        output = output.gt(0).astype(int) 
        return output
    if encoding == 'tfidf':
        t = TfidfVectorizer(max_features=5000, vocabulary = vocab)
        output = t.fit_transform(texts)
        return output.toarray()
    if encoding == 'count':
        c = CountVectorizer(max_features=5000, vocabulary = vocab)
        output = c.fit_transform(texts)
        return output.toarray()
    
    if encoding == 'label':
        l = LabelEncoder()
        labels = l.fit(' '.join(texts).split())
        output = [l.transform(x.split()) for x in texts]
        output = padding_docs(output, max_length)
        return output
    if encoding == 'word2vec':
        nlp = spacy.load('fr_core_news_md')
        output = []
        for text in texts : 
            doc = nlp(text)
            output.append(np.array([x.vector for x in doc]))
        output = np.array(output)
        output = padding_docs(output, max_length)
        return output

def padding_docs(encoded_doc, max_length):
    if type(encoded_doc) != np.ndarray:
        return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))
    else:
        for i in range(len(encoded_doc)):
            matrix = encoded_doc[i]
            pad = np.zeros((max_length-len(matrix), len(matrix[0])))
            encoded_doc[i] = np.concatenate((matrix, pad), axis = 0)
        return np.stack(encoded_doc, axis=0)


def create_vocabulary(texts):
    vocab = ' '.join(texts)
    vocab = dict(collections.Counter(vocab.split()))
    
    vocab = dict([(k,v) for (k, v) in vocab.items() if (int(v) > 2)])
    
    doc_embedding =  text_embedding(texts, list(vocab.keys()), 'one_hot')
    distribution = doc_embedding.apply(sum,axis = 0)
    
    # on ne garde que les mots qui n'aparaissent pas dans + de 90% des documents et dans - de 0.01%
    docs_stopwords = list(distribution[(distribution/len(texts) > 0.9) | 
                                       (distribution/len(texts) < 0.01)].index)   
    
    vocab = pd.DataFrame(vocab.items(), columns = ['word','value'])       
    vocab = vocab[[True if x not in docs_stopwords else False for x in list(vocab.word)]]
    
    vocab = vocab.set_index('word').T.to_dict('records')[0]

    vocab = dict(list(sorted(vocab.items(), key=lambda kv: kv[1], reverse = True)))
    return vocab


################ Courbe ROC #####################



def confus_matrix(y_prob,y_pred,y_test,seuil):
    VP = sum((y_pred==y_test) & (y_prob>seuil))
    FP = sum((y_pred!=y_test) & (y_prob>seuil))
    VN = sum((y_pred==y_test) & (y_prob<=seuil))
    FN = sum((y_pred!=y_test) & (y_prob<=seuil))
    return(np.array([[VP,FP],[VN,FN]]))

def ROC(y_prob,y_pred,y_test):
    
    sensitivity=[]
    specificity=[]
    
    with mp.Pool() as pool:
    
        for seuil in np.linspace(0,1,30) :
            cm = confus_matrix(y_prob,y_pred,y_test,seuil)
            sensitivity.append(cm[0,0]/(cm[0,0]+cm[1,1]) if (cm[0,0]+cm[1,1]) != 0 else 0 )
            specificity.append(1 - (cm[1,0]/(cm[1,0]+cm[0,1])) if (cm[1,0]+cm[0,1]) != 0 else 1)   
            
    return([sensitivity,specificity])

def plot_ROC(dic):
    
    fig = plt.figure(figsize=(15,5))
    plt.title('Courbe ROC multi-classes')
    plt.xlabel('1-spécificité')
    plt.ylabel('sensibilité')
    for modele in dic.keys():
        plt.plot(dic[modele][1],dic[modele][0],label = modele)
        plt.legend()
        
    plt.show()

