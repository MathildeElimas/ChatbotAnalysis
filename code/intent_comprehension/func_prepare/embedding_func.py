import pandas as pd 
import spacy

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#### WORD EMBEDDING ####


def document_term(texts, vocab=None, method = 'count'):
    ''' 
    retourne le "vectorizer" et la matrice documents-termes 
    selon la méthode : count, one_hot, tfidf
    '''
    if method == 'count':
        c = CountVectorizer(max_features=10000, vocabulary = vocab, ngram_range = (1,3))
        output = c.fit_transform(texts.values.astype('U'))
        output = pd.DataFrame(output.toarray(), columns = vocab)
        return(output)
    if method == 'onehot':
        output = document_term(texts, vocab, method = 'count')
        output = output.gt(0).astype(int) 
        return(output)
    if method == 'tfidf':
        t = TfidfVectorizer(max_features=10000, vocabulary = vocab, ngram_range = (1,3))
        output = t.fit_transform(texts.values.astype('U'))
        output = pd.DataFrame(output.toarray(), columns = vocab)
        return(output)
    
def padding_docs(encoded_doc, max_length):
    if type(encoded_doc) != np.ndarray:
        return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))
    else:
        for i in range(len(encoded_doc)):
            matrix = encoded_doc[i]
            pad = np.zeros((max_length-len(matrix), len(matrix[0])))
            encoded_doc[i] = np.concatenate((matrix, pad), axis = 0)
        return np.stack(encoded_doc, axis=0)

def document_matrix(texts, method = 'label'):
    ''' 
    création de la matrice documents-matrice selon la méthode :
    label, word2vec, (doc2vec)
    '''
    max_length = max(texts.astype('U').apply(lambda x : len(x.split())))
    if method == 'label':
        l = LabelEncoder()
        labels = l.fit(' '.join(texts.astype('U')).split())
        output = [l.transform(x.split()) for x in texts.astype('U')]
        output = padding_docs(output, max_length)
        return(output)
    if method == 'word2vec':
        output = []
        for text in texts.astype('U') : 
            doc = nlp(text)
            output.append(np.array([x.vector for x in doc]))
        output = np.array(output)
        output = padding_docs(output, max_length)
        return(output)
    
