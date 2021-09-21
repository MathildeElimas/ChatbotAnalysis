import numpy as np

from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from itertools import repeat, product, combinations


### scores de dispersion ###

def compute_silhouette(data, groups):
    return silhouette_score(data, groups)


### cohérence CV ###

def NPMI(i,j,D):
    '''
    D[i,j] = nombre de documents où les mots i et j apparaissent ensemble 
    D[i,i] = nombre de documents où le mot i apparait
    '''
    num = np.log((D[i,j]+1**(-50))*len(D)/(D[i,i]*D[j,j]))
    den = -np.log(D[i,j]+1**(-50)/len(D))
    return (num /den)

def compute_cv_coherence(result_components, doc_counts):
    term_rankings = [topic_word[:10] for topic_word in (-result_components).argsort()]
    D = np.dot(doc_counts.T, doc_counts)
    coherence = []
    for topic_index in range(len(term_rankings)):
        topic_score = []
        pairs_npmi= list(repeat([],10))
        i = 0
        for pair in product(term_rankings[topic_index], repeat = 2):
            pairs_npmi[i//10].append(NPMI(pair[0],pair[1],D))
            i+=1
        for pair in combinations(range(10),2):
            v0 = np.array(pairs_npmi[pair[0]]).reshape(-1, 1)
            v1 = np.array(pairs_npmi[pair[1]]).reshape(-1, 1)
            topic_score.append(cosine_similarity(v0,v1))
        topic_score = np.mean(topic_score)
        coherence.append(topic_score)
    return np.mean(coherence)

