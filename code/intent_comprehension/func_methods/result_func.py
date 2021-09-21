import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from wordcloud import WordCloud


def describe_topics(data, model_result, path):
    words = list(data.columns)
    pdf = matplotlib.backends.backend_pdf.PdfPages(path + "topics_description.pdf")

    j=0
    for topic in np.array(model_result.components_):
        word = []
        freq = []
        [word.append(words[i]) for i in (-topic).argsort()[0:10]]
        [freq.append(topic[i]) for i in (-topic).argsort()[0:10]]
        plt.bar(word, freq)
        plt.title("Top 10 words of topic #%d" % j)
        plt.xticks(rotation = 20)
        j+=1
        pdf.savefig()
        plt.close()
    return pdf.close()

def wordmap_topic(data, model_result, path):
    words = list(data.columns)
    pdf = matplotlib.backends.backend_pdf.PdfPages(path + "topics_description_wordmap.pdf")

    j=0
    for topic in np.array(model_result.components_):
        word = []
        freq = []
        [word.append(words[i]) for i in (-topic).argsort()]
        [freq.append(topic[i]) for i in (-topic).argsort()]
        word_freq = dict(zip(word, freq))
        wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue', collocations=False)
        wordcloud.fit_words(word_freq)        
        plt.imshow(wordcloud)
        plt.title("Words of topic #%d" % j)
        plt.axis('off')
        j+=1
        pdf.savefig()
        plt.close()
    return pdf.close()

def describe_topics_cluster(data, data_count, model_result, path):
    data_count['groups'] = model_result.fit_predict(data)
    data_agg = data_count.groupby('groups').agg(np.mean)
    words = list(data.columns)
    pdf = matplotlib.backends.backend_pdf.PdfPages(path + "topics_description.pdf")

    j=0
    for topic in np.array(data_agg):
        word = []
        freq = []
        [word.append(words[i]) for i in (-topic).argsort()[0:10]]
        [freq.append(topic[i]) for i in (-topic).argsort()[0:10]]
        plt.bar(word, freq)
        plt.title("Top 10 words of topic #%d" % j)
        plt.xticks(rotation = 20)
        j+=1
        pdf.savefig()
        plt.close()
    return pdf.close()

def wordmap_topic_cluster(data, data_count, model_result, path):
    data_count['groups'] = model_result.fit_predict(data)
    data_agg = data_count.groupby('groups').agg(np.mean)
    words = list(data.columns)
    pdf = matplotlib.backends.backend_pdf.PdfPages(path + "topics_description_wordmap.pdf")

    j=0
    for topic in np.array(data_agg):
        word = []
        freq = []
        [word.append(words[i]) for i in (-topic).argsort()]
        [freq.append(topic[i]) for i in (-topic).argsort()]
        word_freq = dict(zip(word, freq))
        wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue', collocations=False)
        wordcloud.fit_words(word_freq)
        plt.imshow(wordcloud)
        plt.title("Words of topic #%d" % j)
        plt.axis('off')
        j+=1
        pdf.savefig()
        plt.close()
    return pdf.close()
