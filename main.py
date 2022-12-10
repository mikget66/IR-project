import math
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from natsort import natsorted
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = stopwords.words('english')
stop_words.remove('in')
stop_words.remove('to')
stop_words.remove('where')

file_name = natsorted(os.listdir('files'))

documents_of_terms = []
for file in file_name:

    with open(f'files/{file}', 'r') as f:
        document = f.read()
    tokenized_documents = word_tokenize(document)
    terms = []
    for word in tokenized_documents:
        if word not in stop_words:
            terms.append(word)
    documents_of_terms.append(terms)

print(documents_of_terms)

documents = [['antony', 'brutus', 'caeser', 'cleopatra', 'mercy', 'worser'],
             ['antony', 'brutus', 'caeser', 'calpurnia'],
             ['mercy', 'worser'],
             ['brutus', 'caeser', 'mercy', 'worser'],
             ['caeser', 'mercy', 'worser'],
             ['antony', 'caeser', 'mercy'],
             ['angels', 'fools', 'fear', 'in', 'rush', 'to', 'tread', 'where'],
             ['angels', 'fools', 'fear', 'in', 'rush', 'to', 'tread', 'where'],
             ['angels', 'fools', 'in', 'rush', 'to', 'tread', 'where'],
             ['fools', 'fear', 'in', 'rush', 'to', 'tread', 'where']]

document_number = 1
positional_index = {}

for document in documents:

    for positional, term in enumerate(document):

        if term in positional_index:

            positional_index[term][0] = positional_index[term][0] + 1

            if document_number in positional_index[term][1]:
                positional_index[term][1][document_number].append(positional)

            else:
                positional_index[term][1][document_number] = [positional]

        else:
            positional_index[term] = []
            positional_index[term].append(1)
            positional_index[term].append({})
            positional_index[term][1][document_number] = [positional]

    document_number += 1

print(positional_index)

"""
print('write your query')
query = str(input())"""

query = 'antony caeser'

final_list = [[] for i in range(10)]

for word in query.split():
    for key in positional_index[word][1].keys():

        if final_list[key - 1] != []:
            if final_list[key - 1][-1] == positional_index[word][1][key][0] - 1:
                final_list[key - 1].append(positional_index[word][1][key][0])
        else:
            final_list[key - 1].append(positional_index[word][1][key][0])

print(" ")
print(" ")
print(" ")
print(" ")
print('the Query output is')
print(final_list)
for position, list in enumerate(final_list, start=1):
    if len(list) == len(query.split()):
        print(position)


all_words = []
for doc in documents:
    for word in doc:
        all_words.append(word)


def get_term_freq(doc):
    words_found = dict.fromkeys(all_words, 0)
    for word in doc:
        words_found[word] += 1
    return words_found


term_freq = pd.DataFrame(get_term_freq(documents[0]).values(), index=get_term_freq(documents[0]).keys())
#print(term_freq)

for i in range(1, len(documents)):
    term_freq[i] = get_term_freq(documents[i]).values()

term_freq.columns = ['doc' + str(i) for i in range(1, 11)]
print(" ")
print(" ")
print(" ")
print(" ")
print('terms frequency')
print(term_freq)


def get_weighted_term_freq(x):
    if x > 0:
        return math.log(x) + 1
    return 0


for i in range(1, len(documents)+1):
    term_freq['doc' + str(i)] = term_freq['doc' + str(i)].apply(get_weighted_term_freq)

print(term_freq)

tfd = pd.DataFrame(columns=['freq', 'idf'])

for i in range(len(term_freq)):
    frequency = term_freq.iloc[i].values.sum()

    tfd.loc[i, 'freq'] = frequency

    tfd.loc[i, 'idf'] = math.log10(float(10 / frequency))

tfd.index = term_freq.index
print(tfd)

term_freq_inve_doc_frq = term_freq.multiply(tfd['idf'], axis=0)

print(term_freq_inve_doc_frq)

#-------------document linght------------

document_length = pd.DataFrame()

def get_docs_length(col):
    return np.sqrt(term_freq_inve_doc_frq[col].apply(lambda x: x**2).sum())

for column in term_freq_inve_doc_frq.columns:
    document_length.loc[0, column+'_len'] = get_docs_length(column)

print(document_length)

normalized_term_freq_idf = pd.DataFrame()

def get_normalized(col, x):
    try:
        return x / document_length[col+'_len'].values[0]
    except:
        return 0

for column in term_freq_inve_doc_frq.columns:
    normalized_term_freq_idf[column] = term_freq_inve_doc_frq[column].apply(lambda x : get_normalized(column, x))



print(normalized_term_freq_idf)

documents = ['antony brutus caeser cleopatra mercy worser',
             'antony brutus caeser calpurnia',
             'mercy worser',
             'brutus caeser mercy worser',
             'caeser mercy worser',
             'antony caeser mercy',
             'angels fools fear in rush to tread where',
             'angels fools fear in rush to tread where',
             'angels fools in rush to tread where',
             'fools fear in rush to tread where']

vector = TfidfVectorizer()

x = vector.fit_transform(documents)
x = x.T.toarray()
df = pd.DataFrame(x, index=vector.get_feature_names_out())

query2 ='antony brutus'

q = [query2]

q_vector = (vector.transform(q).toarray().reshape(df.shape[0]))

similarity = {}

for i in range(10):
    similarity[i] = np.dot(df.loc[:, i].values, q_vector) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vector)

similarity_sorted = sorted(similarity.items(), key=lambda x: x[1])
for docu, score in similarity_sorted:
        print('similarity score = ',score)
        print('doc is: ', docu+1)

