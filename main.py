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

for document in documents_of_terms:

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

print('write your query')
query = str(input())

"""query = 'antony caeser'
"""
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
# print(term_freq)

for i in range(1, len(documents)):
    term_freq[i] = get_term_freq(documents[i]).values()

term_freq.columns = ['doc' + str(i) for i in range(1, 11)]
print(" ")
print(" ")
print(" ")
print(" ")
print("____________________________tf_______________________________________")
print(term_freq)


def get_weighted_term_freq(x):
    if x > 0:
        return math.log(x) + 1
    return 0


for i in range(1, len(documents) + 1):
    term_freq['doc' + str(i)] = term_freq['doc' + str(i)].apply(get_weighted_term_freq)
print('')
print('')
print('')
print("____________________________tfWeight_______________________________________")
print(term_freq)

tfd = pd.DataFrame(columns=['freq', 'idf'])

for i in range(len(term_freq)):
    frequency = term_freq.iloc[i].values.sum()

    tfd.loc[i, 'freq'] = frequency

    tfd.loc[i, 'idf'] = math.log10(float(10 / frequency))

tfd.index = term_freq.index

print(tfd)

term_freq_inve_doc_frq = term_freq.multiply(tfd['idf'], axis=0)

print('')
print("____________________________TF-IDF_______________________________________")
print(term_freq_inve_doc_frq)

# -------------document length------------

document_length = pd.DataFrame()


def get_docs_length(col):
    return np.sqrt(term_freq_inve_doc_frq[col].apply(lambda x: x ** 2).sum())


for column in term_freq_inve_doc_frq.columns:
    document_length.loc[0, column + '_len'] = get_docs_length(column)

print(document_length)
print('')
print("____________________________normalized TF-IDF_______________________________________")
normalized_term_freq_idf = pd.DataFrame()


def get_normalized(col, x):
    try:
        return x / document_length[col + '_len'].values[0]
    except:
        return 0


for column in term_freq_inve_doc_frq.columns:
    normalized_term_freq_idf[column] = term_freq_inve_doc_frq[column].apply(lambda x: get_normalized(column, x))

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

print('write your query')
query2 = str(input())


q = [query2]

q_vector = (vector.transform(q).toarray().reshape(df.shape[0]))

similarity = {}

for i in range(10):
    similarity[i] = np.dot(df.loc[:, i].values, q_vector) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vector)

similarity_sorted = sorted(similarity.items(), key=lambda x: x[1])
for docu, score in similarity_sorted:
    print('similarity score = ', score)
    print('doc is: ', docu + 1)


def get_w_tf(x):
    try:
        return math.log10(x) + 1
    except:
        return 0


final_set = pd.DataFrame(index=normalized_term_freq_idf.index)
final_set['tf'] = [1 if x in query2.split() else 0 for x in (normalized_term_freq_idf.index)]
final_set['w_tf'] = final_set['tf'].apply(lambda x: get_w_tf(x))
product = normalized_term_freq_idf.multiply(final_set['w_tf'], axis=0)
final_set['idf'] = tfd['idf'] * final_set['w_tf']
final_set['tf_idf'] = final_set['w_tf'] * final_set['idf']
final_set['norm'] = 0
for i in range (len(final_set)):
    final_set['norm'].iloc[i] = float(final_set['idf'].iloc[i]/ math.sqrt(sum(final_set['idf'].values**2)))
product2 = product.multiply(final_set['norm'], axis=0)
print('')
print('')
print('')
print('')
print(final_set)

scores = {}
for col in product2.columns:
    if 0 in product2[col].loc[query2.split()].values:
        pass
    else:
        scores[col] = product2[col].sum()


print('cosine similarity')
print(scores)

Q_l = math.sqrt(sum([x**2 for x in final_set['idf'].loc[query2.split()]]))
print('query length ')
print(Q_l)

prd_res = product2[(scores.keys())].loc[query2.split()]
print('-------product (query*match docs)--------')
print(prd_res)

print('-------product summation (query*match docs)--------')
print(prd_res.sum())
