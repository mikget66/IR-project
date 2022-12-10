import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from natsort import natsorted


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
