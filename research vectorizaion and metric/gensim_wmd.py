import argparse

import os
import random
from scipy.spatial.distance import cosine
from nltk import sent_tokenize, word_tokenize, download
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from helpers import parse_epub_content
import logging
import collections
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gensim
from gensim.similarities import WmdSimilarity
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import download
download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')
download('punkt')  # Download data for tokenizer.

DOC_NAME = 'phys_train.epub'
q1 = 'Which are the key exclusion criteria for patients with subacute phase of ischaemic or haemorrhagic stroke'
DOC_NAME_613 = '6132876.epub'
q2 = 'How should Treatment response be monitored?'
DOC_NAME_CLINICAL  ='Clinical_practice_guidelines_in_Wilson_disease_1.epub'
q3 = 'Which components are included in the Leipzig score used to establish a diagnosis of WD?"'
DOC_NAME_STANDARTS = 'Standards_for_the_diagnosis_and_management_of_complex_regional_pain_syndrome.epub'
q4 = 'Which initial diagnostic criteria should be used to establish a diagnosis of complex regional pain syndrome?'
q5 = 'What is recommended next step for those patients who do not have clearly reducing pain and improving function within 2 months of commencing treatment, despite good patient engagement in rehabilitation?'
DOC_DIARRHEA = 'Travelers_diarrhea.epub'
q6 = 'What is the recommended first-line agent for mild TD?'

NUM_BEST = 5


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False) # if false -- search mode / if True -- train model and insert into DB (!!! delete dir manually before new train first)
    parser.add_argument('--articles_dir')
    parser.add_argument('--multiple_files', default=False)
    parser.add_argument('--gensim_model_name', default= '../gensim_models/'+ DOC_DIARRHEA+'_model' )
    parser.add_argument('--query', default=q2)
    parser.add_argument('--doc_name' , default=DOC_DIARRHEA)
    return  parser

def train(paragraphs_list, fname):
    def read_par_list(paragraphs_list, tokens_only= False):

        for i ,line in enumerate(paragraphs_list):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    train_corpus = list(read_par_list(paragraphs_list))
    test_corpus = list(read_par_list(paragraphs_list, tokens_only=True))

    model = gensim.models.doc2vec.Doc2Vec(vector_size=7268, min_count=2, epochs=200)
    model.build_vocab(train_corpus)

    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    # vector = model.infer_vector(word_tokenize('The Barthel index measures activities of daily living based on 10 items, with scores ranging from 0 to 100 points—higher scores indicating less dependence.'))
    # print('len vector = {}'.format(len(vector)))

    ranks = []
    second_ranks = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

        second_ranks.append(sims[1])
    counter = collections.Counter(ranks)
    print(counter)

    # Pick a random document from the test corpus and infer a vector from the model
    doc_id = random.randint(0, len(test_corpus) - 1)
    inferred_vector = model.infer_vector(test_corpus[doc_id])
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    # Compare and print the most/median/least similar documents from the train corpus
    print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))


    model.save(os.path.join(fname))
    return model

def preprocess(doc):
    doc = doc.lower()  # Lower the text.
    doc = word_tokenize(doc)  # Split into words.
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
    return doc

if __name__ == '__main__':
    parser = createParser()
    namespase = parser.parse_args()
    documents = []
    if not namespase.multiple_files:
        doc = parse_epub_content(namespase.articles_dir+namespase.doc_name)
        documents = [doc]
    else :
        for root, dirs, file in os.walk(namespase.articles_dir):
            doc = parse_epub_content(file)
            documents.append(doc)


    paragraphs = []
    for doc in documents:
        for section in doc['sections']:
            paragraphs.extend(section['paragraphs'])


    if namespase.train:
        train(paragraphs, namespase.gensim_model_name)


    else:
        model_pretrained = Doc2Vec.load(namespase.gensim_model_name)


        paragraphs2 = []
        paragraphs2 = list(map(preprocess, paragraphs))

        # WMD

        instance = WmdSimilarity(paragraphs2, model_pretrained, num_best=len(paragraphs))
        sent = namespase.query
        query = preprocess(sent)

        sims = instance[query]  # A query is simply a "look-up" in the similarity class.

        # Print the query and the retrieved documents, together with their similarities.
        print ('Query:')
        print (sent)
        best = []
        all_scored = []
        count = 0
        for i in range(len(sims)):
            if count <= NUM_BEST:
                best.append(paragraphs[sims[i][0]])
                count += 1
            all_scored.append(paragraphs[sims[i][0]])

        print ('\t '.join(best))

        print('\n\n')
        print('\n '.join(all_scored))
    print('final')