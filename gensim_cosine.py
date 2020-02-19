import argparse

import os
import random
from scipy.spatial.distance import cosine
from nltk import  sent_tokenize, word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from helpers import parse_epub_content
import logging
import collections
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gensim
import gensim.downloader as api
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


class EmbededParagraph:
    def __init__(self, paragraph : str , embed  ):
        self.paragraph = paragraph
        self.embedding = embed
    def find_distance(self,query_embed, metric  = 'scipy_cosine', model = None ):
        # scipy...cosine  0 - параллельны, 1 - перпендикулярны
        if metric == 'scipy_cosine':
            self.dist  = cosine(self.embedding, query_embed)
        else:
            raise Exception('not implemented metric')

def sort_distance(list_em_paragraphs : list ):
    dist_dict = {}
    for em_par in list_em_paragraphs:
        dist_dict[em_par.paragraph] = em_par.dist
    sorted_dict = { par: dist for par, dist in sorted(dist_dict.items(), key= lambda item: item[1] ) }
    return sorted_dict



def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False) # if false -- search mode / if True -- train model and insert into DB (!!! delete dir manually before new train first)
    parser.add_argument('--articles_dir')
    parser.add_argument('--multiple_files', default=False)
    parser.add_argument('--gensim_model_name', default= DOC_NAME+'_model' )
    parser.add_argument('--query', default=q1)
    parser.add_argument('--doc_name' , default=DOC_NAME)
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
        em_query  = model_pretrained.infer_vector(word_tokenize(namespase.query))

        embeded_paragraphs = []
        for paragraph in paragraphs:
            em = model_pretrained.infer_vector(word_tokenize(paragraph))
            em_par = EmbededParagraph(paragraph, em)
            em_par.find_distance(em_query , metric=  'scipy_cosine', model = model_pretrained ) #  'scipy_cosine' 'wmd'
            embeded_paragraphs.append(em_par)

        # sort by em_par.dist
        sorted_list = sort_distance(embeded_paragraphs)
        pars  = list(sorted_list.keys())
        print('###'.join(pars[:6]))


    print('my_metric')