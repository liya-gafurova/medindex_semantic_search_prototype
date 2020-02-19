import argparse
import collections
import json
import os
import random
import time

from aquiladb import AquilaClient as acl
from nltk import  sent_tokenize, word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from helpers import parse_epub_content
import logging
import collections
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gensim
db = acl('localhost', 50051)
DOC_NAME = 'phys_train.epub'

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False) # if false -- search mode / if True -- train model and insert into DB (!!! delete dir manually before new train first)
    parser.add_argument('--articles_dir', default='/home/lgafurova/Documents/projects/medicine/medindex_semantic_search_prototype/texts/')
    parser.add_argument('--multiple_files', default=False)
    parser.add_argument('--gensim_model_name', default= 'test_1article' )
    parser.add_argument('--query', default='Which are the key exclusion criteria for patients with subacute phase of ischaemic or haemorrhagic stroke')

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
        doc = parse_epub_content(namespase.articles_dir+DOC_NAME)
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
        model_pretrained = train(paragraphs, namespase.gensim_model_name)
        model_pretrained.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        # model_pretrained = Doc2Vec.load(fname)

        for paragraph in paragraphs:
            # for s in sent_tokenize(paragraph):
            print('par = {}'.format(paragraph))
            words = []
            for sent in sent_tokenize(paragraph):
                words.extend(word_tokenize(sent))
            embs = model_pretrained.infer_vector(words)
            sample = db.convertDocument(embs, {"text": paragraph})
            db.addDocuments([sample])

    else:

        model_pretrained = Doc2Vec.load(namespase.gensim_model_name)
        em = model_pretrained.infer_vector(word_tokenize(namespase.query))

        query_vec = db.convertMatrix(em)

        k = 5
        result = db.getNearest(query_vec, k)

        r = json.loads(result.documents.decode('utf-8'))

        text = ' \n'.join([t['doc']['text'] for t in r])
        print('text = {}'.format(text))


    print('name')