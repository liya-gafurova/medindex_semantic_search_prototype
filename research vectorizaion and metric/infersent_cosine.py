import argparse
import logging
import os

import torch
from scipy.spatial.distance import cosine
from helpers import parse_epub_content
from models import InferSent
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# before use download models, follow instructions https://github.com/facebookresearch/InferSent
# also download file models.py from https://github.com/facebookresearch/InferSent repository

model_version = 2
MODEL_PATH = "./encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
W2V_PATH = './GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'

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
    def find_distance(self,query_embed):
        # scipy...cosine  0 - параллельны, 1 - перпендикулярны
        self.dist  = cosine(self.embedding, query_embed)

def sort_distance(list_em_paragraphs : list ):
    dist_dict = {}
    for em_par in list_em_paragraphs:
        dist_dict[em_par.paragraph] = em_par.dist
    return { par: dist for par, dist in sorted(dist_dict.items(), key= lambda item: item[1] ) }



def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False) # if false -- search mode / if True -- train model and insert into DB (!!! delete dir manually before new train first)
    parser.add_argument('--articles_dir')
    parser.add_argument('--multiple_files', default=False)
    parser.add_argument('--gensim_model_name', default= '../gensim_models/'+ DOC_NAME+'_model' )
    parser.add_argument('--query', default=q1)
    parser.add_argument('--doc_name' , default=DOC_NAME)
    return  parser



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


    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    use_cuda = False
    model = model.cuda() if use_cuda else model
    model.set_w2v_path(W2V_PATH)
    # Load embeddings of K most frequent words
    model.build_vocab_k_words(K=100000)

    paragraphs = []
    for doc in documents:
        for section in doc['sections']:
            paragraphs.extend(section['paragraphs'])

    embeddings = model.encode(paragraphs, bsize=128, tokenize=False, verbose=True)
    query_embed = model.encode([namespase.query], bsize=128, tokenize=False, verbose=True)[0]
    em_par_list = []
    for embed, par in zip(embeddings, paragraphs):
        em_par = EmbededParagraph(par, embed)
        em_par.find_distance(query_embed)
        em_par_list.append(em_par)


    sorted_list = sort_distance(em_par_list)
    pars = list(sorted_list.keys())
    print('\n '.join(pars))
    print('infersent_cosine')