import argparse
import logging
import os
from flair.data import Sentence
from scipy.spatial.distance import cosine

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from flair.embeddings import ELMoEmbeddings
from flair.embeddings import FlairEmbeddings, DocumentPoolEmbeddings

from helpers import parse_epub_content



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

def make_embed(document: str):
    document_sentence = Sentence(document)
    document_embeddings.embed(document_sentence)
    return  document_sentence.get_embedding().detach().numpy()

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False)
    parser.add_argument('--articles_dir')
    parser.add_argument('--multiple_files', default=False)
    parser.add_argument('--gensim_model_name', default= DOC_NAME_613+'_model' )
    parser.add_argument('--query', default=q2)
    parser.add_argument('--doc_name' , default=DOC_NAME_613)
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


    paragraphs = []
    for doc in documents:
        for section in doc['sections']:
            paragraphs.extend(section['paragraphs'])

    elmo_embedding = ELMoEmbeddings('pubmed')
    flair_embedding_forward = FlairEmbeddings('news-forward')
    flair_embedding_backward = FlairEmbeddings('news-backward')

    document_embeddings = DocumentPoolEmbeddings(
        [
            flair_embedding_forward,
            flair_embedding_backward,
            elmo_embedding,

        ],
        pooling='max',
        fine_tune_mode='none'
    )

    query_embed = make_embed(namespase.query)
    em_pars = []
    for paragraph in paragraphs:

        embs = make_embed(paragraph)
        em_par = EmbededParagraph(paragraph, embs)
        em_par.find_distance(query_embed)
        em_pars.append(em_par)

    sorted_list = sort_distance(em_pars)
    pars = list(sorted_list.keys())
    print('\n '.join(pars))

    print('my_metric')