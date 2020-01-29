import numpy as np
import json

from flair.embeddings import ELMoEmbeddings
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence
from aquiladb import AquilaClient as acl
from nltk import sent_tokenize

from helpers import parse_epub_content

embeddings = ELMoEmbeddings('pubmed')
glove_embedding = WordEmbeddings('glove')
flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')

document_embeddings = DocumentPoolEmbeddings([glove_embedding, flair_embedding_forward, flair_embedding_backward, embeddings])

db = acl('localhost', 50051)

doc = parse_epub_content('/home/developer/PycharmProjects/medindex_semantic_search_prototype/phys_train.epub')

paragraphs = []

for section in doc['sections']:
    paragraphs.extend(section['paragraphs'])


for paragraph in paragraphs[:20]:
    for s in sent_tokenize(paragraph):
        sentence = Sentence(s)
        document_embeddings.embed(sentence)
        embs = sentence.get_embedding()
        sample = db.convertDocument(embs, {"text": s})
        db.addDocuments([sample])


query = 'Which are the key exclusion criteria for patients with subacute phase of ischaemic or haemorrhagic stroke'
sentence = Sentence(query)
document_embeddings.embed(sentence)
query_embs = sentence.get_embedding()
query_vec = db.convertMatrix(query_embs)

k = 5
result = db.getNearest(query_vec, k)

r = json.loads(result.documents.decode('utf-8'))
r

text = ' '.join([t['doc']['text'] for t in r])
text