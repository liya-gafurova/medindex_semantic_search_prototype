import spacy
import wmd

from flair.embeddings import ELMoEmbeddings
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings,  StackedEmbeddings, Sentence
from aquiladb import AquilaClient as acl
from nltk import sent_tokenize
from wmd import WMD

from helpers import parse_epub_content

embeddings = ELMoEmbeddings('pubmed')

glove_embedding = WordEmbeddings('glove')
flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')

document_embeddings = DocumentPoolEmbeddings([embeddings])
stack_emds = StackedEmbeddings([embeddings])

query = 'Which are the key exclusion criteria for patients with subacute phase of ischaemic or haemorrhagic stroke'
sentence = Sentence(query)
glove_embedding.embed(sentence)
for token in sentence:
    print(token)
    print(token.embedding)
