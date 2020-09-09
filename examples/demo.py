import os
import sys
import numpy as np
from haystack import Finder
from IPython.display import display, Markdown

sys.path.insert(1, os.path.join('..', 'common'))
from utils import *
from eval import *

logging.disable(logging.WARNING)

HOST = 'localhost'
PORT = 9200
INDEX_NAME = 'wikipedia_paragraph'
READER_DiR = '../models/electra-base-squad2'
DPR_MODEL_PATH = '../models/dpr/multi_hf_bert_base.cp'
USE_GPU = True
GPU_INDEX = 0 if USE_GPU else -1

document_store = ElasticsearchDocumentStore(host=HOST, port=PORT, username="", password="", index=INDEX_NAME)
retriever_es = ElasticsearchRetriever(document_store=document_store)
retriever_dpr = DensePassageRetriever(document_store=document_store,
                                      embedding_model=DPR_MODEL_PATH,
                                      use_gpu=USE_GPU,
                                      batch_size=16,
                                      do_lower_case=True)
reader = TransformersReader(model=READER_DiR, tokenizer=READER_DiR, use_gpu=GPU_INDEX)
faiss_index = get_faiss_ip_index(d=768, use_gpu=USE_GPU)
finder = Finder(reader, retriever_es)


def print_demo_answers(prediction):
    s = ''
    answers = prediction['answers']
    for i, a in enumerate(answers):
        rank = i + 1
        s += f'##### Answer {rank}: {a["answer"]}\n'
        s += f'* Context: \"{a["context"]}\"\n'
        s += f'* Article: {a["meta"]["name"]}\n'
        s += f'* URL: {a["meta"]["url"]}\n\n'
    display(Markdown(s))


def answer(question, mode, retriever_es_k=100, retriever_dpr_k=10, reader_k=3):
    if mode == 'bm25+reader':
        prediction = finder.get_answers(question=question,
                                        top_k_retriever=retriever_es_k,
                                        top_k_reader=reader_k)
        print_demo_answers(prediction)

    elif mode == 'bm25+dpr+reader':
        docs = retriever_es.retrieve(question, top_k=retriever_es_k)
        q_vecs = retriever_dpr.embed_queries([question])
        es_doc_texts = [normalize_text(d.text) for d in docs]
        p_vecs = retriever_dpr.embed_passages(es_doc_texts)
        faiss_index.add(np.array(p_vecs))
        D, I = faiss_index.search(np.array(q_vecs), retriever_es_k)
        dpr_docs = [docs[I[0][i]] for i in range(retriever_dpr_k)]
        prediction = reader.predict(question=question, documents=dpr_docs, top_k=reader_k)
        faiss_index.reset()
        print_demo_answers(prediction)
