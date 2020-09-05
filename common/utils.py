import json
import faiss
import logging
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.retriever.dense import DensePassageRetriever
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers


def save_json(obj_to_save, output_filename):
    with open(output_filename, 'w', encoding='utf8') as out:
        json.dump(obj_to_save, out, ensure_ascii=False)


def get_faiss_ip_index(d=768, use_gpu=True):
    # build a Inner Product (CPU) index
    index_cpu = faiss.IndexFlatIP(d)
    if use_gpu:
        # claim single GPU resource
        resource = faiss.StandardGpuResources()
        # make it into a gpu index
        index_gpu = faiss.index_cpu_to_gpu(resource, 0, index_cpu)
        return index_gpu
    return index_cpu


def get_elastic_search_document_store(es_host, es_port, es_index_name):
    return ElasticsearchDocumentStore(host=es_host, port=es_port,
                                      username="", password="", index=es_index_name)


def get_elastic_search_retriever(document_store):
    return ElasticsearchRetriever(document_store=document_store)


def get_dense_passage_retriever(document_store,
                                dpr_model_path,
                                use_gpu=True,
                                batch_size=16,
                                do_lower_case=True):
    return DensePassageRetriever(document_store=document_store,
                                 embedding_model=dpr_model_path,
                                 use_gpu=use_gpu,
                                 batch_size=batch_size,
                                 do_lower_case=do_lower_case)


def get_neural_reader(reader_path, use_gpu=True):
    use_gpu = 0 if use_gpu else -1
    return TransformersReader(model=reader_path, tokenizer=reader_path, use_gpu=use_gpu)


def get_logger(logger_name, log_file_path):
    # create logger with 'spam_application'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger