"""
This script has the sole purpose of taking all the documents from ElasticSearch
and encode them with DPR encoder into vectors, then store these vectors into a file.
"""

import os, sys, time
import numpy as np
import json
import elasticsearch
from elasticsearch.helpers import scan
from tqdm import tqdm

sys.path.insert(1, os.path.join('..', 'common'))
from utils import *
from params import *
from eval import *


class VecsWriter:
    def __init__(self, limit=10, output_dir='', output_file_prefix='vecs_'):
        self.vecs = []
        self.count = 0
        self.limit = limit
        self.output_dir = output_dir
        self.output_file_prefix = output_file_prefix

    def save_vecs(self):
        if self.vecs:
            output_file_path = f'{self.output_file_prefix}{str(self.count)}.npy'
            output_file_path = os.path.join(self.output_dir, output_file_path)
            np.save(output_file_path, self.vecs)
            self.vecs = []
            self.count += 1

    def add_a_vec(self, vec):
        self.vecs.append(vec)
        if len(self.vecs) >= self.limit:
            self.save_vecs()


def main():
    logger = get_logger('run_DPR_Encoder', 'run_DPR_Encoder.log')
    count = 0
    batch_size = 64
    docs = []
    vw = VecsWriter(249)
    retriever_dpr = get_dense_passage_retriever(document_store=None,
                                                dpr_model_path=DPR_MODEL_PATH,
                                                use_gpu=USE_GPU, batch_size=batch_size, do_lower_case=True)
    try:
        es = elasticsearch.Elasticsearch(f'http://{ES_HOST}:{ES_PORT}')
        es_response = scan(
            es,
            index=ES_INDEX_NAME,
            query={"query": {"match_all": {}}}
        )
        for item in tqdm(es_response, total=6005733):
            if count < 1000:
                count += 1
                text = normalize_text(item['_source']['text'])
                docs.append(text)
                if len(docs) == batch_size:
                    vecs = retriever_dpr.embed_passages(docs)
                    for vec in vecs:
                        vw.add_a_vec(vec)
                    del vecs
                    docs = []
            else:
                break

        # encode remaining docs
        if docs:
            vecs = retriever_dpr.embed_passages(docs)
            for vec in vecs:
                vw.add_a_vec(vec)
            del vecs

        # save remaining docs
        vw.save_vecs()
    except (Exception, KeyboardInterrupt) as e:
        logger.info(e)
        logger.info(f'An error occurred at Count {count}...')


if __name__ == '__main__':
    main()

    # v0 = np.load('vecs_0.npy')
    # print(len(v0))
    # v1 = np.load('vecs_1.npy')
    # print(len(v1))
    # v2 = np.load('vecs_2.npy')
    # print(len(v2))
    # v3 = np.load('vecs_3.npy')
    # print(len(v3))
    # print(len(v3[0]))