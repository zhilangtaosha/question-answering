import os
import json
from tqdm import tqdm
import time
import numpy
import logging
logging.disable(logging.WARNING)
from haystack import Finder
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever
from params import *
from common.item_qa import ItemQA
from common.utils import save_json


def get_retriever():
    document_store = ElasticsearchDocumentStore(host=ES_HOST, port=ES_PORT,
                                                username="", password="", index=ES_INDEX_NAME)
    retriever = ElasticsearchRetriever(document_store=document_store)
    return retriever


def get_reader(reader_path, use_gpu=True):
    use_gpu = 0 if use_gpu else -1
    reader = TransformersReader(model=reader_path, tokenizer=reader_path, use_gpu=use_gpu)
    return reader


def get_output_filename(reader_path, data_path):
    retriever = 'bm25'
    reader = os.path.basename(reader_path)
    data = os.path.basename(data_path).replace('.json', '')
    return f'output_{retriever}_{reader}_{data}.json'


def main():
    retriever = get_retriever()
    for reader_path in READERS:
        reader = get_reader(reader_path, use_gpu=True)
        finder = Finder(reader, retriever)
        for data_path in DATASETS:
            output_filename = get_output_filename(reader_path, data_path)
            print(f'---------- {output_filename} ----------')
            with open(data_path, 'r', encoding='utf8') as f:
                output_qas = []
                time_qas = []
                data = json.load(f)
                print(f'{len(data)} QA entries loaded')
                count = 0
                for qa in tqdm(data):
                    start_time = time.time()
                    prediction = finder.get_answers(question=qa['question'],
                                                    top_k_retriever=RETRIEVER_TOP_K,
                                                    top_k_reader=READER_TOP_K)
                    time_diff = time.time() - start_time
                    time_qas.append(time_diff)
                    item = ItemQA(qa['question_id'], qa['question'])
                    answer = prediction['answers'][0]['answer'] if len(prediction['answers']) > 0 else ''
                    item.add_answer(answer)
                    output_qas.append(item.json())
                    count += 1
                    if count > 5: break

                save_json(output_qas, output_filename)
                print(f'Output is saved to {output_filename}')
                print(f'Average time per question: {round(sum(time_qas)/len(time_qas), 2)}s')
                print(f'Max time per question: {round(max(time_qas), 2)}s')
                print(f'Min time per question: {round(min(time_qas), 2)}s')
                print(f'Standard Deviation time per question: {round(numpy.std(time_qas), 2)}s')


if __name__ == '__main__':
    main()
