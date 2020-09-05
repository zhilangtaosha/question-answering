import os, sys, time
sys.path.insert(1, os.path.join('..', 'common'))
import numpy as np
from tqdm import tqdm
from item_qa import ItemQA
from utils import *
from params import *



def get_output_filename(reader_path, data_path):
    retriever = 'BM25'
    reader = os.path.basename(reader_path)
    data = os.path.basename(data_path).replace('.json', '')
    return f'qa_{retriever}_DPR_{reader}_{data}.json'


def predict_answer(question, retriever_es, retriever_dpr, reader, use_gpu):
    start_time = time.time()
    docs = retriever_es.retrieve(question, top_k=RETRIEVER_ES_TOP_K)
    answer = ''
    if docs:
        q_vecs = retriever_dpr.embed_queries([question])
        passages = [d.text for d in docs]
        p_vecs = retriever_dpr.embed_passages(passages)
        faiss_index = get_faiss_ip_index(d=FAISS_INDEX_DIMENSION, use_gpu=use_gpu)
        faiss_index.add(np.array(p_vecs))
        D, I = faiss_index.search(np.array(q_vecs), RETRIEVER_DPR_TOP_K)
        candidate_docs = [docs[i] for i in I[0]]
        prediction = reader.predict(question=question, documents=candidate_docs, top_k=1)
        answer = prediction['answers'][0]['answer']

    time_diff = time.time() - start_time
    return answer, time_diff


def main():
    logger = get_logger('run_ES_DPR_Reader-logger', 'run_ES_DPR_Reader.log')
    logger.info('---------------------------------------------------------------')
    logger.info('Parameters:')
    logger.info(f'RETRIEVER_ES_TOP_K = {RETRIEVER_ES_TOP_K}')
    logger.info(f'RETRIEVER_DPR_TOP_K = {RETRIEVER_DPR_TOP_K}')
    logger.info(f'READER_TOP_K = {READER_TOP_K}')
    logger.info(f'FAISS_INDEX_DIMENSION = {FAISS_INDEX_DIMENSION}')
    logger.info(f'USE_GPU = {USE_GPU}')
    logger.info('---------------------------------------------------------------')
    logger.info(f'connecting to ElasticSearch {ES_HOST}:{ES_PORT}@{ES_INDEX_NAME}...')
    document_store = get_elastic_search_document_store(ES_HOST, ES_PORT, ES_INDEX_NAME)
    logger.info('loading ES retriever...')
    retriever_es = get_elastic_search_retriever(document_store)
    logger.info('loading DPR retriever...')
    retriever_dpr = get_dense_passage_retriever(document_store=document_store,
                                                dpr_model_path=DPR_MODEL_PATH,
                                                use_gpu=USE_GPU, batch_size=16, do_lower_case=True)
    for reader_path in READERS:
        logger.info(f'loading reader at {reader_path} ...')
        reader = get_neural_reader(reader_path, use_gpu=USE_GPU)
        for data_path in DATASETS:
            output_filename = get_output_filename(reader_path, data_path)
            logger.info(f'output filename: {output_filename}')
            with open(data_path, 'r', encoding='utf8') as f:
                output_qas = []
                time_qas = []
                data = json.load(f)
                logger.info(f'{len(data)} QA entries loaded')
                count = 0
                for qa in tqdm(data):
                    question = qa['question']
                    answer, time_diff = predict_answer(question, retriever_es, retriever_dpr, reader, USE_GPU)
                    item = ItemQA(qa['question_id'], qa['question'])
                    item.add_answer(answer)
                    output_qas.append(item.json())
                    time_qas.append(time_diff)
                    count += 1

                save_json(output_qas, output_filename)
                logger.info(f'Output is saved to {output_filename}')
                logger.info(f'Average time per question: {round(sum(time_qas)/len(time_qas), 2)}s')
                logger.info(f'Max time per question: {round(max(time_qas), 2)}s')
                logger.info(f'Min time per question: {round(min(time_qas), 2)}s')
                logger.info(f'Standard Deviation time per question: {round(np.std(time_qas), 2)}s')


if __name__ == '__main__':
    main()
