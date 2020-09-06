import os, sys, time
import numpy as np
from tqdm import tqdm
sys.path.insert(1, os.path.join('..', 'common'))
from item_qa import ItemQA
from utils import *
from params import *
from eval import *


def get_output_filename(reader_path, data_path):
    retriever = 'BM25'
    reader = os.path.basename(reader_path)
    data = os.path.basename(data_path).replace('.json', '')
    return f'qa_{retriever}_DPR_{reader}_{data}.json'


def predict_and_evaluate(gold_qa_entry, retriever_es, retriever_dpr, reader, use_gpu):
    question_id = gold_qa_entry['question_id']
    question = gold_qa_entry['question']
    gold_answers = gold_qa_entry['answers']
    pred_answer = ''
    r_acc_es = 0
    r_acc_dpr = 0
    f1 = 0
    em = 0
    t = 0

    start_time = time.time()
    docs = retriever_es.retrieve(question, top_k=RETRIEVER_ES_TOP_K)
    if docs:
        q_vecs = retriever_dpr.embed_queries([question])
        docs_es = [d.text for d in docs]
        p_vecs = retriever_dpr.embed_passages(docs_es)
        faiss_index = get_faiss_ip_index(d=FAISS_INDEX_DIMENSION, use_gpu=use_gpu)
        faiss_index.add(np.array(p_vecs))
        D, I = faiss_index.search(np.array(q_vecs), RETRIEVER_DPR_TOP_K)
        docs_dpr = [docs[i] for i in I[0]]
        prediction = reader.predict(question=question, documents=docs_dpr, top_k=1)
        pred_answers = prediction['answers']
        if pred_answers:
            pred_answer = pred_answers[0]['answer']
        t = time.time() - start_time
        faiss_index.reset()
        docs_dpr = [d.text for d in docs_dpr]
        # eval
        r_acc_es = retrieval_accuracy_max(gold_answers, docs_es)
        r_acc_dpr = retrieval_accuracy_max(gold_answers, docs_dpr)
        f1 = reader_metric_max(f1_score, pred_answer, gold_answers)
        em = reader_metric_max(exact_match_score, pred_answer, gold_answers)

    item = ItemQA(question_id, question,
                  r_acc_es=r_acc_es,
                  r_acc_dpr=r_acc_dpr,
                  f1=f1,
                  em=em,
                  t=t)
    item.add_answer(pred_answer)
    return item


def save_output_items(output_filename: str, output_items: List[ItemQA], logger):
    items_json = [item.json() for item in output_items]
    save_json(items_json, output_filename)
    logger.info(f'Output is saved to {output_filename}')

    r_acc_es = [item.r_acc_es for item in output_items]
    logger.info(f'Avg RetrievalAccuracy @{RETRIEVER_ES_TOP_K} per q: {round(sum(r_acc_es) / len(r_acc_es), 2)}')
    logger.info(f'Max RetrievalAccuracy @{RETRIEVER_ES_TOP_K} per q: {round(max(r_acc_es), 2)}')
    logger.info(f'Min RetrievalAccuracy @{RETRIEVER_ES_TOP_K} per q: {round(min(r_acc_es), 2)}')
    logger.info(f'Std RetrievalAccuracy @{RETRIEVER_ES_TOP_K} per q: {round(np.std(r_acc_es), 2)}')
    r_acc_dpr = [item.r_acc_dpr for item in output_items]
    logger.info(f'Avg RetrievalAccuracy @{RETRIEVER_DPR_TOP_K} per q: {round(sum(r_acc_dpr) / len(r_acc_dpr), 2)}')
    logger.info(f'Max RetrievalAccuracy @{RETRIEVER_DPR_TOP_K} per q: {round(max(r_acc_dpr), 2)}')
    logger.info(f'Min RetrievalAccuracy @{RETRIEVER_DPR_TOP_K} per q: {round(min(r_acc_dpr), 2)}')
    logger.info(f'Std RetrievalAccuracy @{RETRIEVER_DPR_TOP_K} per q: {round(np.std(r_acc_dpr), 2)}')
    f1s = [item.f1 for item in output_items]
    logger.info(f'Avg F1 per q: {round(sum(f1s) / len(f1s), 2)}')
    logger.info(f'Max F1 per q: {round(max(f1s), 2)}')
    logger.info(f'Min F1 per q: {round(min(f1s), 2)}')
    logger.info(f'Std F1 per q: {round(np.std(f1s), 2)}')
    ems = [item.em for item in output_items]
    logger.info(f'Avg EM per q: {round(sum(ems) / len(ems), 2)}')
    logger.info(f'Max EM per q: {round(max(ems), 2)}')
    logger.info(f'Min EM per q: {round(min(ems), 2)}')
    logger.info(f'Std EM per q: {round(np.std(ems), 2)}')
    ts = [item.t for item in output_items]
    logger.info(f'Avg time per q: {round(sum(ts) / len(ts), 2)}s')
    logger.info(f'Max time per q: {round(max(ts), 2)}s')
    logger.info(f'Min time per q: {round(min(ts), 2)}s')
    logger.info(f'Std time per q: {round(np.std(ts), 2)}s')


def main():
    logger = get_logger(__name__, 'run_ES_DPR_Reader.log')
    logger.info('----------------------------')
    logger.info('Parameters:')
    logger.info(f'RETRIEVER_ES_TOP_K = {RETRIEVER_ES_TOP_K}')
    logger.info(f'RETRIEVER_DPR_TOP_K = {RETRIEVER_DPR_TOP_K}')
    logger.info(f'READER_TOP_K = {READER_TOP_K}')
    logger.info(f'FAISS_INDEX_DIMENSION = {FAISS_INDEX_DIMENSION}')
    logger.info(f'USE_GPU = {USE_GPU}')
    logger.info('----------------------------')
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
            output_items = []
            count = 0
            try:
                with open(data_path, 'r', encoding='utf8') as f:
                    data = json.load(f)
                    logger.info(f'{len(data)} QA entries loaded')
                    for qa in tqdm(data):
                        item = predict_and_evaluate(qa, retriever_es, retriever_dpr, reader, USE_GPU)
                        output_items.append(item)
                        count += 1
                        if count > 10: break
                    save_output_items(output_filename, output_items, logger)
            except (Exception, KeyboardInterrupt) as e:
                logger.info(e)
                logger.info(f'An error occurred at Count {count}, saving what we have now...')
                save_output_items(output_filename, output_items, logger)


if __name__ == '__main__':
    main()
