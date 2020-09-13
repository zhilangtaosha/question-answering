import os, sys, time
import numpy as np
import json
from tqdm import tqdm

sys.path.insert(1, os.path.join('..', 'common'))
from item_qa import ItemQA2
from utils import *
from params import *
from eval import *

DATA_DIR = join('..', 'data')
DATASETS = [
    join(DATA_DIR, 'squad2', 'squad2-dev.json'),
    join(DATA_DIR, 'naturalQuestions', 'naturalQuestions-dev-clean.json'),
    join(DATA_DIR, 'wikiQA', 'wikiQA-test.json'),
]
RETRIEVER_ES_TOP_K = 1000
RETRIEVER_DPR_TOP_K = 20


def get_output_filename(reader_path, data_path):
    retriever = 'BM25'
    reader = os.path.basename(reader_path)
    data = os.path.basename(data_path).replace('.json', '')
    output_name = f'qa_{retriever}_{RETRIEVER_ES_TOP_K}_DPR_{RETRIEVER_DPR_TOP_K}_{reader}_{data}'
    if SUBSET is not None:
        output_name += f'_{SUBSET}'
    return output_name + '.json'


def predict_and_evaluate(gold_qa_entry, retriever_es, retriever_dpr, faiss_index, reader, normalize=False):
    gold_answers = gold_qa_entry['answers']
    if len(gold_answers) == 0:
        return None
    question_id = gold_qa_entry['question_id']
    question = gold_qa_entry['question']
    pred_answer = ''
    es_ranks = []
    dpr_ranks = []
    f1 = 0
    p = 0
    r = 0
    em = 0
    t = 0

    start_time = time.time()
    docs = retriever_es.retrieve(question, top_k=RETRIEVER_ES_TOP_K)
    if docs:
        q_vecs = retriever_dpr.embed_queries([question])
        es_doc_texts = [d.text for d in docs]
        if normalize:
            es_doc_texts = [normalize_text(d) for d in es_doc_texts]
        p_vecs = retriever_dpr.embed_passages(es_doc_texts)
        faiss_index.add(np.array(p_vecs))
        D, I = faiss_index.search(np.array(q_vecs), RETRIEVER_ES_TOP_K)
        dpr_docs = [docs[I[0][i]] for i in range(RETRIEVER_DPR_TOP_K)]
        prediction = reader.predict(question=question, documents=dpr_docs, top_k=READER_TOP_K)
        pred_answers = prediction['answers']
        if pred_answers:
            pred_answer = pred_answers[0]['answer']
        t = time.time() - start_time
        faiss_index.reset()
        # eval
        es_ranks = recall_ranks_merge(gold_answers, es_doc_texts)
        dpr_ranks = recall_ranks_convert(es_ranks, I[0])
        f1, p, r = reader_f1_max(pred_answer, gold_answers)
        em = reader_match_max(exact_match_score, pred_answer, gold_answers)

    item = ItemQA2(question_id, question,
                   bm25_ranks=es_ranks,
                   dense_ranks=dpr_ranks,
                   f1=f1,
                   p=p,
                   r=r,
                   em=em,
                   t=t)
    item.add_pred_answer(pred_answer)
    for g_answer in gold_answers:
        item.add_gold_answer(g_answer)
    return item


def summarize(output_filename: str):
    with open(output_filename, 'r', encoding='utf8') as f:
        output_items = json.load(f)
        summary_filename = output_filename.replace('.json', '_summary.md')
        pipeline_name = output_filename.replace(".json", "").replace("qa_", "")
        with open(summary_filename, 'w', encoding='utf8') as fw:
            round_num = 3
            fw.write('### Pipeline Parameters:\n')
            fw.write(f'* Name: {pipeline_name}\n')
            fw.write(f'* BM25 top K = {RETRIEVER_ES_TOP_K}\n')
            fw.write(f'* Dense top K = {RETRIEVER_DPR_TOP_K}\n')
            fw.write(f'* Reader top K = {READER_TOP_K}\n')
            fw.write(f'* FAISS dimension = {FAISS_INDEX_DIMENSION}\n')
            fw.write(f'* USE_GPU = {USE_GPU}\n')
            fw.write(f'* Number of questions: {len(output_items)}\n')
            fw.write(f'* Seed = {SEED}\n')
            fw.write(f'------\n')

            fw.write('### BM25 Retrieval recall \n')
            bm25_ranks_list = [item['bm25_ranks'] for item in output_items]
            fw.write(f'* BM25 Recall @ 5: {round(recall_at_k(bm25_ranks_list, 5), round_num)}\n')
            fw.write(f'* BM25 Recall @ 10: {round(recall_at_k(bm25_ranks_list, 10), round_num)}\n')
            fw.write(f'* BM25 Recall @ 20: {round(recall_at_k(bm25_ranks_list, 20), round_num)}\n')
            fw.write(f'* BM25 Recall @ 50: {round(recall_at_k(bm25_ranks_list, 50), round_num)}\n')
            fw.write(f'* BM25 Recall @ 100: {round(recall_at_k(bm25_ranks_list, 100), round_num)}\n')
            fw.write('### Dense Retrieval recall \n')
            dense_ranks_list = [item['dense_ranks'] for item in output_items]
            fw.write(f'* Dense Recall @ 5: {round(recall_at_k(dense_ranks_list, 5), round_num)}\n')
            fw.write(f'* Dense Recall @ 10: {round(recall_at_k(dense_ranks_list, 10), round_num)}\n')
            fw.write(f'* Dense Recall @ 20: {round(recall_at_k(dense_ranks_list, 20), round_num)}\n')
            fw.write(f'* Dense Recall @ 50: {round(recall_at_k(dense_ranks_list, 50), round_num)}\n')
            fw.write(f'* Dense Recall @ 100: {round(recall_at_k(dense_ranks_list, 100), round_num)}\n')
            fw.write('### BM25 Retrieval precision \n')
            fw.write(f'* BM25 Precision @ 5: {round(precision_at_k(bm25_ranks_list, 5), round_num)}\n')
            fw.write(f'* BM25 Precision @ 10: {round(precision_at_k(bm25_ranks_list, 10), round_num)}\n')
            fw.write(f'* BM25 Precision @ 20: {round(precision_at_k(bm25_ranks_list, 20), round_num)}\n')
            fw.write(f'* BM25 Precision @ 50: {round(precision_at_k(bm25_ranks_list, 50), round_num)}\n')
            fw.write(f'* BM25 Precision @ 100: {round(precision_at_k(bm25_ranks_list, 100), round_num)}\n')
            fw.write('### Dense Retrieval Precision \n')
            fw.write(f'* Dense Precision @ 5: {round(precision_at_k(dense_ranks_list, 5), round_num)}\n')
            fw.write(f'* Dense Precision @ 10: {round(precision_at_k(dense_ranks_list, 10), round_num)}\n')
            fw.write(f'* Dense Precision @ 20: {round(precision_at_k(dense_ranks_list, 20), round_num)}\n')
            fw.write(f'* Dense Precision @ 50: {round(precision_at_k(dense_ranks_list, 50), round_num)}\n')
            fw.write(f'* Dense Precision @ 100: {round(precision_at_k(dense_ranks_list, 100), round_num)}\n')

            fw.write('### F1 \n')
            f1s = [item['f1'] for item in output_items]
            fw.write(f'* Mean F1 per q: {round(sum(f1s) / len(f1s), round_num)}\n')
            fw.write(f'* Median F1 per q: {round(np.median(f1s), round_num)}\n')
            fw.write(f'* Max F1 per q: {round(max(f1s), round_num)}\n')
            fw.write(f'* Min F1 per q: {round(min(f1s), round_num)}\n')
            fw.write(f'* Std F1 per q: {round(np.std(f1s), round_num)}\n')
            fw.write('### Precision \n')
            ps = [item['p'] for item in output_items]
            fw.write(f'* Mean precision per q: {round(sum(ps) / len(ps), round_num)}\n')
            fw.write(f'* Median precision per q: {round(np.median(ps), round_num)}\n')
            fw.write(f'* Max precision per q: {round(max(ps), round_num)}\n')
            fw.write(f'* Min precision per q: {round(min(ps), round_num)}\n')
            fw.write(f'* Std precision per q: {round(np.std(ps), round_num)}\n')
            fw.write('### Recall \n')
            rs = [item['r'] for item in output_items]
            fw.write(f'* Mean recall per q: {round(sum(rs) / len(rs), round_num)}\n')
            fw.write(f'* Median recall per q: {round(np.median(rs), round_num)}\n')
            fw.write(f'* Max recall per q: {round(max(rs), round_num)}\n')
            fw.write(f'* Min recall per q: {round(min(rs), round_num)}\n')
            fw.write(f'* Std recall per q: {round(np.std(rs), round_num)}\n')
            fw.write('### Exact Match \n')
            ems = [item['em'] for item in output_items]
            fw.write(f'* Mean EM per q: {round(sum(ems) / len(ems), round_num)}\n')
            fw.write(f'* Median EM per q: {round(np.median(ems), round_num)}\n')
            fw.write(f'* Max EM per q: {round(max(ems), round_num)}\n')
            fw.write(f'* Min EM per q: {round(min(ems), round_num)}\n')
            fw.write(f'* Std EM per q: {round(np.std(ems), round_num)}\n')
            fw.write('### Time(s) \n')
            ts = [item['t'] for item in output_items]
            fw.write(f'* Mean time per q: {round(sum(ts) / len(ts), 2)}s\n')
            fw.write(f'* Max time per q: {round(max(ts), 2)}s\n')
            fw.write(f'* Min time per q: {round(min(ts), 2)}s\n')
            fw.write(f'* Std time per q: {round(np.std(ts), 2)}s\n')


def save_output(output_filename: str, output_items: List[ItemQA2], logger):
    items_json = [item.json() for item in output_items]
    save_json(items_json, output_filename)
    logger.info(f'Output is saved to {output_filename}')
    summarize(output_filename)


def main():
    logger = get_logger('run_ES_DPR_Reader', 'run_ES_DPR_Reader.log')
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
    logger.info('loading FAISS index...')
    faiss_index = get_faiss_ip_index(d=FAISS_INDEX_DIMENSION, use_gpu=USE_GPU)
    for reader_path in READERS:
        logger.info(f'loading reader at {reader_path} ...')
        reader = get_neural_reader(reader_path, use_gpu=USE_GPU)
        for data_path in DATASETS:
            output_filename = get_output_filename(reader_path, data_path)
            logger.info(f'output filename: {output_filename}')
            output_items = []
            count = 0
            try:
                data = load_qa_data(data_path, seed=SEED, subset=SUBSET)
                logger.info(f'{len(data)} QA entries loaded')
                for qa in tqdm(data):
                    if count > 754:
                        item = predict_and_evaluate(qa, retriever_es, retriever_dpr, faiss_index, reader)
                        if item is not None:
                            output_items.append(item)
                    count += 1
                save_output(output_filename, output_items, logger)

            except (Exception, KeyboardInterrupt) as e:
                logger.info(e)
                logger.info(f'An error occurred at Count {count}, saving what we have now...')
                save_output(output_filename, output_items, logger)


if __name__ == '__main__':
    main()

    # import json
    # with open('qa_BM25_DPR_electra-base-squad2_squad2-dev (932).json', 'r') as f:
    #     obj = json.load(f)
    #     print(len(obj))

    # faiss_index = get_faiss_ip_index(d=3, use_gpu=USE_GPU)
    # xb = np.random.random((10,3)).astype('float32')
    # faiss_index.add(xb)
    # print(faiss_index.ntotal)
    # faiss_index.reset()
    # print(faiss_index.ntotal)
