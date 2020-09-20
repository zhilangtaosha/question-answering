import sys, os

sys.path.insert(1, os.path.join('..', 'common'))
from utils import *
from evaluation import *
import numpy as np
from tqdm import tqdm
import plotly.express as px
from pandas import DataFrame


def normalize_min_max(min_val, max_val, val):
    if max_val > min_val:
        return (val - min_val) / (max_val - min_val)
    elif max_val == min_val:
        return 1
    return None


def bertserinize(top_docs, mu,
                 ranker_field,
                 reader_field='pred_answer_prob',
                 bertserini_field='bertserini_score'):
    if top_docs:
        ranker_scores = [td[ranker_field] for td in top_docs]
        min_ranker_score = min(ranker_scores)
        max_ranker_score = max(ranker_scores)
        reader_scores = [td[reader_field] for td in top_docs]
        min_reader_score = min(reader_scores)
        max_reader_score = max(reader_scores)
        for td in top_docs:
            ranker_s = normalize_min_max(min_ranker_score, max_ranker_score, td[ranker_field])
            reader_s = normalize_min_max(min_reader_score, max_reader_score, td[reader_field])
            assert ranker_s is not None
            assert reader_s is not None
            new_s = (1 - mu) * ranker_s + mu * reader_s
            td[bertserini_field] = new_s
    return top_docs


def evaluate_bertserini_with_mu(qas, mu, ranker_field='dpr_score'):
    f1s = []
    ems = []
    for qa in qas:
        top_docs = bertserinize(qa['top_docs'], mu, ranker_field)
        top_docs = sorted(top_docs, key=lambda x: x['bertserini_score'], reverse=True)
        gold_answers = qa['gold_answers']
        if top_docs:
            top_pred_answer = top_docs[0]['pred_answer']
            f1, p, r = reader_f1_max(top_pred_answer, gold_answers)
            em = reader_match_max(exact_match_score, top_pred_answer, gold_answers)
            f1s.append(f1)
            ems.append(em)
        elif gold_answers:
            f1s.append(0)
            ems.append(0)
        else:
            f1s.append(1)
            ems.append(1)

    f1mean = np.mean(f1s)
    f1sd = np.std(f1s)
    em_mean = np.mean(ems)
    em_sd = np.std(ems)
    return {
        'f1_mean': f1mean,
        'f1_sd': f1sd,
        'em_mean': em_mean,
        'em_sd': em_sd
    }


def evaluate_bertserini(qa_path, dnom=100, ranker_field='dpr_score'):
    qas = load_json(qa_path)
    result = {
        'f1': [],
        'em': [],
        'mu': []
    }
    max_f1 = 0
    max_f1_mu = None
    max_em = 0
    max_em_mu = None
    for i in tqdm(range(dnom + 1)):
        mu = i / dnom
        result['mu'].append(mu)
        res = evaluate_bertserini_with_mu(qas, mu, ranker_field)
        result['f1'].append(res['f1_mean'])
        result['em'].append(res['em_mean'])
        if res['f1_mean'] > max_f1:
            max_f1 = res['f1_mean']
            max_f1_mu = mu
        if res['em_mean'] > max_em:
            max_em = res['em_mean']
            max_em_mu = mu
    df = DataFrame(result)
    fig = px.line(df, x="mu", y="f1", title='F1 over mu')
    fig.show()
    return {
        'max_f1': max_f1,
        'max_f1_mu': max_f1_mu,
        'max_em': max_em,
        'max_em_mu': max_em_mu
    }


if __name__ == '__main__':
    qa_path = '../output/wikipedia_100_stride_50/bm25+dpr+eletra/qa_BM25_wikipedia_100_stride_50__1000_DPR_20__electra-base-squad2__squad2-dev_1000.json'
    evaluate_bertserini(qa_path, dnom=100, ranker_field='bm25_score')
    """
    {'max_f1': 0.39830328682325594,
     'max_f1_mu': 0.38,
     'max_em': 0.305,
     'max_em_mu': 0.37}
    """
    evaluate_bertserini(qa_path, dnom=100, ranker_field='dpr_score')
    """
    {'max_f1': 0.34944871712656406,
     'max_f1_mu': 0.74,
     'max_em': 0.263,
     'max_em_mu': 0.92}
    """
