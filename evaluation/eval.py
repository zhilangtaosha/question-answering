"""following the evaluation script of DrQA"""

from collections import Counter
import string
import re
from typing import List


def normalize_text(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def regex_match_score(prediction: str, pattern: str) -> float:
    """Check if the prediction matches the given regular expression."""
    try:
        compiled = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE
        )
    except BaseException as e:
        print('Regular expression failed to compile: %s' % pattern)
        print(e)
        return False
    return float(compiled.match(prediction) is not None)


def reader_metric_max(metric_fn, pred_answer: str, gold_answers: List[str]) -> float:
    scores_for_ground_truths = []
    for gold_answer in gold_answers:
        score = metric_fn(pred_answer, gold_answer)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def retrieval_accuracy(gold_answer: str, retrieved_documents: List[str]) -> float:
    if not retrieved_documents:
        return 0
    total = float(len(retrieved_documents))
    hits = 0
    for d in retrieved_documents:
        if normalize_text(gold_answer) in normalize_text(d):
            hits += 1

    return hits / total


def retrieval_ranks_of_answer(answer: str, retrieved_documents: List[str]) -> List[int]:
    """
    return a list of ranks of which the retrieved documents contain the given answer
    """
    ranks = []
    answer_n = normalize_text(answer)
    for i, d in enumerate(retrieved_documents):
        d = normalize_text(d)
        rank = i + 1
        if answer_n in d:
            ranks.append(rank)

    return ranks


def retrieval_accuracy_max(gold_answers: List[str], retrieved_documents: List[str]) -> float:
    if not gold_answers:
        raise Exception('Incorrect format of the gold_answers parameter')
    scores = []
    for gold_answer in gold_answers:
        acc = retrieval_accuracy(gold_answer, retrieved_documents)
        scores.append(acc)
    return max(scores)
