"""
This script parses the given datasets
into a format that we can conveniently use
for our open domain QA experiments.
The output format is in JSON

To parse the original data, you need to download it first (see README)
and put it in the same directory as this script, rename it to e.g. "original"

Note that we try to reserve the original text as much as much,
only removing multiple spaces, and converting unreadable u-strings to readable ones.
No normalization is done.
"""

from tqdm import tqdm
from os.path import join
import json
import sys
import re
sys.path.insert(1, join('..', '..', 'common'))
from item_qa import ItemQA


def remove_html_tags(text):
    clean_pattern = re.compile('<.*?>')
    return re.sub(clean_pattern, '', text)


def remove_multiple_spaces(text):
    return re.sub(' +', ' ', text)


def clean(text):
    s = remove_html_tags(text)
    s = remove_multiple_spaces(s)
    return s.strip()


def get_answer(doc_tokens, answer_obj):
    start_byte = answer_obj['start_byte']
    end_byte = answer_obj['end_byte']
    answer = ''
    for t in doc_tokens:
        if t['start_byte'] >= start_byte and t['end_byte'] <= end_byte:
            answer += t['token'] + ' '
    answer = clean(answer)
    return answer


def parse(input_filename, output_filename):
    num_lines = 0
    with open(input_filename, 'r') as f:
        for _ in tqdm(f):
            num_lines += 1

    with open(input_filename, 'r') as f:
        items = []
        total_count = 0
        for l in tqdm(f, total=num_lines):
            l = l.replace('\n', '')
            doc = json.loads(l)
            annotations = doc['annotations']
            question_id = doc['document_url']
            question = doc['question_text']
            doc_tokens = doc['document_tokens']
            qa = ItemQA(question_id, question)
            for annt in annotations:
                long_answer_obj = annt['long_answer']
                long_answer = get_answer(doc_tokens, long_answer_obj)
                qa.add_answer(long_answer)
                total_count += 1
                short_answers = annt['short_answers']
                for sa_obj in short_answers:
                    short_answer = get_answer(doc_tokens, sa_obj)
                    qa.add_answer(short_answer)
                    total_count += 1
            items.append(qa.json())

        with open(output_filename, 'w', encoding='utf8') as out:
            json.dump(items, out, ensure_ascii=False)
        print('---------------------------------------')
        print(f'Finished processing {input_filename}')
        print(f'Number of QA pairs: {total_count}')
        print(f'Number of unique questions: {len(items)}')
        print('---------------------------------------')


if __name__ == '__main__':
    dev_path = 'original/v1.0-simplified_nq-dev-all.jsonl'
    dev_output_path = 'naturalQuestions-dev.json'
    parse(dev_path, dev_output_path)

    '''
    Expected output
    ---------------------------------------
    Finished processing original/v1.0-simplified_nq-dev-all.jsonl
    Number of QA pairs: 55,576
    Number of unique questions: 7,830
    ---------------------------------------
    '''