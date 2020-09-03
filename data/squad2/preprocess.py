"""
This script parses the given datasets
into a format that we can conveniently use
for our open domain QA experiments.
The output format is in JSON

To pare the original data, you need to download it first (see README)
and put it in the same directory as this script, rename it to e.g. "original"

Note that only the questions with answers are extracted.
"""

from tqdm import tqdm
from os.path import join
import json
import sys
sys.path.insert(1, join('..', '..', 'common'))
from item_qa import ItemQA


def parse(input_filename, output_filename):
    with open(input_filename, 'r') as f:
        items = []
        total_count = 0
        obj = json.load(f)
        data = obj['data']
        for entry in tqdm(data):
            title = entry['title']
            paragraphs = entry['paragraphs']
            for p in paragraphs:
                for qa in p['qas']:
                    answers = qa['answers']
                    if len(answers) > 0:
                        question_id = qa['id']
                        question = qa['question']
                        item = ItemQA(question_id, question)
                        for a in answers:
                            answer = a['text']
                            item.add_answer(answer)
                            total_count += 1
                        items.append(item.json())

        with open(output_filename, 'w', encoding='utf8') as out:
            json.dump(items, out)
        print('---------------------------------------')
        print(f'Finished processing {input_filename}')
        print(f'Number of QA pairs: {total_count}')
        print(f'Number of unique questions: {len(items)}')
        print('---------------------------------------')


if __name__ == '__main__':
    original_data_dir = join('original')
    inputs = [join(original_data_dir, 'dev-v2.0.json'),
              join(original_data_dir, 'train-v2.0.json')]
    outputs = ['squad2-dev.json', 'squad2-train.json']
    for i, o in zip(inputs, outputs):
        parse(i, o)

    '''
    Expected output:
    ---------------------------------------
    Finished processing original/dev-v2.0.json
    Number of QA pairs: 20,302
    Number of unique questions: 5,928
    ----------------------------------------
    Finished processing original/train-v2.0.json
    Number of QA pairs: 86,821
    Number of unique questions: 86,821
    ----------------------------------------
    '''