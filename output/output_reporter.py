import json
import sys, os
sys.path.insert(1, os.path.join('..', 'common'))
from utils import *
from eval import *


class OutputReporter:
    def __init__(self, output_path):
        self.output_path = output_path
        self.round_number = 3
        self.items = []
        with open(output_path, 'r', encoding='utf8') as f:
            self.items = json.load(f)

    def get_recalls(self, field):
        if self.items:
            assert field in self.items[0]
            ranks_list = [item[field] for item in self.items]



if __name__ == '__main__':
    example_path = 'wiki_50/qa_BM25_wikipedia_50_5000_naturalQuestions-dev-clean_1000.json'
    reporter = OutputReporter(example_path)
    print(reporter.items[0])


    # round(recall_at_k(bm25_ranks_list, 5), round_num)