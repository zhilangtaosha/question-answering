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

    def get_recalls(self, field, start=5, end=100, step=5):
        recalls = []
        if self.items:
            assert field in self.items[0]
            ranks_list = [item[field] for item in self.items]
            for k in range(start, end, step):
                r = recall_at_k(ranks_list, k)
                recalls.append((k, r))
        return recalls


if __name__ == '__main__':
    example_path = 'wiki_50/qa_BM25_wikipedia_50_5000_naturalQuestions-dev-clean_1000.json'
    reporter = OutputReporter(example_path)
    print(reporter.items[0])