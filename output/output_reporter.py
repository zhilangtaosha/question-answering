import sys, os
import plotly.express as px
from pandas import DataFrame
from tqdm.notebook import tqdm

sys.path.insert(1, os.path.join('..', 'common'))
from utils import *
from eval import *


class OutputReporter:
    def __init__(self, json_path=None, save_cache=True):
        self.json_path = json_path
        self.cache_path = json_path.replace('.json', '_cache.json') if json_path else None
        self.items = []
        self.save_cache = save_cache
        self.cache = None
        if json_path:
            self.items = load_json(json_path)

    def get_recalls(self, field='bm25_ranks', begin=5, end=5000, step=5):
        results = {
            'k': [],
            'recall': []
        }
        if self.items:
            assert field in self.items[0]
            ranks_list = [item[field] for item in self.items]
            for k in tqdm(range(begin, end, step)):
                r = recall_at_k(ranks_list, k)
                results['k'].append(k)
                results['recall'].append(r)
        return results

    def get_precisions(self, field='bm25_ranks', begin=5, end=5000, step=5):
        results = {
            'k': [],
            'precision': []
        }
        if self.items:
            assert field in self.items[0]
            ranks_list = [item[field] for item in self.items]
            for k in tqdm(range(begin, end, step)):
                p = precision_at_k(ranks_list, k)
                results['k'].append(k)
                results['precision'].append(p)
        return results

    def get_recalls_precisions(self, field='bm25_ranks', begin=5, end=5000, step=5):
        if self.cache is None:
            # check disk first
            if self.cache_path and os.path.isfile(self.cache_path):
                self.cache = load_json(self.cache_path)
                return self.cache
            rs = self.get_recalls(field, begin, end, step)
            ps = self.get_precisions(field, begin, end, step)
            result = {
                'k': [],
                'precision': [],
                'recall': []
            }
            for i, k in enumerate(rs['k']):
                result['k'].append(k)
                r = rs['recall'][i]
                result['recall'].append(r)
                p = ps['precision'][i]
                result['precision'].append(p)
            if self.save_cache:
                print('Saving cache ', self.cache_path)
                save_json(result, self.cache_path)
            return result
        else:
            return self.cache

    def plot_recalls(self, field='bm25_ranks', begin=5, end=5000, step=5):
        res = self.get_recalls(field, begin, end, step)
        df = DataFrame(res)
        fig = px.line(df, x='k', y='recall', title='Recalls')
        fig.show()

    def plot_precisions(self, field='bm25_ranks', begin=5, end=5000, step=5):
        res = self.get_precisions(field, begin, end, step)
        df = DataFrame(res)
        fig = px.line(df, x='k', y='precision', title='Precisions')
        fig.show()

    def plot_recalls_precisions(self, field='bm25_ranks', begin=5, end=5000, step=5):
        res = self.get_recalls_precisions(field, begin, end, step)
        df = DataFrame(res)
        fig = px.line(df, x='recall', y='precision', title='Precision-Recall')
        fig.show()


class OutputReporterAgg:
    def __init__(self, json_paths: List[str] = None):
        self.cache = None
        self.reporters: List[OutputReporter] = []
        for _path in tqdm(json_paths):
            # caching
            reporter = OutputReporter(_path)
            reporter.get_recalls_precisions()
            self.reporters.append(reporter)

    def aggregate(self):
        if self.cache is None:
            res = {
                'index': [],
                'k': [],
                'recall': [],
                'precision': []
            }
            for reporter in self.reporters:
                index = reporter.json_path.split('/')[0]
                output = reporter.get_recalls_precisions()
                for i, k in enumerate(output['k']):
                    res['index'].append(index)
                    res['k'].append(k)
                    res['recall'].append(output['recall'][i])
                    res['precision'].append(output['precision'][i])
            self.cache = res
            return res
        else:
            return self.cache

    def plot_recalls(self):
        agg = self.aggregate()
        df = DataFrame(agg)
        fig = px.line(df, x='k', y='recall', title='Recalls', color='index')
        fig.show()

    def plot_precisions(self):
        agg = self.aggregate()
        df = DataFrame(agg)
        fig = px.line(df, x='k', y='precision', title='Precisions', color='index')
        fig.show()

    def plot_recalls_precisions(self):
        agg = self.aggregate()
        df = DataFrame(agg)
        fig = px.line(df, x='recall', y='precision', title='Precision-Recall', color='index')
        fig.show()


def merge_outputs(json_paths: List[str], merged_filename):
    items = []
    for _path in json_paths:
        items += load_json(_path)
    if items:
        save_json(items, merged_filename)


def show_plots():
    index_types = ['50', '100', '100_stride_50', '150', '200', 'paragraph']
    json_paths = []
    for t in index_types:
        json_paths.append(f'wikipedia_{t}/qa_BM25_wikipedia_{t}_5000_merged.json')
    agg = OutputReporterAgg(json_paths)
    agg.plot_recalls()
    agg.plot_precisions()
    agg.plot_recalls_precisions()


if __name__ == '__main__':
    show_plots()
