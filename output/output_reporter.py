import sys, os
import plotly.express as px
from pandas import DataFrame
from tqdm.notebook import tqdm

sys.path.insert(1, os.path.join('..', 'common'))
from utils import *
from eval import *


def plot_line(px, df, x, y, title, template='plotly_white', **kwargs):
    fig = px.line(df, x=x, y=y, title=title, template=template, **kwargs)
    fig.show()


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
        result = {
            'k': [],
            'precision': [],
            'recall': []
        }
        if self.cache is None:
            # check disk first
            if self.cache_path and os.path.isfile(self.cache_path):
                self.cache = load_json(self.cache_path)
                return self.cache
            rs = self.get_recalls(field, begin, end, step)
            ps = self.get_precisions(field, begin, end, step)
            for i, k in enumerate(rs['k']):
                result['k'].append(k)
                r = rs['recall'][i]
                result['recall'].append(r)
                p = ps['precision'][i]
                result['precision'].append(p)
            if self.save_cache:
                print('Saving cache ', self.cache_path)
                save_json(result, self.cache_path)
        else:
            for k in range(begin, end, step):
                i = self.cache['k'].index(k)
                if i > 0:
                    result['k'].append(k)
                    r = self.cache['recall'][i]
                    result['recall'].append(r)
                    p = self.cache['precision'][i]
                    result['precision'].append(p)

        return result

    def plot_recalls(self, field='bm25_ranks', begin=5, end=5000, step=5):
        res = self.get_recalls(field, begin, end, step)
        df = DataFrame(res)
        plot_line(px, df, x='k', y='recall', title='Recalls')

    def plot_precisions(self, field='bm25_ranks', begin=5, end=5000, step=5):
        res = self.get_precisions(field, begin, end, step)
        df = DataFrame(res)
        plot_line(px, df, x='k', y='precision', title='Precisions')

    def plot_recalls_precisions(self, field='bm25_ranks', begin=5, end=5000, step=5):
        res = self.get_recalls_precisions(field, begin, end, step)
        df = DataFrame(res)
        plot_line(px, df, x='recall', y='precision', title='Precision-Recall')


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

    def aggregate_adjust(self):
        ref_k = 5000
        k_map = {
            'wikipedia_50': ref_k,
            'wikipedia_100': int(ref_k * (50 / 100)),
            'wikipedia_100_stride_50': int(ref_k * (50 / 100)),
            'wikipedia_150': int(ref_k * (50 / 150)),
            'wikipedia_200': int(ref_k * (50 / 200)),
            'wikipedia_paragraph': int(ref_k * (50 / 61)),
            'wikipedia': int(ref_k * (50 / 372)),
        }
        if self.cache is None:
            res = {
                'index': [],
                'k': [],
                'recall': [],
                'precision': []
            }
            for reporter in self.reporters:
                index = reporter.json_path.split('/')[0]
                max_k = k_map[index]
                output = reporter.get_recalls_precisions(end=max_k)
                cur_max_k = max(output['k'])
                cur_min_k = min(output['k'])
                target_max_k = ref_k
                target_min_k = cur_min_k
                for i, k in enumerate(output['k']):
                    res['index'].append(index)
                    target_k = (target_max_k - target_min_k) * (k - cur_min_k) / (cur_max_k - cur_min_k) + target_min_k
                    res['k'].append(int(target_k))
                    res['recall'].append(output['recall'][i])
                    res['precision'].append(output['precision'][i])
            self.cache = res
            return res
        else:
            return self.cache

    def plot_recalls(self, adjusted=True):
        agg = self.aggregate_adjust() if adjusted else self.aggregate()
        df = DataFrame(agg)
        plot_line(px, df, x='k', y='recall', title='Recalls', color='index')

    def plot_precisions(self, adjusted=True):
        agg = self.aggregate_adjust() if adjusted else self.aggregate()
        df = DataFrame(agg)
        plot_line(px, df, x='k', y='precision', title='Precisions', color='index')

    def plot_recalls_precisions(self, adjusted=True):
        agg = self.aggregate_adjust() if adjusted else self.aggregate()
        df = DataFrame(agg)
        plot_line(px, df, x='recall', y='precision', title='Precision-Recall', color='index')


def merge_outputs(json_paths: List[str], merged_filename):
    items = []
    for _path in json_paths:
        items += load_json(_path)
    if items:
        save_json(items, merged_filename)


def show_plots(adjusted=True):
    index_types = ['_50', '_100', '_100_stride_50', '_150', '_200', '_paragraph']
    json_paths = []
    for t in index_types:
        json_paths.append(f'wikipedia{t}/qa_BM25_wikipedia{t}_5000_merged.json')
    agg = OutputReporterAgg(json_paths)
    agg.plot_recalls(adjusted=adjusted)
    agg.plot_precisions(adjusted=adjusted)
    agg.plot_recalls_precisions(adjusted=adjusted)


if __name__ == '__main__':
    show_plots()
