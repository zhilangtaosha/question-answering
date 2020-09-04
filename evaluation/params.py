from os.path import join

# Path params
# Retriever - ElasticSearch
ES_HOST = 'localhost'
ES_PORT = 9200
ES_INDEX_NAME = 'wikipedia_en'
# Datasets
DATADIR = join('..', 'data')
DATASETS = [
    join(DATADIR, 'squad2', 'squad2-dev.json'),
]
# Readers
MODELSDIR = join('..', 'models')
READERS = [
    join(MODELSDIR, 'roberta-base-squad2')
]

# Model params
RETRIEVER_TOP_K = 10
READER_TOP_K = 1
