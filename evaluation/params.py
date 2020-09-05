from os.path import join

# Path params
# Retriever - ElasticSearch
ES_HOST = 'localhost'
ES_PORT = 9200
ES_INDEX_NAME = 'wikipedia_en'

# Retriever - DPR
DPR_MODEL_PATH = '../models/dpr/multi_hf_bert_base.cp'

# Datasets
DATA_DIR = join('..', 'data')
DATASETS = [
    join(DATA_DIR, 'squad2', 'squad2-dev.json'),
]
# Readers
MODELS_DIR = join('..', 'models')
READERS = [
    join(MODELS_DIR, 'electra-base-squad2')
]

# Model params
USE_GPU = True
RETRIEVER_ES_TOP_K = 100
RETRIEVER_DPR_TOP_K = 3
READER_TOP_K = 1
FAISS_INDEX_DIMENSION = 768
