import pandas as pd
import faiss
import numpy as np
from datetime import datetime

from .retriever import retriever_model_inference

pd.set_option("display.max_rows", 500)
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 500)


##############################
### FAISS - vector db part ###
##############################
# TODO - wrap everything into one class
# TODO - add feature to only return similar documents if there are some (based on a sim. threshold)


def index_initializaton(vector_dim: int, partition_nr: int):
    quantizer = faiss.IndexFlatL2(vector_dim)
    index = faiss.IndexIVFFlat(
        quantizer, vector_dim, partition_nr, faiss.METRIC_INNER_PRODUCT
    )
    return index


def check_index_state(index):
    return index.is_trained


def train_index(index, vectors):
    faiss.normalize_L2(vectors)
    return index.train(vectors)


def add_index_vectors(index, vectors):
    return index.add(vectors)


def vector_search(search_text, top_k, index):
    search_vector = retriever_model_inference([search_text])
    _vector = np.array(search_vector)

    print(_vector.shape)

    faiss.normalize_L2(_vector)
    index.nprobe = 10  # increase the number of nearby cells to search too with `nprobe`
    k = min(top_k, index.ntotal)

    print(_vector.shape)
    print(index.d)

    start_time = datetime.now()
    distances, ann = index.search(_vector, k=k)
    end_time = datetime.now()
    process_time = end_time - start_time
    print(f"time required for clustered-based FAISS search: {process_time}")
    return distances, ann
