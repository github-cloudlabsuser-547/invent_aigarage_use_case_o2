import pandas as pd
import faiss
import numpy as np
from datetime import datetime

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