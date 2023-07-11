import numpy as np
from uuid import uuid4
from .model import EmbeddingModel
from .utils import chop_and_chunk, cos_sim

class VLite:
    def __init__(self, collection='vlite.npz', model_name=None):
        self.collection = collection
        self.model = EmbeddingModel() if model_name is None else EmbeddingModel(model_name)
        try:
            with np.load(self.collection, allow_pickle=True) as data:
                self.texts = data['texts'].tolist()
                self.metadata = data['metadata'].tolist()
                self.vectors = data['vectors']
        except FileNotFoundError:
            self.texts = []
            self.metadata = {}
            self.vectors = np.empty((0, self.model.dimension))

    def _add_vector(self, vector):
        self.vectors = np.vstack((self.vectors, vector))

    def get_similar_vectors(self, vector, top_k=5):
        sims = cos_sim(vector, self.vectors)
        sims = sims[0]
        top_k_idx = np.argsort(sims)[::-1][:top_k]
        return top_k_idx, sims[top_k_idx]

    def memorize(self, text, id=None, metadata=None):
        id = id or str(uuid4())
        chunks = chop_and_chunk(text)
        encoded_data = self.model.embed(texts=chunks)
        self._add_vector(encoded_data)
        for chunk in chunks:
            self.texts.append(chunk)
            idx = len(self.texts) - 1
            self.metadata[idx] = metadata or {}
            self.metadata[idx]['index'] = id or
