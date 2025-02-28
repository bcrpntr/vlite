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
            self.metadata[idx]['index'] = id or idx
        self.save()
        return id, self.vectors

    def remember_by_id(self, id):
        return self.metadata[id]

    def remember_by_text(self, text, top_k=5):
        sims = cos_sim(self.model.embed(texts=text), self.vectors)
        sims = sims[0]
        top_5_idx = np.argpartition(sims, -top_k)[-top_k:]
        top_5_idx = top_5_idx[np.argsort(sims[top_5_idx])[::-1]]
        return [self.texts[idx] for idx in top_5_idx], sims[top_5_idx]

    def save(self):
        with open(self.collection, 'wb') as f:
            np.savez(f, texts=self.texts, metadata=self.metadata, vectors=self.vectors)
