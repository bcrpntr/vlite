import torch
from transformers import AutoModel, AutoTokenizer
import logging

from .utils import visualize_tokens

logging.basicConfig(level=logging.INFO)

class EmbeddingModel:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.dimension = self.model.embeddings.position_embeddings.embedding_dim
        self.max_seq_length = self.model.embeddings.position_embeddings.num_embeddings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Tokenizer: {self.tokenizer}")
        logging.info(f"Device: {self.device}")

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed(self, texts, max_seq_length=256):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_length)
        logging.info(f"Encoded input done {encoded_input['input_ids'].shape}")
        encoded_input = {name: tensor.to(self.device) for name, tensor in encoded_input.items()}
        logging.info("Encoded input moved to device")
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        tensor_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        np_embeddings = tensor_embeddings.cpu().numpy()
        return np_embeddings

    def token_count(self, texts):
        tokens = 0
        for text in texts:
            tokens += len(self.tokenizer.tokenize(text))
        return tokens
