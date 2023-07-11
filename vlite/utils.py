import numpy as np
import regex as re

def chop_and_chunk(text, max_seq_length=256):
    if isinstance(text, str):
        text = [text]
    if all('\n' in t for t in text):
        return text 
    chunks = []
    for t in text: 
        parts = re.split('\n+', t)  
        for p in parts:
            tokens = p.split()
            chunk = ''
            count = 0
            for t in tokens:
                if count + len(t) < max_seq_length:
                    count += len(t) 
                    chunk += t + ' '
                else:
                    chunks.append(chunk.strip())
                    count = 0
                    chunk = ''
            if chunk != '':
                chunks.append(chunk.strip())
    return chunks

def cos_sim(a, b):
    sims = a @ b.T
    sims /= np.linalg.norm(a) * np.linalg.norm(b, axis=1) 
    return sims
