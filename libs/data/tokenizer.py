from copy import deepcopy
import re, torch
import os
import numpy as np
from typing import Optional, Dict, Tuple
import tempfile
import mmap

# 1) simple tokenizer
def basic_english_tokenizer(text):
    return re.findall(r"\w+", text.lower())

# 2) registry
tokenizers = {}

def register_tokenizer(name):
    def decorator(cls):
        tokenizers[name] = cls
        return cls
    return decorator

# Global cache for GloVe embeddings to avoid loading multiple times
_GLOVE_CACHE = {}

# 3) your new GloVeTokenizer
@register_tokenizer('glove')
class GloVeTokenizer:
    def __init__(self,
                 glove_path: str = "glove.6B.300d.txt",
                 embedding_dim: int = 300):
        self.vocab = GloVeVocab(glove_path, embedding_dim)
        self.tokenizer = basic_english_tokenizer

    def __call__(self, text: str, max_len: Optional[int] = None):
        tokens = self.tokenizer(text)
        feats = self.vocab.get_vecs_by_tokens(tokens)
        if max_len is not None:
            feats = feats[:max_len]       # (t, c) → maybe shorter
        return feats.transpose(0, 1)     # → (c, t)

def make_tokenizer(name, **kwargs):
    return tokenizers[name](**kwargs)

def load_glove_embeddings_mmap(glove_path: str, embedding_dim: int) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Memory-efficient GloVe embedding loader using memory mapping.
    
    Args:
        glove_path: Path to the GloVe file
        embedding_dim: Dimension of embeddings
        
    Returns:
        Word-to-index dictionary and memory-mapped ndarray of embeddings
    """
    # Create cached binary version if it doesn't exist
    bin_path = f"{glove_path}.{embedding_dim}d.bin"
    vocab_path = f"{glove_path}.{embedding_dim}d.vocab.txt"
    
    if not os.path.exists(bin_path) or not os.path.exists(vocab_path):
        print(f"Creating memory-mappable GloVe files from {glove_path}")
        # First pass: count words and create word-to-index mapping
        word_to_idx = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                word = line.split(' ', 1)[0]
                word_to_idx[word] = line_idx
        
        # Save vocabulary to file for future use
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for word, idx in word_to_idx.items():
                f.write(f"{word} {idx}\n")
                
        # Second pass: write embeddings to binary file
        embeddings = np.zeros((len(word_to_idx), embedding_dim), dtype=np.float32)
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip().split(' ')
                word, values = parts[0], parts[1:]
                if len(values) != embedding_dim:
                    continue
                embeddings[word_to_idx[word]] = [float(x) for x in values]
                
        # Write to binary file
        embeddings.tofile(bin_path)
        print(f"Created memory-mappable files: {bin_path} and {vocab_path}")
    
    # Load vocabulary from file
    word_to_idx = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, idx = line.rstrip().split(' ', 1)
            word_to_idx[word] = int(idx)
    
    # Memory map the binary file
    embeddings = np.memmap(
        bin_path, 
        dtype=np.float32, 
        mode='r',
        shape=(len(word_to_idx), embedding_dim)
    )
    
    return word_to_idx, embeddings

class GloVeVocab:
    def __init__(self, glove_path: str, embedding_dim: int):
        self.dim = embedding_dim
        
        # Use global cache to avoid loading multiple times in multi-process setting
        global _GLOVE_CACHE
        cache_key = f"{glove_path}_{embedding_dim}"
        
        if cache_key in _GLOVE_CACHE:
            print(f"Using cached GloVe embeddings for {glove_path}")
            self.word_to_idx, self.embeddings = _GLOVE_CACHE[cache_key]
        else:
            print(f"Loading GloVe embeddings from {glove_path}")
            # Use memory-efficient loading
            self.word_to_idx, self.embeddings = load_glove_embeddings_mmap(glove_path, embedding_dim)
            # Store in cache for reuse
            _GLOVE_CACHE[cache_key] = (self.word_to_idx, self.embeddings)
            print(f"Loaded {len(self.word_to_idx)} GloVe word vectors")
            
        self.unk = torch.zeros(embedding_dim)

    def get_vecs_by_tokens(self, tokens, lower_case_backup=True):
        vecs = []
        for w in tokens:
            if w in self.word_to_idx:
                idx = self.word_to_idx[w]
                vecs.append(torch.from_numpy(self.embeddings[idx].copy()))
            elif lower_case_backup and w.lower() in self.word_to_idx:
                idx = self.word_to_idx[w.lower()]
                vecs.append(torch.from_numpy(self.embeddings[idx].copy()))
            else:
                vecs.append(self.unk)
        # returns (t, c)
        return torch.stack(vecs)
