#!/usr/bin/env python3
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.
    
    Parameters:
    - sentences (list): List of sentences to analyze
    - vocab (list or None): List of vocabulary words
                             If None, generate from
                             
    Returns:
    - embeddings (np.ndarray): Shape (s, f), s = #sentences
    - features (list): List of features used (vocab)
    """
    if vocab is None:
        # Build vocab from all words in sentences
        words = set()
        for sentence in sentences:
            for word in sentence.lower().split():
                words.add(word)
        features = sorted(words)  # consistent order
    else:
        features = vocab

    # Create embeddings matrix
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, sentence in enumerate(sentences):
        words = sentence.lower().split()
        for j, feature in enumerate(features):
            embeddings[i, j] = words.count(feature)

    return embeddings, features
