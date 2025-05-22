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
    # Tokenize and lower all words in sentences
    tokenized_sentences = [sentence.lower().split() for 
                           sentence in sentences]

    # Build vocabulary if not provided
    if vocab is None:
        vocab_set = set()
        for sentence in tokenized_sentences:
            vocab_set.update(sentence)
        features = sorted(list(vocab_set))
        features = vocab

    # Create embeddings matrix
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, sentence in enumerate(tokenized_sentences):
        for j, word in enumerate(features):
            embeddings[i, j] = sentence.count(word)

    return embeddings, features
