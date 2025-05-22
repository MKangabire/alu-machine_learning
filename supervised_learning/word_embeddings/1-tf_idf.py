#!/usr/bin/env python3

import numpy as np
import math


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    Parameters:
    - sentences: list of sentences (strings)
    Returns:
    - embeddings: numpy.ndarray of shape (s, f)
    - features: list of features (words)
    """
    # Build vocabulary if not provided
    if vocab is None:
        words = set()
        for sentence in sentences:
            for word in sentence.lower().split():
                words.add(word)
        features = sorted(words)
    else:
        features = vocab

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f))

    # Compute IDF for each feature
    idf = []
    for feature in features:
        doc_count = sum(1 for sentence in sentences if 
                        feature in sentence.lower().split())
        idf_score = math.log((1 + s) / (1 + doc_count)) + 1
        idf.append(idf_score)

    # Compute TF and TF-IDF
    for i, sentence in enumerate(sentences):
        words = sentence.lower().split()
        total_words = len(words)
        for j, feature in enumerate(features):
            tf = words.count(feature) / total_words \
            if total_words > 0 else 0
            embeddings[i, j] = tf * idf[j]

    return embeddings, features
