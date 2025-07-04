import torch
import numpy as np

def least_confident_sampling(probas, n_samples):
    #selects samples where the model is least confident.
    confidence = probas.max(axis=1)
    least_confident_indices = np.argsort(confidence)[:n_samples]
    return least_confident_indices


def entropy_sampling(probas, n_samples):
    #selects samples with the highest prediction entropy which means the model is most uncertain
    entropy = -np.sum(probas * np.log2(probas + 1e-10), axis=1)   #https://encord.com/blog/active-learning-computer-vision-guide/

    entropy_indices = np.argsort(entropy)[-n_samples:]
    return entropy_indices
