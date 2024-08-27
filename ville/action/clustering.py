import marigold as mg
import numpy as np
from sklearn.metrics import calinski_harabasz_score

# Using buffer of observations, uses a pre-trained model to encode actions


def cluster(cfg, actions):
    # Find optimal centroids for the actions seen through the buffer
    score = 0
    argmax = 0
    for i in range(2, cfg.max_primitives + 1, 2):
        _, assignment, _, _ = mg.marigold(X=actions, n_clusters=i, n_init=3)
        sc = calinski_harabasz_score(actions, assignment)
        if sc>score:
            score = sc
            argmax = i
    return mg.marigold(X= actions, n_clusters = argmax, n_init = cfg.k_means_attempts)

def gale_shapley(m_preferences, f_preferences, distances):
    return True

