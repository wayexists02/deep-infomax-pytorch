import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_feature_vector(feature_vectors, labels):
    feature_vectors = feature_vectors.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    N = feature_vectors.shape[0]
    C = feature_vectors.shape[1]

    feature_vectors = feature_vectors.reshape(N*C, -1)
    labels = np.tile(labels.reshape(N, 1), [1, C]).reshape(N*C,)

    tsne = TSNE(n_components=2)
    feature_vectors_tsne = tsne.fit_transform(feature_vectors)
    
    for l in range(10):
        feature_vectors_tsne_l = feature_vectors_tsne[labels == l]
        plt.scatter(feature_vectors_tsne_l[:, 0], feature_vectors_tsne_l[:, 1], label=f"{l}", alpha=0.5)

    plt.show()
