import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as skm
import scipy
from scipy.sparse import diags
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from utils import plot_clustering_result, plot_the_bend
from build_similarity_graph import build_similarity_graph
from generate_data import blobs, two_moons, point_and_circle


def build_laplacian(W, laplacian_normalization=""):
    """
    Compute graph Laplacian.
    :param W: adjacency matrix
    :param laplacian_normalization:  string selecting which version of the laplacian matrix to construct
                                     'unn':  unnormalized,
                                     'sym': symmetric normalization
                                     'rw':  random-walk normalization
    :return: L: (n x n) dimensional matrix representing the Laplacian of the graph
    """
    D = np.asarray(W.sum(1)).squeeze()

    if laplacian_normalization == 'unn':
        L = diags(D) - W
    
    elif laplacian_normalization == 'sym':
        D = diags(D ** (-1/2.))
        L = diags(np.ones(W.shape[0])) - D.dot(W).dot(D)
    
    elif laplacian_normalization == 'rw':
        D = diags(D ** (-1))
        L = diags(np.ones(W.shape[0])) - D.dot(W)
    
    else:
        raise ValueError('Unknown normalization type.')

    return L



def spectral_clustering(L, chosen_eig_indices, num_classes=2):
    """
    :param L: Graph Laplacian (standard or normalized)
    :param chosen_eig_indices: indices of eigenvectors to use for clustering
    :param num_classes: number of clusters to compute (defaults to 2)
    :return: Y: Cluster assignments
    """

    """
    Use the function scipy.linalg.eig or the function scipy.sparse.linalg.eigs to compute:
    U = (n x n) eigenvector matrix           (sorted)
    E = (n x n) eigenvalue diagonal matrix   (sorted)
    """
    # Number of eigenvectors to compute by scipy.sparse.linalg.eigsh
    n_eigenvectors = max(chosen_eig_indices) + 1

    # Find the first eigenvectors and eigenvalues of L
    E, U = eigsh(L, k=n_eigenvectors, which='SM')    

    # Consider only the chosen eigenvectors to cluster 
    U_chosen = U[:, chosen_eig_indices]

    """
    compute the clustering assignment from the eigenvectors        
    Y = (n x 1) cluster assignments [0,1,...,c-1]                   
    """
    # Apply k-means to find the clustering assignment
    Y = KMeans(num_classes).fit_predict(U_chosen)

    return Y


def two_blobs_clustering():
    """
    TO BE COMPLETED
    Clustering of two blobs. Used in questions 2.1 and 2.2
    """

    # Get data and compute number of classes
    X, Y = blobs(600, n_blobs=2, blob_var=0.15, surplus=0)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 20
    var = 1.0  # exponential_euclidean's sigma^2

    laplacian_normalization = 'unn'
    
    # indices of the ordered eigenvalues to pick
    chosen_eig_indices = [1]

    # build laplacian
    W = build_similarity_graph(X, var=var, k=k)
    L = build_laplacian(W, laplacian_normalization)

    # run spectral clustering
    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)

    # Plot results
    plot_clustering_result(X, Y, L, Y_rec, KMeans(num_classes).fit_predict(X))


def choose_eigenvalues(eigenvalues, threshold=0.9):
    """
    Function to choose the indices of which eigenvalues to use for clustering.
    :param eigenvalues: sorted eigenvalues (in ascending order)
    :param threshold: thresholg value to choose the first significant gap between the eigenvalues
    :return: indices of the eigenvalues to use
    """

    # Compute the opposite of the second derivative approximation for the eigenvalues 
    eig_d2 = - np.diff(np.diff(eigenvalues))

    # Normalize it
    eig_d2 = eig_d2 / np.max(np.abs(eig_d2))

    # Find the minimum index where the value is higher than a threshold
    max_gap_idx = np.argwhere(eig_d2 > threshold)[0][0]
    max_gap_idx = max(max_gap_idx, 1)

    # Return the eigenvectors [1, ..., max_gap_idx]
    eig_ind = np.arange(1, max_gap_idx + 1)
    print(eig_ind)
    
    return eig_ind


def spectral_clustering_adaptive(L, num_classes=2, max_eigen_vectors=20, threshold=0.3):
    """
    Spectral clustering that adaptively chooses which eigenvalues to use.
    :param L: Graph Laplacian (standard or normalized)
    :param num_classes: number of clusters to compute (defaults to 2)
    :param max_eigen_vectors: maximum number of eigenvectors to compute
    :param threshold: parameter for the eigengap heuristic
    :return: Y: Cluster assignments
    """

    """
    Use the function scipy.linalg.eig or the function scipy.linalg.eigs to compute:
    U = (n x n) eigenvector matrix           (sorted)
    E = (n x n) eigenvalue diagonal matrix   (sorted)
    """
    # Number of eigenvectors to compute by scipy.sparse.linalg.eigsh
    n_eigenvectors = max(num_classes, max_eigen_vectors)

    # Find the first eigenvectors and eigenvalues of L
    E, U = eigsh(L, k=n_eigenvectors, which='SM')

    # Choose the eigenvectors indices with choose_eigenvalues
    chosen_eig_indices = choose_eigenvalues(E, threshold=threshold)

    # Consider only the chosen eigenvectors to cluster 
    U_chosen = U[:, chosen_eig_indices]

    """
    compute the clustering assignment from the eigenvectors   
    Y = (n x 1) cluster assignments [1,2,...,c]                   
    """
    # Apply k-means to find the clustering assignment
    Y = KMeans(num_classes).fit_predict(U_chosen)

    return Y


def find_the_bend():
    """
    TO BE COMPLETED
    Used in question 2.3
    :return:
    """

    # the number of samples to generate
    num_samples = 600

    # Generate blobs and compute number of clusters
    X, Y = blobs(num_samples, 4, 0.20)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 200
    eps = 0.4
    var = 1.0  # exponential_euclidean's sigma^2
    laplacian_normalization = 'unn'  # either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization
    threshold = 0.5  # For the function choose_eigenvalues()

    # build laplacian
    W = build_similarity_graph(X, var=var, k=k)  # knn graph
    # W = build_similarity_graph(X, var=var, eps=eps)  # eps graph
    L = build_laplacian(W, laplacian_normalization)

    """
    compute first 15 eigenvalues and call choose_eigenvalues() to choose which ones to use. 
    """
    eigenvalues, _ = eigsh(L, k=15, which='SM')
    chosen_eig_indices = choose_eigenvalues(eigenvalues, threshold=threshold)  # indices of the ordered eigenvalues to pick
    print(chosen_eig_indices)

    """
    compute spectral clustering solution using a non-adaptive method first, and an adaptive one after (see handout) 
    Y_rec = (n x 1) cluster assignments [0,1,..., c-1]    
    """
    # run spectral clustering
    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)
    Y_rec_adaptive = spectral_clustering_adaptive(L, num_classes=num_classes)

    plot_the_bend(X, Y, L, Y_rec, eigenvalues)


def two_moons_clustering():
    """
    TO BE COMPLETED.
    Used in question 2.7
    """
    # Generate data and compute number of clusters
    X, Y = two_moons(600)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 10
    var = 1.0  # exponential_euclidean's sigma^2

    laplacian_normalization = 'unn'
    chosen_eig_indices = [1]    # indices of the ordered eigenvalues to pick


    # build laplacian
    W = build_similarity_graph(X, var=var, k=k)
    L = build_laplacian(W, laplacian_normalization)
    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)
    plot_clustering_result(X, Y, L, Y_rec, KMeans(num_classes).fit_predict(X))


def point_and_circle_clustering():
    """
    TO BE COMPLETED.
    Used in question 2.8
    """
    # Generate data and compute number of clusters
    X, Y = point_and_circle(600, sigma=0.3)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 15
    eps = 0.7
    var = 1.0  # exponential_euclidean's sigma^2

    chosen_eig_indices = [1]    # indices of the ordered eigenvalues to pick


    # build laplacian
    # W = build_similarity_graph(X, var=var, k=k)  # knn graph
    W = build_similarity_graph(X, var=var, eps=eps)  # eps graph
    L_unn = build_laplacian(W, 'unn')
    L_norm = build_laplacian(W, 'sym')

    Y_unn = spectral_clustering(L_unn, chosen_eig_indices, num_classes=num_classes)
    Y_norm = spectral_clustering(L_norm, chosen_eig_indices, num_classes=num_classes)

    plot_clustering_result(X, Y, L_unn, Y_unn, Y_norm, 1)


def parameter_sensitivity():
    """
    TO BE COMPLETED.
    A function to test spectral clustering sensitivity to parameter choice.
    Used in question 2.9
    """
    # the number of samples to generate
    num_samples = 500

    """
    Choose parameters
    """
    var = 1.0  # exponential_euclidean's sigma^2
    laplacian_normalization = 'unn'
    chosen_eig_indices = [1]

    """
    Choose candidate parameters
    """
    parameter_candidate = np.linspace(0.0, 1.0, 100)  # the number of neighbours for the graph or the epsilon threshold
    parameter_performance = []

    for eps in parameter_candidate:
        # Generate data
        X, Y = two_moons(num_samples, 1, 0.1)
        num_classes = len(np.unique(Y))

        W = build_similarity_graph(X, eps=eps)
        L = build_laplacian(W, laplacian_normalization)

        Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes)

        parameter_performance += [skm.adjusted_rand_score(Y, Y_rec)]

    plt.figure()
    plt.plot(parameter_candidate, parameter_performance)
    plt.title('parameter sensitivity for $\epsilon$')
    plt.show()

if __name__=='__main__':
    parameter_sensitivity()