import numpy as np 
from numpy.linalg import solve, norm
from scipy import *
from scipy.sparse import *
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from scipy.stats import poisson
import scipy.optimize as op
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans



def spring_rank(A, return_correlation=True):
    """Given adjacency matrix A of a directed unweighted graph, 
    find the ranking of the nodes to minimize the potential energy of the network,
    assuming spring constants k between all nodes are the same.
    Output: 
        s is the vector of rankings/scores
        Sigma is covariance matrix
    """
   
    d_out = np.sum(A, axis=1)
    d_in = np.sum(A, axis=0)
    D_out = np.diag(d_out)
    D_in = np.diag(d_in)
    M = D_out + D_in - A - np.transpose(A)
    b = d_out - d_in
    s = np.linalg.lstsq(M,b)[0]
    Sigma = np.linalg.pinv(M) #covariance matrix
    
    if return_correlation:
        return (s, pearson_correlation(Sigma))
    
    return (s, Sigma)



def spring_rank_sparse(A):
    """
    Given sparse adjacency matrix A of a directed weighted graph, 
    find the ranking of the nodes to minimize the potential energy of the network,
    assuming spring constants k between all nodes are the same.
    Output: 
        s is the vector of rankings/scores
        Sigma is the covariance matrix
    """
    
    d_out = np.asarray(lil_matrix.sum(A, axis=1)).flatten()
    d_in = np.asarray(lil_matrix.sum(A, axis=0)).flatten()
    D_out = dia_matrix((d_out ,0), shape=(d_out.size, d_out.size))
    D_in = dia_matrix((d_in, 0), shape=(d_in.size, d_in.size))
    M = D_out + D_in - A - np.transpose(A)
    b = d_out.T - d_in
    s = lsqr(M, b)[0]
    #Sigma = lsqr(M, np.repeat(0,d_in.size)) #TODO: still need to test this
    
    return s



def pearson_correlation(Sigma):
    """
    Input: Sigma covariance matrix
    Output: correlation matrix C calculated with Pearson Correlation formula
    """
    
    n = Sigma.shape[0]
    C = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if Sigma[i,i] < 10**-10 and Sigma[j,j] > 10**-10:
                C[i,j] = Sigma[i,j]/(np.sqrt(10**-10)*np.sqrt(Sigma[j,j]))
            elif Sigma[i,i] > 10**-10 and Sigma[j,j] < 10**-10: 
                C[i,j] = Sigma[i,j]/(np.sqrt(10**-10)*np.sqrt(Sigma[i,i]))
            elif Sigma[i,i] < 10**-10 and Sigma[j,j] < 10**-10:
                C[i,j] = Sigma[i,j]/(np.sqrt(10**-10)*np.sqrt(10**-10))
            else:
                C[i,j] = Sigma[i,j]/(np.sqrt(Sigma[i,i])*np.sqrt(Sigma[j,j]))
    return C
   
    
    
    
def generate_graph(s, beta=1, c=1, bernoulli=False):
    """
    Generate a graph given scores/rankings s
    Edges are drawn from a Poisson distribution, assuming all spring constants are 1
    Input: 
        s is scores
        beta is inversed temperature (noise)
        c is sparsity constant
        bernoulli use a Bernoulli distribution instead of Poisson
    Output:
        A is adjacency matrix where A[i,j] is the weight of an edge from node i to j
    """
    
    A = np.zeros((s.size, s.size))
    
    for i in range(s.size):
        for j in range(s.size):
            mu = np.exp(-0.5*beta*(s[i]-s[j]-1)**2)
            if bernoulli:
                A[i,j] = np.random.binomial(1, c*mu)
            else:
                A[i,j] = np.random.poisson(c*mu)
   
    return A
 



def estimate_beta(matrix, scores):
    """
    Use MLE to estimate inversed temperature beta from a network and its inferred rankings/scores
    Input:
        matrix: graph adjacency matrix where matrix[i,j] is the weight of an edge from node i to j
        scores: SpringRank inferred scores/rankings of nodes
    Output:
        beta: MLE estimate of inversed tempurature
    """
    
    def f(beta, matrix, scores):
        n = scores.size
        y = 0.
        
        for i in range(n):
            for j in range(n):
                d = scores[i] - scores[j]
                p = (1 + np.exp(-2 * beta * d))**(-1)
                y = y + d * (matrix[i,j] - (matrix[i,j] + matrix[j,i]) * p)
        
        return y
    
    beta_0 = 0.1
    b = op.fsolve(f, beta_0, args=(matrix, scores))
    
    return b
    

    

def estimate_c(avg_degree, beta, scores):
    """
    Input:
        avg_degree: average degree of the graph to be generated
        beta: inversed tempurature
        scores: rankings/scores of the nodes
    Output:
        c sparsity constant (to be given to generative model)
    """
    
    Z = 0.0
    n = scores.size
    
    for i in range(n):
        for j in range(n):
            Z = Z + np.exp( -0.5 * beta * (scores[i] - scores[j] - 1)**2)
            
    return (avg_degree*n)/Z




def plot(scores, correlation=None, labels=None, community=None, figsize=(6,6)):
    """
    Input:
        correlation: pearson correlation matrix
        scores: SR scores
    Output:
        heatmap of corralation, columns and rows ordered by scores
        density plot of scores
        
        These can be used to inform choice of cluster
    """
    
    inds_sort = np.argsort(scores)
    if community is None:
        #Plot correlation heatmap
        if correlation is not None:
            correlation = correlation[inds_sort,][:,inds_sort]
            plt.matshow(correlation, cmap='RdBu_r')
            plt.colorbar()
            print('Note: Nodes are ordered by scores (increasing) in orrelation matrix')


        #Plot scores density
        X = scores[:, np.newaxis]
        X_plot = np.linspace(np.min(X), np.max(X), 1000)[:, np.newaxis]
        fig, ax = plt.subplots()
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
        log_dens = kde.score_samples(X_plot) 
        ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
                label="kernel = '{0}'".format('gaussian'))


    #Plot scores with nodes labels
    fig1, ax1 = plt.subplots(figsize=figsize)
    if community is None:
        ax1.scatter(range(scores.size), scores[inds_sort], s=100)
    else:
        ax1.scatter(range(scores.size), scores[inds_sort], s=100, c=np.asarray(community)[inds_sort], cmap='Set1')
    if labels is not None:
        labels = np.asarray(labels)
        ax1.set_xticks(range(scores.size))
        ax1.set_xticklabels(labels[inds_sort], rotation='vertical')
        
    plt.show()
    
    
    
def test_ranks_significance(A, n_repetitions=100, plot=True):
    """
    Given an adjacency matrix, test if the hierarchical structure is statitically significant compared to null model
    The null model contains randomized directions of edges while preserving total degree between each pair of nodes
    Adapted from Dan Larremore's code: http://danlarremore.com
    
    Output:
        p-value: probability of observing the Hamiltonion energy lower than that of the real network if null model is true
        plot: histogram of energy of null models with dashed line as the energy of real network
    """
    
    if not scipy.sparse.issparse(A):
        A = scipy.sparse.csr_matrix(A)     #Convert to sparsed matrix for better efficiency
        
    #check if matrix contains integer values or not
    v = scipy.sparse.find(A)[2]
    if np.sum( np.mod(v,1)==0) == v.size:
        is_int = True
        Abar = A + A.transpose()
    else:
        is_int = False
        
    H = np.zeros(n_repetitions)
    H0 = spring_Hamiltonion(A, spring_rank_sparse(A))
    
    for i in range(n_repetitions):
        if is_int:
            B = randomize_edge_dir(Abar)
        else:
            B = randomize_edge_dir_none_int(A)
        H[i] = spring_Hamiltonion(B, spring_rank_sparse(B))
    
    p_val = np.sum(H<H0)/n_repetitions
    
    #Plot
    if plot:
        plt.hist(H)
        plt.axvline(x=H0, color='r', linestyle='dashed')
        plt.show()
    
    return p_val


def spring_Hamiltonion(A, scores):
    """
    Calculate the Hamiltonion of the system given an adjacency matrix
    """
    
    H = 0.0
        
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            H = H + 0.5*A[i,j]*(scores[i] - scores[j] - 1)**2
    
    return H
    

    
def randomize_edge_dir(Abar):
    """
    Randomize directions of edges while preserving the total in and out degree
    """
    
    try:
        n = Abar.shape[0]
        (r, c, v) = scipy.sparse.find(scipy.sparse.triu(Abar, 1))
        up = np.random.binomial((v*1).astype(int), 0.5)
        down = v - up
        data = np.concatenate((up, down))
        row_ind = np.concatenate((r, c))
        col_ind = np.concatenate((c, r))
        return scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n,n)) 
    
    except ValueError:
        print('Invalid type of matrix. Use scipy.sparse.crs_matrix for A')  

        
        
def randomize_edge_dir_none_int(A):
    """
    Randomize directions of edges while preserving the total in and out degree.
    Used when values in A are not integers
    """

    n = A.shape[0]
    (r, c, v) = scipy.sparse.find(A)
    for i in range(v.size):
        if np.random.rand > 0.5:
            temp = r[i]
            r[i] = c[i]
            c[i] = temp
    return scipy.sparse.csr_matrix((v, (r, c)), shape=(n,n)) 
    
    
    
def summary(A, data_name=None, data_type=None):
    """
    Return a tuple with summary statistics for a given adjacency matrix
    Output
        N: number of nodes
        H/m: energy per edge
        Depth: distance between highest and lowest ranked nodes
        p-value: probability of observing the Hamiltonion energy lower than that of the real network if null model is true
    """
    
    N = A.shape[0]
    m = np.sum(A)
    if not scipy.sparse.issparse(A):
        A = scipy.sparse.csr_matrix(A)
    scores = spring_rank_sparse(A)
    H = spring_Hamiltonion(A, scores=scores)
    depth = np.max(scores) - np.min(scores)
    viol, percent_viol = violation(A, scores)
    min_viol = min_violation(A)
    weighted_viol = weighted_violation(A, scores)
    p_val = test_ranks_significance(A) 
    
    return {'Data name': data_name, 'Data type': data_type, 'N': N, 'Energy per edge': H/m, 'Depth': depth, 'p-value': p_val, 'Viol.': viol, '% Viol.': percent_viol, 'Min. Viol.': min_viol, 'Weighted viol.': weighted_viol }   
  
    
    
def violation(A, scores):
    
    inds_sort = np.argsort(- scores)
    A_sort = A[inds_sort,][:,inds_sort]
    viol = scipy.sparse.tril(A_sort, -1).sum()
    m = A_sort.sum()
    return (viol, viol/m)
    

    
def min_violation(A):
    
    min_viol = 0
    for i in range(A.shape[0]):
        for j in range(i+1, A.shape[0]-1):
            if A[i,j] >0 and A[j,i] > 0:
                min_viol = min_viol + min(A[i,j], A[j,i])
    return min_viol



def weighted_violation(A, scores):
    r, c, v = scipy.sparse.find(A)
    s = (scores - min(scores))/(max(scores) - min(scores))
    wv = 0
    for i in range(len(v)):
        if s[r[i]] < s[c[i]]:
            wv = wv + v[i]*(s[c[i]] - s[r[i]])
            
    return wv
    
    
    
def community(A, n_clusters, c_type='tiers', used_correlation=True):
    """
    Input
        
        n_clusters: number of clusters to return
        
    """
    
    try:
        scores, correlation = spring_rank(A)
        
        #Infer tiers
        if c_type == 'tiers':
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(scores.reshape((scores.size,1)))
            clusters_scores = kmeans.labels_

            if used_correlation:
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(correlation)
                clusters_correlation = kmeans.labels_
                
                return (clusters_scores, clusters_correlation)

            return clusters_scores
    
    
        #Infer parallel sub-hierarchy
        if c_type == 'parallel':
            print('Generating null model ...')
            avg_degree = np.sum(A)/(A.shape[0])
            beta_hat = estimate_beta(matrix=A, scores=scores)
            c_hat = estimate_c(avg_degree=avg_degree, beta=beta_hat, scores=scores)
            null_model = generate_graph(s=scores, beta=beta_hat, c=c_hat)
            null_scores, null_correlation = spring_rank(null_model)
            print('Finding clusters ...')
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(correlation - null_correlation)
            clusters = kmeans.labels_
            print('Done')
            return clusters
        
    except ValueError:
        print('Use numpy.matrix instead of scipy.sparse.csr_matrix')


    
    
    
    
    
    