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
import networkx as nx



def spring_rank(A, return_covariance=True):
    """
    Given adjacency matrix A of a directed unweighted graph, 
    find the ranking of the nodes to minimize the potential energy of the network,
    assuming spring constants k between all nodes are the same.
    Input:
        A: adjacency matrix where A[i,j] is the weight of an edge from node i to j
    Output: 
        scores: the vector of rankings/scores
        cov_matrix: covariance matrix
    """
   
    d_out = np.sum(A, axis=1) #out degree sequence
    d_in = np.sum(A, axis=0) #in degree sequence
    D_out = np.diag(d_out) 
    D_in = np.diag(d_in)
    M = D_out + D_in - A - np.transpose(A)
    b = d_out - d_in
    scores = np.linalg.lstsq(M,b)[0]
    cov_matrix = np.linalg.pinv(M) #covariance matrix
    
    if return_covariance:
        return (scores, cov_matrix)
         
    return (scores, pearson_correlation(cov_matrix))



def spring_rank_sparse(A):
    """
    Given sparse adjacency matrix A of a directed weighted graph, 
    find the ranking of the nodes to minimize the potential energy of the network,
    assuming spring constants k between all nodes are the same.
    Use spring_rank_sparse when A is large for efficient computation
    Input:
        A: adjacency matrix where A[i,j] is the weight of an edge from node i to j
    Output: 
        scores: the vector of rankings/scores
    """
    
    d_out = np.asarray(lil_matrix.sum(A, axis=1)).flatten()
    d_in = np.asarray(lil_matrix.sum(A, axis=0)).flatten()
    D_out = dia_matrix((d_out ,0), shape=(d_out.size, d_out.size))
    D_in = dia_matrix((d_in, 0), shape=(d_in.size, d_in.size))
    M = D_out + D_in - A - np.transpose(A)
    b = d_out.T - d_in
    scores = lsqr(M, b)[0]
    #cov_matrix = lsqr(M, np.repeat(0,d_in.size)) #TODO: still need to test this
    
    return scores



def pearson_correlation(cov_matrix):
    """
    Given a covariance matrix, convert it into a correlation matrix
    Input: 
        cov_matrix: a covariance matrix
    Output: 
        cor_matrix: correlation matrix calculated with Pearson Correlation formula
    """
    
    n = cov_matrix.shape[0]
    cor_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if cov_matrix[i,i] < 10**-10 and cov_matrix[j,j] > 10**-10:
                cor_matrix[i,j] = cov_matrix[i,j]/(np.sqrt(10**-10)*np.sqrt(cov_matrix[j,j]))
            elif cov_matrix[i,i] > 10**-10 and cov_matrix[j,j] < 10**-10: 
                cor_matrix[i,j] = cov_matrix[i,j]/(np.sqrt(10**-10)*np.sqrt(cov_matrix[i,i]))
            elif cov_matrix[i,i] < 10**-10 and cov_matrix[j,j] < 10**-10:
                cor_matrix[i,j] = cov_matrix[i,j]/(np.sqrt(10**-10)*np.sqrt(10**-10))
            else:
                cor_matrix[i,j] = cov_matrix[i,j]/(np.sqrt(cov_matrix[i,i])*np.sqrt(cov_matrix[j,j]))
    return cor_matrix
   
    
    
def generate_graph(scores, beta=1, c=1, bernoulli=False):
    """
    Generate a graph given scores/rankings s
    Edges are drawn from a Poisson distribution, assuming all spring constants are 1
    Input: 
        scores: SpringRank scores
        beta: inversed temperature (noise)
        c: sparsity constant
        bernoulli means use a Bernoulli distribution instead of Poisson
    Output:
        A: an adjacency matrix where A[i,j] is the weight of an edge from node i to j
    """
    
    A = np.zeros((scores.size, scores.size))
    
    for i in range(scores.size):
        for j in range(scores.size):
            mu = np.exp(-0.5*beta*(scores[i]-scores[j]-1)**2)
            if bernoulli:
                A[i,j] = np.random.binomial(1, c*mu)
            else:
                A[i,j] = np.random.poisson(c*mu)
   
    return A
 


def estimate_beta(A, scores):
    """
    Use Maximum Likelihood Estimate to estimate inversed temperature beta from a network and its inferred rankings/scores
    Input:
        A: graph adjacency matrix where matrix[i,j] is the weight of an edge from node i to j
        scores: SpringRank inferred scores/rankings of nodes
    Output:
        beta: MLE estimate of inversed tempurature
    """
    
    def f(beta, A, scores):
        n = scores.size
        y = 0.
        
        for i in range(n):
            for j in range(n):
                d = scores[i] - scores[j]
                p = (1 + np.exp(-2 * beta * d))**(-1)
                y = y + d * (A[i,j] - (A[i,j] + A[j,i]) * p)
        
        return y
    
    beta_0 = 0.1
    beta_est = op.fsolve(f, beta_0, args=(A, scores))
    
    return beta_est

    

def estimate_c(avg_degree, beta, scores):
    """
    Estimate sparcity parameter c given average degree, beta and SpringRank scores
    Input:
        avg_degree: average degree of the graph to be generated
        beta: inversed tempurature
        scores: rankings/scores of the nodes
    Output:
        c: sparsity constant (to be given to generative model)
    """
    
    Z = 0.0
    n = scores.size
    
    for i in range(n):
        for j in range(n):
            Z = Z + np.exp( -0.5 * beta * (scores[i] - scores[j] - 1)**2)
            
    return (avg_degree*n)/Z



def plot(scores, cor_matrix=None, labels=None, community=None, figsize=(6,6)):
    """
    Create general summary plots for SpringRank results including
        Heatmap of correlation matrix
        Density plot of SpringRank's scores distribution using Gaussian Kernel Density approximation
        These two plots can be used to inform choice of cluster
        Scatter plot of nodes (with labels on x-axis) against SpringRank scores (on y-axis)
        If community labels are given, nodes are colored based on their community in scatter plot
    Input:
        scores: SpringRank scores
        cor_matrix: pearson correlation matrix
        labels: nodes' labels
        community: community classifications of nodes
        figsize: tuple of (width, height) of output figure
    Output:
        Heatmap of corralation, columns and rows ordered by scores
        Density plot of scores
        Scatter plot
    """
    
    inds_sort = np.argsort(scores)
    if community is None:
        #Plot correlation heatmap
        if cor_matrix is not None:
            #sort correlation matrix by node's scores, ascending
            cor_matrix = cor_matrix[inds_sort,][:,inds_sort]
            plt.matshow(cor_matrix, cmap='RdBu_r')
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
    
    
    
def test_ranks_significance(A, n_repetitions=100, plot=True, model='SR-null'):
    """
    Given an adjacency matrix, test if the hierarchical structure is statitically significant compared to null model
    The null model contains randomized directions of edges while preserving total degree between each pair of nodes
    Adapted from Dan Larremore's code: http://danlarremore.com
    Input:
        A: graph adjacency matrix where matrix[i,j] is the weight of an edge from node i to j
        n_repetitions: number of null models to generate
        plot: if True shows histogram of null models' energy distribution
        model: 
            'SR_null' is from the original paper, edges are preserved but their directions randomized
            'configuration' preserves degree distributions, randomizes connections
    Output:
        p-value: probability of observing the Hamiltonion energy lower than that of the real network if null model is true
        plot: histogram of energy of null models with dashed line as the energy of real network
    """
    
    #convert to sparsed matrix for better efficiency
    A_spr = scipy.sparse.csr_matrix(A)     
        
    #check if matrix contains integer values or not
    v = scipy.sparse.find(A_spr)[2]
    if np.sum( np.mod(v,1)==0) == v.size:
        is_int = True
        Abar = A_spr + A_spr.transpose()
    else:
        is_int = False
    
    #place holder for outputs    
    H = np.zeros(n_repetitions)
    H0 = spring_Hamiltonion(A_spr, spring_rank_sparse(A_spr))
    
    #generate null models
    for i in range(n_repetitions):
        if is_int:
            if model == 'SR-null':
                B = randomize_edge_dir(Abar)
            elif model == 'configuration':
                B = configuration_model(A)
            else:
                print('Value Error: use valid model name: "SR-null", "configuration"') 
        else:
            if model != 'SR-null':
                print('Warning: used "SR-null", other models only works with integer valued edges')
            B = randomize_edge_dir_none_int(A_spr)
        #calculate energy of each null model
        H[i] = spring_Hamiltonion(B, spring_rank_sparse(B))
    
    p_val = np.sum(H<H0)/n_repetitions
    
    #Plot
    if plot:
        plt.hist(H)
        plt.axvline(x=H0, color='r', linestyle='dashed')
        plt.show()
    
    return (p_val, H)



def spring_Hamiltonion(A, scores):
    """
    Calculate the Hamiltonion of the network
    Input:
        A: graph adjacency matrix where matrix[i,j] is the weight of an edge from node i to j
        scores: SpringRank scores
    Output:
        H: Hamiltonion energy of the system
    """
    
    H = 0.0
        
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            H = H + 0.5*A[i,j]*(scores[i] - scores[j] - 1)**2
    
    return H
    

    
def randomize_edge_dir(Abar):
    """
    Randomize directions of edges while preserving the total degree to create SR null model
    Input:
        Abar: graph adjacency matrix where Abar[i,j] is the total weight of connections between i and j
    Output:
        An adjacency matrix representing a null model
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
    Randomize directions of edges while preserving the total degree.
    Used when values in A are not integers
    Input:
        A: graph adjacency matrix where A[i,j] is the weight of an edge from node i to j
    Output:
        An adjacency matrix representing a null model
    """

    n = A.shape[0]
    (r, c, v) = scipy.sparse.find(A)
    for i in range(v.size):
        if np.random.rand > 0.5:
            temp = r[i]
            r[i] = c[i]
            c[i] = temp
    return scipy.sparse.csr_matrix((v, (r, c)), shape=(n,n)) 



def configuration_model(A):
    """
    Create a new graph by preserving the in and out degree sequences
    but randomizing the connections
    Only used when values in A are integers
    Input: 
        A: graph adjacency matrix where A[i,j] is the weight of an edge from node i to j
    Output:
        An adjacency matrix of a null model
    """
    in_degree_seq = list(np.sum(A, axis=0).real.astype(np.int64)) #TODO: make this work with sparse matrix?
    out_degree_seq = list(np.sum(A, axis=1).real.astype(np.int64))
    config_model = nx.directed_configuration_model(in_degree_sequence=in_degree_seq, out_degree_sequence=out_degree_seq)
    return scipy.sparse.csr_matrix(nx.to_numpy_matrix(config_model))



def summary(A, data_name=None, data_type=None):
    """
    Return a tuple with summary statistics for a given adjacency matrix
    Adapted from Daniel Larremore's MATLAB code: http://danlarremore.com
    Input:
        A: graph adjacency matrix where A[i,j] is the weight of an edge from node i to j
        data_name: name of data set, optional
        data_type: type of network, optional
    Output
        N: number of nodes
        H/m: energy per edge
        Depth: distance between scores of the highest and lowest ranked nodes
        p-value: probability of observing the Hamiltonion energy lower than that of the real network if null model is true
        Viol: number of edges going from a lower ranked node to a higher ranked one
        % Viol: proportion of edges that are violations
        Min. Viol: minimum number of violations given the graph for all possible rankings
        Weighted viol: number of violations weighted by difference in scores
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
    p_val, temp = test_ranks_significance(A) 
    
    return {'Data name': data_name, 'Data type': data_type, 'N': N, 'Energy per edge': H/m, 'Depth': depth, 'p-value': p_val, 'Viol.': viol, '% Viol.': percent_viol, 'Min. Viol.': min_viol, 'Weighted viol.': weighted_viol }   
  
    
    
def violation(A, scores):
    """
    Calculate number of violations in a graph given SpringRank scores
    A violaton is an edge going from a lower ranked node to a higher ranked one
    Input:
        A: graph adjacency matrix where A[i,j] is the weight of an edge from node i to j
        scores: SpringRank scores
    Output:
        number of violations
        proportion of all edges that are violations
    """
    
    inds_sort = np.argsort(- scores)
    A_sort = A[inds_sort,][:,inds_sort]
    viol = scipy.sparse.tril(A_sort, -1).sum()
    m = A_sort.sum()
    return (viol, viol/m)
    

    
def min_violation(A):
    """
    Calculate the minimum number of violations in a graph for all possible rankings
    A violaton is an edge going from a lower ranked node to a higher ranked one
    Input:
        A: graph adjacency matrix where A[i,j] is the weight of an edge from node i to j
    Output:
        minimum number of violations
    """
    
    min_viol = 0
    for i in range(A.shape[0]):
        for j in range(i+1, A.shape[0]-1):
            if A[i,j] >0 and A[j,i] > 0:
                min_viol = min_viol + min(A[i,j], A[j,i])
    return min_viol



def weighted_violation(A, scores):
    """
    Calculate number of violations in a graph given SpringRank scores 
    A violaton is an edge going from a lower ranked node to a higher ranked one
        weighted by the difference between these two nodes
    Input:
        A: graph adjacency matrix where A[i,j] is the weight of an edge from node i to j
        scores: SpringRank scores
    Output:
        number of violations
        proportion of all edges that are violations
    """
    r, c, v = scipy.sparse.find(A)
    s = (scores - min(scores))/(max(scores) - min(scores))
    wv = 0
    for i in range(len(v)):
        if s[r[i]] < s[c[i]]:
            wv = wv + v[i]*(s[c[i]] - s[r[i]])
            
    return wv
        
