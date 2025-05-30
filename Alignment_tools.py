import ot
import numpy as np
import anndata as ad
from anndata import AnnData
import torch
import scipy.sparse
from sklearn.neighbors import NearestNeighbors



# Functions for the PASTE algorithm
# Credit to https://github.com/raphael-group/paste

def my_fused_gromov_wasserstein(M, C1, C2, p, q, G_init=None, loss_fun='square_loss', alpha=0.5, log=False,
                                numItermax=200, use_gpu=False, **kwargs):
    """
    Adapted fused_gromov_wasserstein with the added capability of defining a G_init (inital mapping).
    Also added capability of utilizing different POT backends to speed up computation.

    For more info, see: https://pythonot.github.io/gen_modules/ot.gromov.html
    """

    p, q = ot.utils.list_to_array(p, q)

    p0, q0, C10, C20, M0 = p, q, C1, C2, M
    nx = ot.backend.get_backend(p0, q0, C10, C20, M0)

    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)

    if G_init is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = (1 / nx.sum(G_init)) * G_init
        if use_gpu:
            G0 = G0.cuda()

    def f(G):
        return ot.gromov.gwloss(constC, hC1, hC2, G)

    def df(G):
        return ot.gromov.gwggrad(constC, hC1, hC2, G)

    if log:
        res, log = ot.optim.cg(p, q, (1 - alpha) * M, alpha, f, df, G0, C1=C1, C2=C2, constC=constC, log=True, **kwargs)

        fgw_dist = log['loss'][-1]

        log['fgw_dist'] = fgw_dist
        log['u'] = log['u']
        log['v'] = log['v']
        return res, log

    else:
        return ot.optim.cg(p, q, (1 - alpha) * M, alpha, f, df, G0, C1=C1, C2=C2, constC=constC, **kwargs)



def intersect(lst1, lst2):
    """
    Gets and returns intersection of two lists.
    Args:
        lst1: List
        lst2: List
    Returns:
        lst3: List of common elements.
    """

    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

## Covert a sparse matrix into a dense np array
to_dense_array = lambda X: X.toarray() if isinstance(X,scipy.sparse.csr.spmatrix) else np.array(X)

## Returns the data matrix or representation
extract_data_matrix = lambda adata,rep: adata.X if rep is None else adata.obsm[rep]


def kl_divergence_backend(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.
    Takes advantage of POT backend to speed up computation.
    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)
    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X,Y)

    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i',X,log_X)
    X_log_X = nx.reshape(X_log_X,(1,X_log_X.shape[0]))
    D = X_log_X.T - nx.dot(X,log_Y.T)
    return nx.to_numpy(D)



def pairwise_align_paste(
        sliceA,
        sliceB,
        alpha=0.1,
        dissimilarity='kl',
        use_rep=None,
        G_init=None,
        a_distribution=None,
        b_distribution=None,
        norm=False,
        numItermax=200,
        backend=ot.backend.NumpyBackend(),
        use_gpu=False,
        return_obj=False,
        verbose=False,
        gpu_verbose=False,
        coor_key="spatial",
        **kwargs):
    """
    Calculates and returns optimal alignment of two slices.

    Args:
        sliceA: Slice A to align.
        sliceB: Slice B to align.
        alpha:  Alignment tuning parameter. Note: 0 <= alpha <= 1.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        use_rep: If ``None``, uses ``slice.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``slice.obsm[use_rep]``.
        G_init (array-like, optional): Initial mapping to be used in FGW-OT, otherwise default is uniform mapping.
        a_distribution (array-like, optional): Distribution of sliceA spots, otherwise default is uniform.
        b_distribution (array-like, optional): Distribution of sliceB spots, otherwise default is uniform.
        numItermax: Max number of iterations during FGW-OT.
        norm: If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        backend: Type of backend to run calculations. For list of backends available on system: ``ot.backend.get_backend_list()``.
        use_gpu: If ``True``, use gpu. Otherwise, use cpu. Currently we only have gpu support for Pytorch.
        return_obj: If ``True``, additionally returns objective function output of FGW-OT.
        verbose: If ``True``, FGW-OT is verbose.
        gpu_verbose: If ``True``, print whether gpu is being used to user.
        coor_key: The key of spatial coordinates, ``spatial`` for default.

    Returns:
        - Alignment of spots.
        If ``return_obj = True``, additionally returns:

        - Objective function output of FGW-OT.
    """

    # Determine if gpu or cpu is being used
    if use_gpu:
        try:
            import torch
        except:
            print("We currently only have gpu support for Pytorch. Please install torch.")

        if isinstance(backend, ot.backend.TorchBackend):
            if torch.cuda.is_available():
                if gpu_verbose:
                    print("gpu is available, using gpu.")
            else:
                if gpu_verbose:
                    print("gpu is not available, resorting to torch cpu.")
                use_gpu = False
        else:
            print(
                "We currently only have gpu support for Pytorch, please set backend = ot.backend.TorchBackend(). Reverting to selected backend cpu.")
            use_gpu = False
    else:
        if gpu_verbose:
            print("Using selected backend cpu. If you want to use gpu, set use_gpu = True.")

    # subset for common genes
    common_genes = intersect(sliceA.var.index, sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]

    # Backend
    nx = backend

    # Calculate spatial distances
    coordinatesA = sliceA.obsm[coor_key].copy()
    coordinatesA = nx.from_numpy(coordinatesA)
    coordinatesB = sliceB.obsm[coor_key].copy()
    coordinatesB = nx.from_numpy(coordinatesB)

    if isinstance(nx, ot.backend.TorchBackend):
        coordinatesA = coordinatesA.float()
        coordinatesB = coordinatesB.float()
    D_A = ot.dist(coordinatesA, coordinatesA, metric='euclidean')
    D_B = ot.dist(coordinatesB, coordinatesB, metric='euclidean')

    if isinstance(nx, ot.backend.TorchBackend) and use_gpu:
        D_A = D_A.cuda()
        D_B = D_B.cuda()

    # Calculate expression dissimilarity
    A_X, B_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceA, use_rep))), nx.from_numpy(
        to_dense_array(extract_data_matrix(sliceB, use_rep)))

    if isinstance(nx, ot.backend.TorchBackend) and use_gpu:
        A_X = A_X.cuda()
        B_X = B_X.cuda()

    if dissimilarity.lower() == 'euclidean' or dissimilarity.lower() == 'euc':
        M = ot.dist(A_X, B_X)
    else:
        s_A = A_X + 0.01
        s_B = B_X + 0.01
        M = kl_divergence_backend(s_A, s_B)
        M = nx.from_numpy(M)

    if isinstance(nx, ot.backend.TorchBackend) and use_gpu:
        M = M.cuda()

    # init distributions
    if a_distribution is None:
        a = nx.ones((sliceA.shape[0],)) / sliceA.shape[0]
    else:
        a = nx.from_numpy(a_distribution)

    if b_distribution is None:
        b = nx.ones((sliceB.shape[0],)) / sliceB.shape[0]
    else:
        b = nx.from_numpy(b_distribution)

    if isinstance(nx, ot.backend.TorchBackend) and use_gpu:
        a = a.cuda()
        b = b.cuda()

    if norm:
        D_A /= nx.min(D_A[D_A > 0])
        D_B /= nx.min(D_B[D_B > 0])

    # Run OT
    if G_init is not None:
        G_init = nx.from_numpy(G_init)
        if isinstance(nx, ot.backend.TorchBackend):
            G_init = G_init.float()
            if use_gpu:
                G_init.cuda()
    pi, logw = my_fused_gromov_wasserstein(M, D_A, D_B, a, b, G_init=G_init, loss_fun='square_loss', alpha=alpha,
                                           log=True, numItermax=numItermax, verbose=verbose, use_gpu=use_gpu)
    pi = nx.to_numpy(pi)
    obj = nx.to_numpy(logw['fgw_dist'])
    if isinstance(backend, ot.backend.TorchBackend) and use_gpu:
        torch.cuda.empty_cache()

    if return_obj:
        return pi, obj
    return pi