from ..get import _get_obs_rep

import numpy as np
from scipy import sparse


def spatial_connectivity(
    adata: "AnnData", obsm: str = "spatial", key_added: str = "spatial_connectivity",
):
    """Creates an adjacency matrix of wells for visium data.

    Params
    ------
    adata
        The AnnData object.
    obsm
        Key in obsm which stores coordinates of wells.
    key_added
        Key in obsp to add spatial graph as.

    Usage
    -----
    >>> adata = sc.datasets.visium_sge("V1_Adult_Mouse_Brain")
    >>> adata
    AnnData object with n_obs × n_vars = 2698 × 31053
    obs: 'in_tissue', 'array_row', 'array_col'
    var: 'gene_ids', 'feature_types', 'genome'
    uns: 'spatial'
    obsm: 'spatial'
    >>> sc.pp.spatial_connectivity(adata)
    >>> adata
    AnnData object with n_obs × n_vars = 2698 × 31053
    obs: 'in_tissue', 'array_row', 'array_col'
    var: 'gene_ids', 'feature_types', 'genome'
    uns: 'spatial'
    obsm: 'spatial'
    obsp: 'spatial_connectivity'
    """
    adata.obsp[key_added] = _spatial_connectivity(_get_obs_rep(adata, obsm=obsm))


def _spatial_connectivity(coords: np.ndarray):
    """
    Given the coordinates of hex based cells from a visium experiment, this returns an adjacency matrix for those cells.
    
    Usage
    -----
    >>> adata.obsp["spatial_connectivity"] = spatial_connectivity(adata.obsm["spatial"])
    """
    from sklearn.neighbors import NearestNeighbors

    N = coords.shape[0]
    dists, row_indices = (
        x.reshape(-1)
        for x in NearestNeighbors(n_neighbors=6, metric="euclidean")
        .fit(coords)
        .kneighbors()
    )
    col_indices = np.repeat(np.arange(N), 6)
    dist_cutoff = np.median(dists) * 1.3  # There's a small amount of sway
    mask = dists < dist_cutoff
    return sparse.csr_matrix(
        (np.ones(mask.sum()), (row_indices[mask], col_indices[mask])), shape=(N, N)
    )
