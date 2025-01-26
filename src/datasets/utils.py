import multiprocessing

from scipy.spatial.distance import cdist
import numpy as np


def parallel_unsigned_distances(cc_cords, point_cloud, n_jobs):
    """
    Split the cc_cords array into n_jobs chunks, compute min pairwise distances in parallel, and then concat results
    """
    chunk_size = len(cc_cords) // n_jobs
    cc_cords_chunks = [cc_cords[i:i + chunk_size] for i in range(0, len(cc_cords), chunk_size)]

    with multiprocessing.Pool(n_jobs) as pool:
        results = pool.starmap(cdist, [(cc_chunk.astype(np.float32), point_cloud.astype(np.float32))
                                       for cc_chunk in cc_cords_chunks])

    pairwise_dists = np.concatenate(results, axis=0)  # Shape (N_vol**3, N_points)
    unsigned_distances = np.min(pairwise_dists, axis=1)  # Shape (N_vol**3,)

    assert len(unsigned_distances) == len(cc_cords), f'Unexpected length: {len(unsigned_distances)}'

    return unsigned_distances