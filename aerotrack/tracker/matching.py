import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from aerotrack.tracker import kalman_filter
import time
from loguru import logger

try:
    from aerotrack.tracker.cython_nwd import nwd_distance_c
except ImportError as e:
    logger.warning(f"Cython NWD non trouvé ({e}). Utilisation de NumPy (plus lent).")
    nwd_distance_c = None

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    """
    Fuses the tracking cost matrix with detection scores.
    """
    # 1. Safely bypass completely empty frames
    if cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
        return cost_matrix
        
    # 2. Convert Cost to Similarity (1.0 is a perfect match)
    sim_matrix = 1.0 - cost_matrix
        
    # 3. Extract detection confidence scores
    det_scores = np.array([det.score for det in detections])
    
    # 4. Apply Fusion (Confident detections keep similarity high)
    # If this crashes, it means byte_tracker.py passed the wrong 'detections' list!
    sim_matrix *= det_scores
    
    # 5. Convert back to Tracking Cost (0.0 is a perfect match)
    fuse_cost = 1.0 - sim_matrix
    
    return fuse_cost

def nwd_distance(atracks, btracks, C=12.8):
    if (len(atracks) == 0 or len(btracks) == 0):
        return np.zeros((len(atracks), len(btracks)))

    tlwhs_a = np.array([track.tlwh for track in atracks], dtype=np.float64)
    tlwhs_b = np.array([track.tlwh for track in btracks], dtype=np.float64)

    # Exécution ultra-rapide si compilé
    if nwd_distance_c is not None:
        return nwd_distance_c(tlwhs_a, tlwhs_b, C)

    # Repli NumPy classique si échec (Plan B)
    cx_a = tlwhs_a[:, 0] + tlwhs_a[:, 2] / 2.0
    cy_a = tlwhs_a[:, 1] + tlwhs_a[:, 3] / 2.0
    w_half_a = tlwhs_a[:, 2] / 2.0
    h_half_a = tlwhs_a[:, 3] / 2.0

    cx_b = tlwhs_b[:, 0] + tlwhs_b[:, 2] / 2.0
    cy_b = tlwhs_b[:, 1] + tlwhs_b[:, 3] / 2.0
    w_half_b = tlwhs_b[:, 2] / 2.0
    h_half_b = tlwhs_b[:, 3] / 2.0

    cx_a, cy_a = cx_a[:, np.newaxis], cy_a[:, np.newaxis]
    w_half_a, h_half_a = w_half_a[:, np.newaxis], h_half_a[:, np.newaxis]

    cx_b, cy_b = cx_b[np.newaxis, :], cy_b[np.newaxis, :]
    w_half_b, h_half_b = w_half_b[np.newaxis, :], h_half_b[np.newaxis, :]

    W2_sq = (cx_a - cx_b)**2 + (cy_a - cy_b)**2 + (w_half_a - w_half_b)**2 + (h_half_a - h_half_b)**2
    nwd = np.exp(-np.sqrt(W2_sq) / C)

    return 1.0 - nwd

