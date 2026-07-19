# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp

def nwd_distance_c(np.ndarray[np.float64_t, ndim=2] tlwhs_a, 
                   np.ndarray[np.float64_t, ndim=2] tlwhs_b, 
                   double C=12.8):
    """
    Calcule la matrice de coût NWD en C pur pour des performances maximales.
    """
    # Déclaration des types C pour optimiser la mémoire
    cdef int N = tlwhs_a.shape[0]
    cdef int M = tlwhs_b.shape[0]
    
    # Matrice de sortie pré-allouée
    cdef np.ndarray[np.float64_t, ndim=2] cost_matrix = np.zeros((N, M), dtype=np.float64)
    
    cdef int i, j
    cdef double cx_a, cy_a, w_half_a, h_half_a
    cdef double cx_b, cy_b, w_half_b, h_half_b
    cdef double W2_sq, nwd
    
    # Double boucle C (extrêmement rapide contrairement aux boucles Python)
    for i in range(N):
        # Pré-calcul pour la cible A (évite de le recalculer à chaque itération j)
        cx_a = tlwhs_a[i, 0] + tlwhs_a[i, 2] / 2.0
        cy_a = tlwhs_a[i, 1] + tlwhs_a[i, 3] / 2.0
        w_half_a = tlwhs_a[i, 2] / 2.0
        h_half_a = tlwhs_a[i, 3] / 2.0
        
        for j in range(M):
            # Calcul pour la cible B
            cx_b = tlwhs_b[j, 0] + tlwhs_b[j, 2] / 2.0
            cy_b = tlwhs_b[j, 1] + tlwhs_b[j, 3] / 2.0
            w_half_b = tlwhs_b[j, 2] / 2.0
            h_half_b = tlwhs_b[j, 3] / 2.0
            
            # W2 distance squared (Multiplication directe plus rapide que ** 2 en C)
            W2_sq = (cx_a - cx_b) * (cx_a - cx_b) + \
                    (cy_a - cy_b) * (cy_a - cy_b) + \
                    (w_half_a - w_half_b) * (w_half_a - w_half_b) + \
                    (h_half_a - h_half_b) * (h_half_a - h_half_b)
            
            # NWD via libc.math
            nwd = exp(-sqrt(W2_sq) / C)
            
            # Sauvegarde du coût
            cost_matrix[i, j] = 1.0 - nwd
            
    return cost_matrix