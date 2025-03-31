# -*- coding: utf-8 -*-
import numpy as np
import random
from functools import partial


def generate_list_for_matrice(n):
    '''
    Parameters
    ----------
    n : int

    Returns
    -------
    list
        return a list containing n times each integer from 1 to n

    '''
    return [i for i in range(1, n + 1) for _ in range(n)]

def cofactor_determinants_numpy(matrice):
    '''   
    Parameters
    ----------
    matrice : np.array
        n x n matrix

    Returns
    -------
    det_matrix : np.array
        n x n matrix containing the determinants of the co matrices

    '''
    n = matrice.shape[0]
    det_matrice = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(matrice, i, axis=0), j, axis=1)
            det_matrice[i, j] = round(np.linalg.det(minor))

    return det_matrice

def check_sync(arr1, arr2, tol=1e-8):
    '''
    Check if the sort order of arr1 is the reverse of that of arr2,
    allowing for ties.

    Parameters
    ----------
    arr1 : np.array
        1 x n matrix (a row or a col)
    arr2 : np.array
        1 x n matrix (a row or a col)
    tol : float, optional
        random small value to deal with ties (common floating error in numpy). The default is 1e-8.

    Returns
    -------
    bool
        DESCRIPTION.

    '''
    n = len(arr1)
    for i in range(n):
        for j in range(i+1, n):
            if np.abs(arr1[i] - arr1[j]) <= tol:
                continue
            if arr1[i] < arr1[j]:
                if not (arr2[i] > arr2[j] or np.abs(arr2[i]-arr2[j]) <= tol):
                    return False
            elif arr1[i] > arr1[j]:
                if not (arr2[i] < arr2[j] or np.abs(arr2[i]-arr2[j]) <= tol):
                    return False
    return True

def correct_sync(arr1, arr2):
    '''

    Parameters
    ----------
    arr1 : np.array
        1 x n matrix (a row or a col)
    arr2 : np.array
        1 x n matrix (a row or a col)

    Returns
    -------
    corrected_arr : np.array
        reordered version of arr2, so the sort order is reversed compared to arr1

    '''
    idx = np.argsort(arr1, kind='stable')
    arr2_ordered = arr2[idx]
    target_order = np.sort(arr2_ordered)[::-1]
    corrected_arr = np.copy(arr2)
    corrected_arr[idx] = target_order
    return corrected_arr

def correct_matrix_sync_one(result, mat, verbose=True):
    """
    Process the matrices line by line.
    
    It first checks each row (of 'result' and 'mat'). If a row is found out of sync,
    it corrects that row in 'mat', prints a message (e.g. "Corrected row 3"), and returns.
    If all rows are in sync, it then checks each column and does the same.
    
    Parameters
    ----------
    result : np.ndarray, shape (9,9)
        The first matrix (with arbitrary values).
    mat : np.ndarray, shape (9,9)
        The second matrix (values between 1 and 9) to be corrected.
    verbose : bool, optional
        Put it to False if you want a quiet console
    Returns
    -------
    np.ndarray
        The corrected version of mat.
    """
    print_func = partial(print, end='') if verbose else lambda *args, **kwargs: None
    corrected_mat = np.copy(mat)
    # Process row-by-row
    for i in range(result.shape[0]):
        if not check_sync(result[i, :], corrected_mat[i, :]):
            corrected_mat[i, :] = correct_sync(result[i, :], corrected_mat[i, :])
            print_func(f"Corrected row {i}")
            return corrected_mat
    # If rows are all in sync, process column-by-column
    for j in range(result.shape[1]):
        if not check_sync(result[:, j], corrected_mat[:, j]):
            corrected_mat[:, j] = correct_sync(result[:, j], corrected_mat[:, j])
            print_func(f"Corrected column {j}")
            return corrected_mat
    print_func("No corrections needed; all rows and columns are in sync.")
    return corrected_mat


def search_new_mat(mat, randomness=True, max_attempts=10000,verbose=True):
    """
    Given a matrix 'mat', search for a new matrix (by swapping two entries)
    that has a lower determinant than the current one.
    
    Parameters
    ----------
    mat : np.ndarray
        The matrix to be modified.
    randomness : bool, optional
        If True, random swaps will be attempted until a candidate with a lower
        determinant is found. If False, an ordered search is performed.
    verbose : bool, optional
        Put it to False if you want a quiet console
    
    Returns
    -------
    new_mat : np.ndarray
        The modified matrix with a lower determinant.
    """
    print_func = partial(print, end='') if verbose else lambda *args, **kwargs: None
    current_det = np.linalg.det(mat)
    print_func("Starting swap search process since no correction was needed.")
    
    if randomness:
        found_swap = False
        attempt = 0
        
        while not found_swap and attempt < max_attempts:
            attempt += 1
            # Randomly select two distinct indices in the flattened matrix.
            idx_flat = np.random.choice(mat.size, 2, replace=False)
            candidate = np.copy(mat)
            idx1 = np.unravel_index(idx_flat[0], mat.shape)
            idx2 = np.unravel_index(idx_flat[1], mat.shape)
            if mat[idx1] == mat[idx2]:
                print_func(f"Skipping swap between {idx1} and {idx2} as both entries are equal ({mat[idx1]}).")
                continue
            # Swap the two entries.
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
            candidate_det = np.linalg.det(candidate)
            print_func(f"Attempted swap between {idx1} and {idx2}: old det = {current_det:.4f}, candidate det = {candidate_det:.4f}")
            if candidate_det <= current_det:
                print_func("Swap accepted.")
                return candidate
            else:
                print_func("Swap rejected; trying again.")
        print_func("Max attempts reached. No valid swap found.")
    else:
        # Ordered search: iterate over all unique pairs.
        for flat_i in range(mat.size):
            for flat_j in range(flat_i+1, mat.size):
                candidate = np.copy(mat)
                idx1 = np.unravel_index(flat_i, mat.shape)
                idx2 = np.unravel_index(flat_j, mat.shape)
                if mat[idx1] == mat[idx2]:
                    continue
                
                candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
                candidate_det = np.linalg.det(candidate)
                print_func(f"Checking swap between {idx1} and {idx2}: old det = {current_det:.4f}, candidate det = {candidate_det:.4f}")
                if candidate_det < current_det:
                    print_func("Swap accepted.")
                    return candidate
        print_func("No swap found that lowers the determinant; returning original matrix.")
    return mat

def correct_and_recompute(result, mat, randomness=True, verbose=True):
    """
    Corrects the first unsynced row (or, if all rows are synced, the first unsynced column) in mat
    (based on result) and then recomputes result and the determinant of mat.
    
    The recomputation steps are:
      1. Compute: result = cofactor_determinants_numpy(mat)
      2. Create a mask (9x9) as given and multiply the masked entries of result by -1.
      3. Compute the determinant of mat using np.linalg.det.
      
    If no row/column correction occurs (i.e. mat and new_mat are identical), then the function
    calls 'search_new_mat' to find a swap that lowers the determinant.
    
    Parameters
    ----------
    result : np.ndarray, shape (9,9)
        The first matrix (used as a reference).
    mat : np.ndarray, shape (9,9)
        The second matrix to be corrected.
    randomness : bool, optional
        Passed to 'search_new_mat' to determine the swap search mode.
    verbose : bool, optional
        Put it to False if you want a quiet console
    
    Returns
    -------
    new_result : np.ndarray, shape (9,9)
        The recomputed result matrix.
    new_mat : np.ndarray, shape (9,9)
        The (possibly) corrected version of mat.
    det : float
        The determinant of new_mat.
    """
    print_func = partial(print, end='') if verbose else lambda *args, **kwargs: None
    # First, try to correct one row or column.
    new_mat = correct_matrix_sync_one(result, mat, verbose=verbose)
    
    # If no correction was applied, use search_new_mat to swap entries until a lower determinant is found.
    if np.array_equal(new_mat, mat):
        new_mat = search_new_mat(mat, randomness=randomness, verbose=verbose)
    
    # Recompute result.
    new_result = cofactor_determinants_numpy(new_mat)
    
    # Define the mask as provided (9x9 boolean array)
    mask = np.array([0,1,0,1,0,1,0,1,0,
                     1,0,1,0,1,0,1,0,1,
                     0,1,0,1,0,1,0,1,0,
                     1,0,1,0,1,0,1,0,1,
                     0,1,0,1,0,1,0,1,0,
                     1,0,1,0,1,0,1,0,1,
                     0,1,0,1,0,1,0,1,0,
                     1,0,1,0,1,0,1,0,1,
                     0,1,0,1,0,1,0,1,0], dtype=bool).reshape(9,9)
    
    new_result[mask] *= -1
    
    det = np.linalg.det(new_mat)
    print_func("Recomputed determinant:", det)
    
    return new_result, new_mat, det

def generate_start_matrice(lst):
    '''
    generate a random starting matrix from a "square" list of number

    Parameters
    ----------
    lst : list of int

    Returns
    -------
    mat : np.array
        the square matrix out of the list
    result : np.array
        the matrix of the cofactor
    det : float (technically int)
        mat's determinant
    det_last : int
        always 0, used for the loop, technically could be placed elsewhere in the loop

    '''
    random.shuffle(lst)
    mat=np.array(lst).reshape(9,9)
    reshuffle_number=0
    while np.linalg.det(mat)==0:
        reshuffle_number+=1
        print("matrix's determinant is 0, reschuffling number to get a new initial matrix"
              f" reshuffle count={reshuffle_number}")
        random.shuffle(lst)
        mat=np.array(lst).reshape(9,9)
        if np.linalg.det(mat)>0:
            print("swapping two row to change the sign of the matrix's determinant")
            mat[[0,1]] = mat[[1,0]]
            
    result = cofactor_determinants_numpy(mat)
    result[mask] *= -1
    det_last = 0
    det = np.linalg.det(mat)
    return mat,result,det,det_last


if __name__ == "__main__":
    #will generate a matrix with det>900 000 000 about 92% of the time
    det_threshold=900000000
    
    mask = np.array([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
                     0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
                     0],dtype=bool).reshape(9,9)
    lst=generate_list_for_matrice(9)
    mat,result,det,det_last=generate_start_matrice(lst)
    
    retry_number=0
    while True:
        while det_last != det :
            det_last = det
            result, mat, det = correct_and_recompute(result, mat, randomness=False, verbose=False)
        
        if det>=-det_threshold:
            retry_number+=1
            print("failed to find a suitable matrix, reshuffling the original matrix and retrying."
                  f" number of retry: {retry_number}")
            
            mat,result,det,det_last=generate_start_matrice(lst)
        else:
            break

    mat[[0,1]] = mat[[1,0]] 
    print(mat)
    print(f"determinant of mat = {round(np.linalg.det(mat)):,.0f}")


        
            
            
            
            
            