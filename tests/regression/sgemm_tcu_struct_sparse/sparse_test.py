import numpy as np

def prune_2_4_blockwise_with_mask(matrix):
    """
    Perform 2:4 structured sparsity pruning on each row of the input matrix.
    For each consecutive block of 4 elements, keep the two largest (by absolute value)
    and zero out the rest.
    Returns:
        pruned: np.ndarray of same shape, with smaller elements zeroed out
        mask: np.ndarray of bools, True where elements were kept
    """
    pruned = matrix.copy()
    mask = np.zeros_like(matrix, dtype=bool)
    rows, cols = matrix.shape

    for i in range(rows):
        for j in range(0, cols, 4):
            block = pruned[i, j:j+4]
            # Skip blocks that have fewer than 4 elements (at row end)
            if block.shape[0] < 4:
                continue

            abs_vals = np.abs(block)
            sorted_idx = np.argsort(abs_vals)
            top2_idx = sorted_idx[-2:] # Indices of the two largest absolute values

            block_mask = np.zeros_like(block, dtype=bool)
            block_mask[top2_idx] = True

            #apply mask: zero out the smaller two elements in the block
            pruned[i, j:j+4] = block * block_mask
            mask[i, j:j+4] = block_mask

    return pruned, mask

if __name__ == "__main__":
    np.random.seed(42)
    matrix = np.random.randn(8, 8)

    pruned_matrix, mask_matrix = prune_2_4_blockwise_with_mask(matrix)

    print("Original matrix:\n", matrix)
    print("\nPruned matrix (2:4 structured sparse):\n", pruned_matrix)
    print("\nMask matrix (True=kept, False=pruned):\n", mask_matrix)