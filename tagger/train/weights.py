import numpy as np

def flatten_weights(var, mi, mx, b):
    """
    Generate sample weights to flatten the mass distribution.
    No classes are involved.
    """

    num_samples = var.shape[0]
    sample_weights = np.ones(num_samples)

    bins = np.linspace(mi, mx, b)

    # Count how many jets fall into each mass bin
    counts, _ = np.histogram(var, bins=bins)

    # Find the maximum bin count
    max_count = np.max(counts)

    # Compute weights for each bin
    weights_per_bin = np.zeros(len(bins) - 1)
    for bin_idx in range(len(bins) - 1):
        if counts[bin_idx] == 0:
            weights_per_bin[bin_idx] = 0.  # no samples in this bin
        else:
            weights_per_bin[bin_idx] = max_count / counts[bin_idx]

    # Assign weight to each sample
    bin_indices = np.digitize(var, bins) - 1
    bin_indices[bin_indices == len(bins) - 1] = len(bins) - 2  # Edge handling

    sample_weights = weights_per_bin[bin_indices]

    # Normalize so mean weight = 1
    sample_weights /= np.mean(sample_weights)
    print(f"Mean weight: {np.mean(sample_weights)}")
    print(f"Max weight: {np.max(sample_weights)}")
    print(f"Min weight: {np.min(sample_weights)}")
    print(f"Median weight: {np.median(sample_weights)}")
    print(f"Std weight: {np.std(sample_weights)}")
    print(f"Number of 0s: {np.sum(sample_weights == 0)}")
    print(f"Number of NaNs: {np.sum(np.isnan(sample_weights))}")
    print(f"Number of Infs: {np.sum(np.isinf(sample_weights))}")
    return sample_weights
