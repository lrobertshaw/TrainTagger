import numpy as np

def mass_flatten_weights(truth_mass_train):
    """
    Generate sample weights to flatten the mass distribution.
    No classes are involved.
    """

    num_samples = truth_mass_train.shape[0]
    sample_weights = np.ones(num_samples)

    mass_bins = np.linspace(0, 257, 65)

    # Count how many jets fall into each mass bin
    mass_counts, _ = np.histogram(truth_mass_train, bins=mass_bins)

    # Find the maximum bin count
    max_count = np.max(mass_counts)

    # Compute weights for each bin
    weights_per_bin = np.zeros(len(mass_bins) - 1)
    for bin_idx in range(len(mass_bins) - 1):
        if mass_counts[bin_idx] == 0:
            weights_per_bin[bin_idx] = 0.  # no samples in this bin
        else:
            weights_per_bin[bin_idx] = max_count / mass_counts[bin_idx]

    # Assign weight to each sample
    bin_indices = np.digitize(truth_mass_train, mass_bins) - 1
    bin_indices[bin_indices == len(mass_bins) - 1] = len(mass_bins) - 2  # Edge handling

    sample_weights = weights_per_bin[bin_indices]

    sample_weights[3:23] *= 2 # Increase weight for jets in region 8 - 128 GeV

    # Normalize so mean weight = 1
    sample_weights /= np.mean(sample_weights)

    return sample_weights
