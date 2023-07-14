
import numpy as np
import copy

# Use at least float64 for the accumulating functions to avoid precision issue
# see https://github.com/numpy/numpy/issues/9393. The float64 is also retained
# as it is in case the float overflows
def _safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides numpy accumulator functions with a float64 dtype
    when used on a floating point input. This prevents accumulator overflow on
    smaller floating point dtypes.

    Parameters
    ----------
    op : function
        A numpy accumulator function such as np.mean or np.sum.
    x : ndarray
        A numpy array to apply the accumulator function.
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x.
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function.

    Returns
    -------
    result
        The output of the accumulator function passed to this function.
    """
    if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype=np.float64)
    else:
        result = op(x, *args, **kwargs)
    return result


def _incremental_mean_and_var(new_mean, new_variance, new_sample_count, last_mean, last_variance, last_sample_count):
    """Calculate mean update and a Youngs and Cramer variance update.

    last_mean and last_variance are statistics computed at the last step by the
    function. Both must be initialized to 0.0. In case no scaling is required
    last_variance can be None. The mean is always required and returned because
    necessary for the calculation of the variance. last_n_samples_seen is the
    number of samples encountered until now.

    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to use for variance update.

    last_mean : array-like of shape (n_features,)

    last_variance : array-like of shape (n_features,)

    last_sample_count : array-like of shape (n_features,)

    Returns
    -------
    updated_mean : ndarray of shape (n_features,)

    updated_variance : ndarray of shape (n_features,)
        If None, only mean is computed.

    updated_sample_count : ndarray of shape (n_features,)

    Notes
    -----
    NaNs are ignored during the algorithm.

    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247

    Also, see the sparse implementation of this in
    `utils.sparsefuncs.incr_mean_variance_axis` and
    `utils.sparsefuncs_fast.incr_mean_variance_axis0`
    """
    # old = stats until now
    # new = the current increment
    # updated = the aggregated stats
    last_sum = last_mean * last_sample_count
    # new_sum = _safe_accumulator_op(np.nansum, X, axis=0)
    new_sum = new_mean * new_sample_count

    # new_sample_count = np.sum(~np.isnan(X), axis=0)
    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        # new_unnormalized_variance = (
        #    _safe_accumulator_op(np.nanvar, X, axis=0) * new_sample_count)
        new_unnormalized_variance = new_variance * new_sample_count

        last_unnormalized_variance = last_variance * last_sample_count

        with np.errstate(divide='ignore', invalid='ignore'):
            last_over_new_count = last_sample_count / new_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance + new_unnormalized_variance +
                last_over_new_count / updated_sample_count *
                (last_sum / last_over_new_count - new_sum) ** 2)

        zeros = last_sample_count == 0
        updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
        updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count


def _merge_scalers(scaler1, scaler2):
    last_mean = scaler1.mean_
    last_variance = scaler1.var_
    last_sample_count = scaler1.n_samples_seen_

    new_mean = scaler2.mean_
    new_variance = scaler2.var_
    new_sample_count = scaler2.n_samples_seen_

    updated_mean, updated_variance, updated_sample_count = _incremental_mean_and_var(new_mean, new_variance, new_sample_count, last_mean, last_variance, last_sample_count)
    updated_scale = np.sqrt(updated_variance)

    # print("updated_mean=", updated_mean)
    # print("updated_variance=", updated_variance)
    # print("updated_sample_count=", updated_sample_count)
    # print("updated_scale=", updated_scale)

    scaler1.mean_ = updated_mean
    scaler1.var_ = updated_variance
    scaler1.n_samples_seen_ = updated_sample_count
    scaler1.scale_ = updated_scale
    return scaler1


def reduce_scalers(scalers):
    head = scalers.pop(0)
    scaler = copy.deepcopy(head)
    for sc in scalers:
        scaler = _merge_scalers(scaler, sc)

    return scaler







