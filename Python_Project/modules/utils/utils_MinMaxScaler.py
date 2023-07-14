
import numpy as np
import copy

def _handle_zeros_in_scale(scale, copy=True, constant_mask=None):
    """Set scales of near constant features to 1.

    The goal is to avoid division by very small or zero values.

    Near constant features are detected automatically by identifying
    scales close to machine precision unless they are precomputed by
    the caller and passed with the `constant_mask` kwarg.

    Typically for standard scaling, the scales are the standard
    deviation while near constant features are better detected on the
    computed variances which are closer to machine precision by
    construction.
    """
    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    elif isinstance(scale, np.ndarray):
        if constant_mask is None:
            # Detect near constant values to avoid dividing by a very small
            # value that could lead to surprising results and numerical
            # stability issues.
            constant_mask = scale < 10 * np.finfo(scale.dtype).eps

        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[constant_mask] = 1.0
        return scale



def _merge_scalers(scaler1, scaler2):

    last_min = scaler1.min_
    last_data_min = scaler1.data_min_
    last_data_max = scaler1.data_max_
    last_data_range = scaler1.data_range_
    last_sample_count = scaler1.n_samples_seen_
    last_scale = scaler1.scale_
    last_feature_range = scaler1.feature_range

    new_min = scaler2.min_
    new_data_min = scaler2.data_min_
    new_data_max = scaler2.data_max_
    new_data_range = scaler2.data_range_
    new_sample_count = scaler2.n_samples_seen_
    new_scale = scaler2.scale_

    # Calculate new scaler attributes
    ######################################
    updated_data_min = np.minimum(last_data_min , new_data_min)
    updated_data_max = np.maximum(last_data_max , new_data_max)
    updated_sample_count = last_sample_count + new_sample_count
    updated_data_range = updated_data_max - updated_data_min
    updated_scale = (last_feature_range[1] - last_feature_range[0]) / _handle_zeros_in_scale(updated_data_range, copy=True)
    updated_min = last_feature_range[0] - updated_data_min * updated_scale

    #######################################

    # print("updated_mean=", updated_mean)
    # print("updated_variance=", updated_variance)
    # print("updated_sample_count=", updated_sample_count)
    # print("updated_scale=", updated_scale)

    # Update new scaler attributes
    scaler1.min_ = updated_min
    scaler1.data_min_ = updated_data_min
    scaler1.data_max_ = updated_data_max
    scaler1.data_range_ = updated_data_range
    scaler1.n_samples_seen_ = updated_sample_count
    scaler1.scale_ = updated_scale

    return scaler1


def reduce_scalers(scalers):
    head = scalers.pop(0)
    scaler = copy.deepcopy(head)
    for sc in scalers:
        scaler = _merge_scalers(scaler, sc)

    return scaler







