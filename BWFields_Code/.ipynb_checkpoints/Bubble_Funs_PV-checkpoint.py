import numpy as np
from pvextractor import Path as PathPV
from pvextractor import extract_pv_slice
from scipy.optimize import curve_fit
import statsmodels.api as sm
import copy


def Cal_PV_Path(line_coords, bubble_item, data_wcs_item, width=6):
    """
    Build a PV-extractor Path from a polyline and extract a PV slice from a 3D cube.

    Parameters
    ----------
    line_coords : (N, 2) array-like
        Polyline coordinates along which the PV slice is extracted.
        NOTE: pvextractor expects coordinates in (x, y) order in the image plane.
    bubble_item : 3D ndarray
        Data cube (typically [v, y, x] or similar) from which to extract the PV slice.
    data_wcs_item : astropy.wcs.WCS
        WCS object associated with bubble_item (used by pvextractor to build axes).
    width : int
        Slice width (in pixels) perpendicular to the path.

    Returns
    -------
    pv_path : pvextractor.Path
        Path object used for extraction.
    bubble_item_pv : 2D ndarray
        Extracted PV data array of shape (Nv, Npos).
    """
    pv_path_coords = line_coords
    pv_path_coords = list(map(tuple, pv_path_coords))  # pvextractor expects tuples
    pv_path = PathPV(pv_path_coords, width=width)

    pv_slice = extract_pv_slice(bubble_item, pv_path, wcs=data_wcs_item)
    bubble_item_pv = pv_slice.data
    return pv_path, bubble_item_pv


def Cal_Bubble_Center_On_PVPath(bubble_item_pv, pv_path, bubble_center, pix_scale_arcmin):
    """
    Project a bubble center onto the PV path and build a position axis centered on that projection.

    Parameters
    ----------
    bubble_item_pv : 2D ndarray (Nv, Npos)
        PV data (velocity x position).
    pv_path : pvextractor.Path
        The path used for PV extraction.
    bubble_center : array-like
        Bubble center in *cube pixel coordinates*. In this codebase it is often [v, b, l],
        so the projection uses [l, b] = [bubble_center[2], bubble_center[1]].
    pix_scale_arcmin : float
        Pixel scale for the spatial plane (arcmin per pixel).

    Returns
    -------
    bubble_position_arcmin : float
        Projected bubble center position along the PV path, in arcmin from the path start.
    position_centered_arcmin : (Npos,) ndarray
        Position axis in arcmin, centered so that the bubble center is at 0.
    """
    # Extract path endpoints in image plane (x, y)
    xy_coords = pv_path.get_xy()
    path_start = np.array(xy_coords[0])
    path_end = np.array(xy_coords[-1])

    # Bubble position in the same 2D plane as the PV path
    bubble_pos = np.array([bubble_center[2], bubble_center[1]])

    # Compute unit vector along the path (global image frame)
    path_vector = path_end - path_start
    path_length = np.linalg.norm(path_vector)
    path_unit_vector = path_vector / path_length

    # Project bubble position onto the path
    bubble_to_start = bubble_pos - path_start
    projection_length = np.dot(bubble_to_start, path_unit_vector)

    # Build position axis in arcmin
    n_position = bubble_item_pv.shape[1]
    path_length_arcmin = path_length * pix_scale_arcmin
    position_original = np.linspace(0, path_length_arcmin, n_position)

    bubble_position_arcmin = projection_length * pix_scale_arcmin
    position_centered_arcmin = position_original - bubble_position_arcmin

    return bubble_position_arcmin, position_centered_arcmin


def Extract_Velocity_Profile_From_PV(pv_data, position_axis, velocity_axis,
    systemic_velocity=None, delta_v=0.0,clip_negative=True, min_weight=0):
    """
    Compute intensity-weighted velocity moments from a PV diagram, and optionally
    separate redshifted/blueshifted components relative to a systemic velocity.

    Parameters
    ----------
    pv_data : (Nv, Npos) array
        PV diagram. First dimension is velocity channels, second is position bins.
    position_axis : (Npos,) array
        Spatial positions (arcmin or pixels). (Not used in calculations here, but kept for consistency.)
    velocity_axis : (Nv,) array
        Velocity channels (km/s).
    systemic_velocity : float, optional
        If provided, compute separate red/blue weighted means:
          - red:  v > v_sys + delta_v
          - blue: v < v_sys - delta_v
    delta_v : float
        Exclusion zone around v_sys to avoid ambiguous near-systemic channels.
    clip_negative : bool
        If True, negative intensities are set to 0 before computing moments.
    min_weight : float
        Positions with total intensity <= min_weight are considered invalid.

    Returns
    -------
    weighted_velocity : (Npos,) array
        Intensity-weighted mean velocity mu(x).
    velocity_dispersion : (Npos,) array
        Intensity-weighted velocity dispersion sigma(x).
    intensity_weights : (Npos,) array
        Total integrated intensity S(x) = sum_v T(v, x).
    red_component : dict
        {'mean': (Npos,), 'weights': (Npos,)} for v > v_sys + delta_v (zeros where empty).
    blue_component : dict
        {'mean': (Npos,), 'weights': (Npos,)} for v < v_sys - delta_v (zeros where empty).
    """
    pv = np.asarray(pv_data, dtype=float)
    vel = np.asarray(velocity_axis, dtype=float)

    if pv.ndim != 2:
        raise ValueError("pv_data must be 2D: (Nv, Npos)")
    Nv, Npos = pv.shape
    if vel.shape[0] != Nv:
        raise ValueError("velocity_axis length must match pv_data.shape[0]")

    # Replace NaNs with zeros; optionally clip negatives
    pv = np.nan_to_num(pv, nan=0.0)
    if clip_negative:
        pv[pv < 0] = 0.0

    # Total intensity per position
    S = np.nansum(pv, axis=0)  # (Npos,)
    valid = S > min_weight

    # First moment: mu = sum(T*v) / sum(T)
    V = np.einsum('vi,v->i', pv, vel, optimize=True)  # (Npos,)
    mu = np.zeros(Npos)
    mu[valid] = V[valid] / S[valid]

    # Second moment: E[v^2] then var = E[v^2] - mu^2
    V2 = np.einsum('vi,v->i', pv, vel**2, optimize=True)
    m2 = np.zeros(Npos)
    m2[valid] = V2[valid] / S[valid]
    var = np.clip(m2 - mu**2, a_min=0.0, a_max=None)
    sigma = np.sqrt(var)

    # Allocate arrays for red/blue components (defaults)
    red_mean = np.full(Npos, np.nan)
    blue_mean = np.full(Npos, np.nan)
    red_w = np.zeros(Npos)
    blue_w = np.zeros(Npos)

    # Optional: split into redshifted and blueshifted sides
    if systemic_velocity is not None:
        m_red_v = vel > (systemic_velocity + delta_v)
        m_blue_v = vel < (systemic_velocity - delta_v)

        S_red = np.nansum(pv[m_red_v, :], axis=0)
        S_blue = np.nansum(pv[m_blue_v, :], axis=0)

        if np.any(m_red_v):
            V_red = np.einsum('vi,v->i', pv[m_red_v, :], vel[m_red_v], optimize=True)
            mask = S_red > min_weight
            red_mean[mask] = V_red[mask] / S_red[mask]
            red_w[mask] = S_red[mask]

        if np.any(m_blue_v):
            V_blue = np.einsum('vi,v->i', pv[m_blue_v, :], vel[m_blue_v], optimize=True)
            mask = S_blue > min_weight
            blue_mean[mask] = V_blue[mask] / S_blue[mask]
            blue_w[mask] = S_blue[mask]

    # Replace NaNs with 0 for convenience downstream (you can keep NaNs if preferred)
    red_mean = np.nan_to_num(red_mean)
    blue_mean = np.nan_to_num(blue_mean)

    red_component = {'mean': red_mean, 'weights': red_w}
    blue_component = {'mean': blue_mean, 'weights': blue_w}

    return mu, sigma, S, red_component, blue_component


def Cal_Mean_Profile_PV(position_axis_record, weighted_velocity_record,
                        systemic_v=None, axis_coords=None, mean_line_num=6):
    """
    Compute a binned mean PV velocity profile from multiple cuts/lines.

    The function stacks all (position, velocity) samples from multiple PV cuts, bins
    them along the position axis, and computes a mean velocity per bin.

    Parameters
    ----------
    position_axis_record : list of 1D arrays
        Each element is a position axis for one PV cut (arcmin or pixels).
    weighted_velocity_record : list of 1D arrays
        Each element is the corresponding weighted velocity profile for one PV cut.
    systemic_v : float or None
        If None, systemic_v is estimated as the mean profile value near position 0.
    axis_coords : 1D array or None
        If provided, use these bin centers. If None, build integer bins from min..max.
    mean_line_num : int
        Minimum number of non-zero samples required in a bin to compute a mean;
        otherwise the bin is set to 0.

    Returns
    -------
    mean_profile : 1D ndarray
        Mean velocity profile in the bins.
    std_arr : 1D ndarray
        Standard deviation per bin.
    axis_coords : 1D ndarray
        Bin centers used.
    systemic_v : float
        Systemic velocity used/estimated.
    """
    mean_arr, std_arr = [], []
    xall_peak = np.array([])
    yall_peak = np.array([])

    # Concatenate samples from all cuts
    for i in range(len(position_axis_record)):
        xall_peak = np.concatenate([xall_peak, position_axis_record[i]])
        yall_peak = np.concatenate([yall_peak, weighted_velocity_record[i]])

    # Keep only non-zero velocities (zeros are treated as "no data")
    xall_peak_eff = xall_peak[np.where(yall_peak != 0)]
    yall_peak_eff = yall_peak[np.where(yall_peak != 0)]

    # Define bins / axis coords if not provided
    if axis_coords is None:
        min_range = np.int32(np.min(xall_peak_eff))
        max_range = np.int32(np.max(xall_peak_eff))
        bins = np.linspace(min_range, max_range, max_range - min_range + 1)
        axis_coords = bins[:-1]

    bins = np.r_[axis_coords, axis_coords[-1] + 1]

    # Compute mean/std per bin
    for axis_coords_i in axis_coords:
        # Select samples within this bin (bin width ~ diff(bins)[0])
        yall_bin_i = yall_peak[
            ((xall_peak >= (axis_coords_i - .5 * np.diff(bins)[0])) &
             (xall_peak <  (axis_coords_i + .5 * np.diff(bins)[0])))
        ]
        yall_bin_i = np.nan_to_num(yall_bin_i)

        coords_bin_i = np.where(yall_bin_i != 0)[0]
        if len(coords_bin_i) > mean_line_num:
            mean_arr.append(np.mean(yall_bin_i[coords_bin_i]))
            std_arr.append(np.std(yall_bin_i))
        else:
            mean_arr.append(0)
            std_arr.append(0)

    # Estimate systemic velocity from the bin closest to 0
    if systemic_v is None:
        center_idx = np.argmin(np.abs(axis_coords))
        systemic_v = mean_arr[center_idx]

    mean_profile = np.nan_to_num(mean_arr, 0)
    std_arr = np.nan_to_num(std_arr, 0)

    return mean_profile, std_arr, axis_coords, systemic_v


def Analyze_Expansion_Signature(position_axis, weighted_velocity, velocity_dispersion,
                               valid_mask, systemic_v=None):
    """
    Analyze expansion-like signatures in a velocity profile across the bubble.

    This function compares mean velocities on the left (pos <= 0) and right (pos >= 0)
    relative to a systemic velocity, producing simple expansion metrics and a heuristic
    classification.

    Parameters
    ----------
    position_axis : 1D array
        Positions along the cut (arcmin/pixels), typically centered on the bubble.
    weighted_velocity : 1D array
        Weighted mean velocity mu(x) at each position.
    velocity_dispersion : 1D array
        Velocity dispersion sigma(x) at each position.
    valid_mask : 1D bool array
        Mask selecting positions considered "inside" the bubble radius (or otherwise valid).
    systemic_v : float or None
        Systemic velocity. If None, it is estimated near position 0.

    Returns
    -------
    results : dict
        Contains expansion metrics, asymmetry, a classification label, and turbulence stats.
    """
    # Keep only valid points
    pos_valid = position_axis[valid_mask]
    vel_valid = weighted_velocity[valid_mask]
    disp_valid = velocity_dispersion[valid_mask]

    # If not given, estimate systemic velocity from the point closest to center
    if systemic_v is None:
        center_idx = np.argmin(np.abs(pos_valid))
        systemic_v = vel_valid[center_idx]

    results = {'systemic_v': systemic_v}

    # Not enough data -> return zeros
    if len(pos_valid) < 3:
        results.update({
            'expansion_left': 0, 'expansion_right': 0,
            'expansion_left_max': 0, 'expansion_right_max': 0,
            'mean_expansion': 0, 'asymmetry': 0,
            'classification': 0, 'confidence': 0
        })
        return results

    # Split into left/right/center subsets
    left_mask = pos_valid <= 0
    right_mask = pos_valid >= 0
    center_mask = np.abs(pos_valid) < 1.0

    if np.any(left_mask) and np.any(right_mask):
        # Left side expansion (relative to systemic)
        vel_left = vel_valid[left_mask]
        vel_left_mean = np.mean(vel_left)
        expansion_left = vel_left_mean - systemic_v
        expansion_left_deltas = vel_left - systemic_v
        expansion_left_max = expansion_left_deltas[np.argsort(np.abs(expansion_left_deltas))[-1]]

        # Right side expansion (relative to systemic)
        vel_right = vel_valid[right_mask]
        vel_right_mean = np.mean(vel_right)
        expansion_right = vel_right_mean - systemic_v
        expansion_right_deltas = vel_right - systemic_v
        expansion_right_max = expansion_right_deltas[np.argsort(np.abs(expansion_right_deltas))[-1]]

        # Center region mean (optional diagnostic)
        vel_center = vel_valid[center_mask]
        vel_center_mean = np.mean(vel_center) if len(vel_center) > 0 else systemic_v

        # Heuristic classification thresholds
        if expansion_left > 1.0 and expansion_right > 1.0:
            expansion_classification = "Strong Bilateral Expansion"
            confidence = "High"
        elif expansion_left > 0.5 or expansion_right > 0.5:
            expansion_classification = "Moderate Expansion"
            confidence = "Medium"
        else:
            expansion_classification = "Weak/Complex Motion"
            confidence = "Low"

        # Mean expansion and asymmetry metric
        mean_expansion_velocity = (expansion_left + expansion_right) / 2
        asymmetry_factor = abs(expansion_left - expansion_right) / (expansion_left + expansion_right + 1e-6)

        results.update({
            'expansion_left': expansion_left,
            'expansion_right': expansion_right,
            'expansion_left_max': expansion_left_max,
            'expansion_right_max': expansion_right_max,
            'mean_expansion': mean_expansion_velocity,
            'asymmetry': asymmetry_factor,
            'classification': expansion_classification,
            'confidence': confidence
        })
    else:
        results.update({
            'expansion_left': 0, 'expansion_right': 0,
            'expansion_left_max': 0, 'expansion_right_max': 0,
            'mean_expansion': 0, 'asymmetry': 0,
            'classification': 0, 'confidence': 0
        })

    # Turbulence diagnostics from velocity dispersion
    mean_dispersion = np.mean(disp_valid)
    max_dispersion = np.max(disp_valid)

    if mean_dispersion > 2.0:
        turbulence_impact = "High turbulence may affect expansion dynamics"
    elif mean_dispersion > 1.0:
        turbulence_impact = "Moderate turbulence influence"
    else:
        turbulence_impact = "Low turbulence impact"

    results.update({
        'mean_dispersion': mean_dispersion,
        'max_dispersion': max_dispersion,
        'turbulence_impact': turbulence_impact
    })

    return results


def Fit_Expansion_Models(position_axis, weighted_velocity, valid_mask):
    """
    Fit simple expansion models to a velocity profile.

    Models
    ------
    1) Linear:     v = v_sys + k * |r|
    2) Power-law:  v = v_sys + k * |r|^alpha

    Parameters
    ----------
    position_axis : 1D array
        Positions along the PV cut.
    weighted_velocity : 1D array
        Weighted mean velocity at each position.
    valid_mask : 1D bool array
        Positions used for fitting.

    Returns
    -------
    models : dict or None
        Dict of fitted model info (params, function, r2, etc.), or None if no model fit succeeded.
    best_model_key : str or None
        Key of the model with the highest R^2, or None.
    """
    pos_valid = position_axis[valid_mask]
    vel_valid = weighted_velocity[valid_mask]

    if len(pos_valid) < 3:
        print("Insufficient data points for PV model fitting")
        return None, None

    # Use the point closest to 0 as systemic velocity anchor
    center_idx = np.argmin(np.abs(pos_valid))
    v_sys = vel_valid[center_idx]

    # Model 1: v = v_sys + k*|r|
    def linear_expansion(r, k):
        return v_sys + k * np.abs(r)

    # Model 2: v = v_sys + k*|r|^alpha
    def power_law_expansion(r, k, alpha):
        return v_sys + k * np.abs(r)**alpha

    models = {}

    # Fit linear model
    try:
        popt1, pcov1 = curve_fit(linear_expansion, pos_valid, vel_valid, p0=[0.5])
        pred1 = linear_expansion(pos_valid, *popt1)
        r2_1 = 1 - np.sum((vel_valid - pred1)**2) / np.sum((vel_valid - np.mean(vel_valid))**2)

        models['linear'] = {
            'params': popt1,
            'function': linear_expansion,
            'name': 'Linear Expansion',
            'formula': f'v = {v_sys:.2f} + {popt1[0]:.3f}|r|',
            'r2': r2_1
        }
    except:
        print("Linear expansion model fitting failed")

    # Fit power-law model
    try:
        popt2, pcov2 = curve_fit(
            power_law_expansion, pos_valid, vel_valid,
            p0=[0.5, 1.0], bounds=([0, 0.1], [5, 3])
        )
        pred2 = power_law_expansion(pos_valid, *popt2)
        r2_2 = 1 - np.sum((vel_valid - pred2)**2) / np.sum((vel_valid - np.mean(vel_valid))**2)

        models['power'] = {
            'params': popt2,
            'function': power_law_expansion,
            'name': 'Power Law Expansion',
            'formula': f'v = {v_sys:.2f} + {popt2[0]:.3f}|r|^{popt2[1]:.2f}',
            'r2': r2_2
        }
    except:
        print("Power law expansion model fitting failed")

    # Pick best model by R^2
    if models:
        best_model_key = max(models.keys(), key=lambda k: models[k]['r2'])
        return models, best_model_key
    else:
        return None, None


def Calculate_Bubble_Age_Energy(analysis_results, position_range_arcmin, distance_kpc=1.0):
    """
    Estimate bubble age and energetic requirements from expansion velocity.

    Assumptions
    ----------
    - Bubble radius is half of the position range.
    - Age ~ R / v_exp (simple kinematic age)
    - Gas density is assumed constant (default 100 cm^-3)
    - Total input energy is kinetic_energy / efficiency (efficiency=0.1)

    Parameters
    ----------
    analysis_results : dict
        Output of Analyze_Expansion_Signature, expected to contain 'mean_expansion'.
    position_range_arcmin : float
        Full diameter range along the position axis, in arcmin.
    distance_kpc : float
        Distance to the object in kpc.

    Returns
    -------
    dict or None
        {'age_years', 'mass_solar', 'kinetic_energy', 'required_power'} or None if no expansion.
    """
    print(f"\n===== Bubble Age and Energy Estimation =====")

    if 'mean_expansion' not in analysis_results:
        print("No expansion detected, cannot estimate age")
        return None

    # Convert radius from arcmin to parsec
    radius_arcmin = position_range_arcmin / 2
    radius_pc = radius_arcmin / 60 * np.pi / 180 * distance_kpc * 1000  # pc

    expansion_velocity = analysis_results['mean_expansion']  # km/s

    # Expansion age estimate (seconds -> years)
    if expansion_velocity > 0:
        age_seconds = radius_pc * 3.086e13 / (expansion_velocity * 1e5)
        age_years = age_seconds / (365.25 * 24 * 3600)
    else:
        age_seconds = 0
        age_years = 0

    # Mass estimate assuming constant density
    assumed_density = 100  # cm^-3
    volume_cm3 = (4/3) * np.pi * (radius_pc * 3.086e18)**3
    mass_g = assumed_density * 1.67e-24 * volume_cm3
    mass_solar = mass_g / 1.989e33

    kinetic_energy = 0
    required_power = 0

    if expansion_velocity > 0:
        kinetic_energy = 0.5 * mass_g * (expansion_velocity * 1e5)**2  # erg
        efficiency = 0.1
        total_energy = kinetic_energy / efficiency
        if age_seconds > 0:
            required_power = total_energy / age_seconds  # erg/s
            print(f"Required average power: {required_power:.1e} erg/s")

    return {
        'age_years': age_years,
        'mass_solar': mass_solar,
        'kinetic_energy': kinetic_energy,
        'required_power': required_power
    }


def Analyze_Mean_PV_Profile(weighted_velocity_mean, position_axis_mean,
                           systemic_v, bubble_outer_radius, pix_scale_arcmin,
                           exp_base='systemic_v'):
    """
    Analyze expansion in a *mean* PV velocity profile and fit expansion models.

    Parameters
    ----------
    weighted_velocity_mean : 1D ndarray
        Mean velocity profile (zeros indicate invalid/no data bins).
    position_axis_mean : 1D ndarray
        Mean position axis (centered around 0).
    systemic_v : float
        Systemic velocity used for expansion analysis (if exp_base='systemic_v').
    bubble_outer_radius : float
        Bubble outer radius in pixels.
    pix_scale_arcmin : float
        Pixel scale (arcmin/pixel).
    exp_base : {'systemic_v','central_v'}
        - 'systemic_v': use provided systemic_v
        - 'central_v' : let Analyze_Expansion_Signature estimate central systemic velocity

    Returns
    -------
    expansion_analysis_mean : dict
        Expansion metrics for the mean profile.
    models_mean : dict or None
        Fitted model dict.
    best_model_key_mean : str or None
        Best model key by R^2.
    """
    # Replace zeros with systemic_v for smoother analysis (keeps shape consistent)
    weighted_velocity_mean_copy = weighted_velocity_mean.copy()
    weighted_velocity_mean_copy[np.where(weighted_velocity_mean_copy == 0)] = systemic_v

    # Valid region: within projected bubble radius and with non-zero mean profile
    valid_mask_mean = (np.abs(position_axis_mean) <= np.around(bubble_outer_radius * pix_scale_arcmin)) & \
                      (weighted_velocity_mean != 0)

    if exp_base == 'systemic_v':
        expansion_analysis_mean = Analyze_Expansion_Signature(
            position_axis_mean, weighted_velocity_mean_copy,
            position_axis_mean, valid_mask_mean, systemic_v
        )
    elif exp_base == 'central_v':
        expansion_analysis_mean = Analyze_Expansion_Signature(
            position_axis_mean, weighted_velocity_mean_copy,
            position_axis_mean, valid_mask_mean, systemic_v=None
        )

    models_mean, best_model_key_mean = Fit_Expansion_Models(
        position_axis_mean, weighted_velocity_mean_copy, valid_mask_mean
    )
    return expansion_analysis_mean, models_mean, best_model_key_mean


def Analyze_Exp_Vs(bubbleObj, exp_base='central_v'):
    """
    Summarize expansion sign consistency across multiple PV cuts.

    This function computes sign statistics (positive/negative/different) for expansion
    velocities measured on left and right sides across PV cuts. It also builds a
    compact label array ('P','N','D') for a selected cut.

    Parameters
    ----------
    bubbleObj : object
        Must contain:
          - exp_vs_central_v or exp_vs_systemic_v : array shape (Ncuts, 2, 2)
            [[left,right],[left_max,right_max]]
          - exp_central_delta_v_arg_max : int index of the representative cut
    exp_base : {'central_v','systemic_v'}
        Choose which expansion reference to use.
    """
    exp_central_delta_v_arg_max = bubbleObj.exp_central_delta_v_arg_max

    if exp_base == 'central_v':
        exp_signs = np.sign(bubbleObj.exp_vs_central_v)
    elif exp_base == 'systemic_v':
        exp_signs = np.sign(bubbleObj.exp_vs_systemic_v)

    bubbleObj.exp_signs = exp_signs

    # Fractions of different / consistent sign pairs across cuts
    bubbleObj.exp_sign_diff_mean_per = np.around(np.where(exp_signs.sum(2)[:, 0] == 0)[0].shape[0] / len(exp_signs.sum(2)), 3)
    bubbleObj.exp_sign_diff_max_per = np.around(np.where(exp_signs.sum(2)[:, 1] == 0)[0].shape[0] / len(exp_signs.sum(2)), 3)

    bubbleObj.exp_sign_positive_mean_per = np.around(np.where(exp_signs.sum(2)[:, 0] == 2)[0].shape[0] / len(exp_signs.sum(2)), 3)
    bubbleObj.exp_sign_positive_max_per = np.around(np.where(exp_signs.sum(2)[:, 1] == 2)[0].shape[0] / len(exp_signs.sum(2)), 3)

    bubbleObj.exp_sign_negative_mean_per = np.around(np.where(exp_signs.sum(2)[:, 0] == -2)[0].shape[0] / len(exp_signs.sum(2)), 3)
    bubbleObj.exp_sign_negative_max_per = np.around(np.where(exp_signs.sum(2)[:, 1] == -2)[0].shape[0] / len(exp_signs.sum(2)), 3)

    # Build P/N/D labels for the chosen cut
    exp_signs_light_black = exp_signs.sum(2)[exp_central_delta_v_arg_max]
    exp_signs_light_black_str = exp_signs_light_black.astype(str)
    exp_signs_light_black_str[np.where(exp_signs_light_black == 2)] = 'P'
    exp_signs_light_black_str[np.where(exp_signs_light_black == -2)] = 'N'
    exp_signs_light_black_str[np.where(exp_signs_light_black == 0)] = 'D'
    bubbleObj.exp_signs_light_black_str = exp_signs_light_black_str


def Cal_Expansion_V_Delta(bubbleObj, expansion_analysis_mean,
                          expansion_analysis_mean_red, expansion_analysis_mean_blue):
    """
    Compute "delta" metrics between max expansion and mean expansion for three profiles:
    total (black), red, and blue.

    Side Effects
    ------------
    Sets:
      - bubbleObj.expansion_v_delta
      - bubbleObj.expansion_v_delta_red
      - bubbleObj.expansion_v_delta_blue
    """
    # Total/black
    expansion_v_mean = (expansion_analysis_mean['expansion_left'] +
                        expansion_analysis_mean['expansion_right']) / 2
    expansion_v_max = expansion_analysis_mean['expansion_left_max']
    bubbleObj.expansion_v_delta = expansion_v_max - expansion_v_mean

    # Red
    expansion_v_mean = (expansion_analysis_mean_red['expansion_left'] +
                        expansion_analysis_mean_red['expansion_right']) / 2
    expansion_v_max = expansion_analysis_mean_red['expansion_left_max']
    bubbleObj.expansion_v_delta_red = expansion_v_max - expansion_v_mean

    # Blue
    expansion_v_mean = (expansion_analysis_mean_blue['expansion_left'] +
                        expansion_analysis_mean_blue['expansion_right']) / 2
    expansion_v_max = expansion_analysis_mean_blue['expansion_left_max']
    bubbleObj.expansion_v_delta_blue = expansion_v_max - expansion_v_mean


def Cal_Radial_Velocity_Profile(bubbleObj, line_coords_index=0, width=2, plot_logic=False):
    """
    Compute a radial velocity profile (and expansion analysis) for a single PV cut.

    Parameters
    ----------
    bubbleObj : object
        Must provide bubble_item, WCS, bubble center, bubble radius, systemic velocity, etc.
    line_coords_index : int
        Index selecting which polyline to use from bubbleObj.dictionary_cuts['lines_coords'].
    width : int
        PV slice width (pixels).
    plot_logic : bool
        Placeholder for plotting logic (not used in this snippet).

    Side Effects
    ------------
    Stores PV path/data, profiles, and analysis results back into bubbleObj.
    """
    dictionary_cuts = bubbleObj.dictionary_cuts
    bubble_item = bubbleObj.bubble_item
    data_wcs_item = bubbleObj.data_wcs_item
    pix_scale_arcmin = bubbleObj.pix_scale_arcmin
    bubble_com_item = bubbleObj.bubble_com_item
    bubble_outer_radius = bubbleObj.bubble_outer_radius
    systemic_v = bubbleObj.systemic_v
    velocity_axis = bubbleObj.velocity_axis

    # Select one cut line and extract PV slice
    line_coords = dictionary_cuts['lines_coords'][line_coords_index]
    pv_path, bubble_item_pv = Cal_PV_Path(line_coords, bubble_item, data_wcs_item, width)

    # Center position axis on bubble center projection
    bubble_position_arcmin, position_axis = Cal_Bubble_Center_On_PVPath(
        bubble_item_pv, pv_path, bubble_com_item, pix_scale_arcmin
    )

    # Extract velocity moments (with red/blue split)
    weighted_velocity, velocity_dispersion, intensity_weights, red_comp, blue_comp = \
        Extract_Velocity_Profile_From_PV(
            bubble_item_pv, position_axis, velocity_axis,
            systemic_velocity=systemic_v, delta_v=0.17
        )

    # Valid region within bubble radius
    valid_mask = np.abs(position_axis) < bubble_outer_radius * pix_scale_arcmin
    used_ids = weighted_velocity != 0

    # Expansion feature analysis + model fitting
    expansion_analysis = Analyze_Expansion_Signature(
        position_axis[used_ids], weighted_velocity[used_ids],
        velocity_dispersion[used_ids], valid_mask[used_ids], systemic_v
    )
    models, best_model_key = Fit_Expansion_Models(
        position_axis[used_ids], weighted_velocity[used_ids], valid_mask[used_ids]
    )

    # Estimate a "central velocity" as the first non-zero near position 0
    for i in range(len(position_axis)):
        central_v = weighted_velocity[np.argsort(np.abs(position_axis))[i]]
        if central_v != 0:
            break
    bubbleObj.central_v = central_v

    # Store results in bubbleObj
    bubbleObj.pv_path = pv_path
    bubbleObj.bubble_item_pv = bubble_item_pv
    bubbleObj.bubble_position_arcmin = bubble_position_arcmin
    bubbleObj.position_axis = position_axis
    bubbleObj.weighted_velocity = weighted_velocity
    bubbleObj.velocity_dispersion = velocity_dispersion
    bubbleObj.wv_red = red_comp['mean']
    bubbleObj.wv_blue = blue_comp['mean']
    bubbleObj.iw_red = red_comp['weights']
    bubbleObj.iw_blue = blue_comp['weights']
    bubbleObj.valid_mask = valid_mask
    bubbleObj.intensity_weights = intensity_weights
    bubbleObj.models = models
    bubbleObj.best_model_key = best_model_key
    bubbleObj.expansion_analysis = expansion_analysis


def Cal_Radial_Velocity_Profile_Mean_I(bubbleObj, mean_line_num=6, width=4):
    """
    Compute velocity profiles for *all* PV cuts, then build a mean profile (black/red/blue),
    run expansion analyses, and compute significance metrics.

    This is the main multi-cut workflow:
      - Loop over all lines
      - Extract PV slice
      - Extract velocity moments (total/red/blue)
      - Analyze expansion vs systemic and vs central
      - Aggregate profiles and compute binned mean profiles
      - Analyze mean profiles and compute derived metrics (delta, signs, significance)

    Parameters
    ----------
    bubbleObj : object
        Must contain the PV cut definitions in bubbleObj.dictionary_cuts['lines_coords']
        and many fields used in sub-functions.
    mean_line_num : int
        Minimum samples per bin when constructing the mean profile.
    width : int
        PV slice width (pixels).

    Side Effects
    ------------
    Writes many attributes into bubbleObj, including mean profiles, model fits,
    expansion sign statistics, sigma_env, and significance results.
    """
    weighted_velocity_record = []
    position_axis_record = []
    intensity_weights_record = []
    velocity_dispersion_record = []
    pv_data_record = []

    dictionary_cuts = bubbleObj.dictionary_cuts
    bubble_item = bubbleObj.bubble_item
    data_wcs_item = bubbleObj.data_wcs_item
    bubble_com_item = bubbleObj.bubble_com_item
    pix_scale_arcmin = bubbleObj.pix_scale_arcmin
    bubble_outer_radius = bubbleObj.bubble_outer_radius
    systemic_v = bubbleObj.systemic_v
    delta_v = bubbleObj.clumpsObj.delta_v
    velocity_axis = bubbleObj.velocity_axis

    bubbleObj.red_mean_record, bubbleObj.blue_mean_record = [], []

    exp_vs_systemic_v = []
    exp_vs_central_v = []
    central_delta_vs = []

    # --- Loop over all PV cuts ---
    for line_coords_index in range(len(dictionary_cuts['lines_coords'])):
        line_coords = dictionary_cuts['lines_coords'][line_coords_index]
        pv_path, bubble_item_pv = Cal_PV_Path(line_coords, bubble_item, data_wcs_item, width)
        pv_data_record.append(bubble_item_pv)

        bubble_position_arcmin, position_axis = Cal_Bubble_Center_On_PVPath(
            bubble_item_pv, pv_path, bubble_com_item, pix_scale_arcmin
        )

        # Extract moments + red/blue split
        weighted_velocity, velocity_dispersion, intensity_weights, red_comp, blue_comp = \
            Extract_Velocity_Profile_From_PV(
                bubble_item_pv, position_axis, velocity_axis,
                systemic_velocity=systemic_v, delta_v=delta_v
            )

        # Expansion vs systemic velocity
        expansion_analysis, models, best_model_key = Analyze_Mean_PV_Profile(
            weighted_velocity, position_axis, systemic_v,
            bubble_outer_radius, pix_scale_arcmin, exp_base='systemic_v'
        )
        exp_left = expansion_analysis['expansion_left']
        exp_right = expansion_analysis['expansion_right']
        exp_left_max = expansion_analysis['expansion_left_max']
        exp_right_max = expansion_analysis['expansion_right_max']
        exp_vs_systemic_v.append([[exp_left, exp_right], [exp_left_max, exp_right_max]])

        # Expansion vs central velocity (auto-estimated)
        expansion_analysis, models, best_model_key = Analyze_Mean_PV_Profile(
            weighted_velocity, position_axis, systemic_v,
            bubble_outer_radius, pix_scale_arcmin, exp_base='central_v'
        )
        exp_left = expansion_analysis['expansion_left']
        exp_right = expansion_analysis['expansion_right']
        exp_left_max = expansion_analysis['expansion_left_max']
        exp_right_max = expansion_analysis['expansion_right_max']
        exp_vs_central_v.append([[exp_left, exp_right], [exp_left_max, exp_right_max]])

        # Central delta-v (closest to x=0 among non-zero samples)
        v_argmin = np.argmin(np.abs(position_axis[weighted_velocity != 0]))
        central_delta_vs.append(weighted_velocity[weighted_velocity != 0][v_argmin] - systemic_v)

        # Record per-cut profiles
        weighted_velocity_record.append(weighted_velocity)
        position_axis_record.append(position_axis)
        intensity_weights_record.append(intensity_weights)
        velocity_dispersion_record.append(velocity_dispersion)
        bubbleObj.red_mean_record.append(red_comp['mean'])
        bubbleObj.blue_mean_record.append(blue_comp['mean'])

    # Choose the cut with the largest |central_delta_v| as a representative
    exp_central_delta_v_arg_max = np.argmax(np.abs(central_delta_vs))

    # --- Build mean profiles (black/red/blue) ---
    weighted_velocity_mean, weighted_velocity_mean_std, position_axis_mean, systemic_v = \
        Cal_Mean_Profile_PV(position_axis_record, weighted_velocity_record,
                           systemic_v, mean_line_num=mean_line_num)

    weighted_velocity_mean_red, weighted_velocity_mean_std_red, position_axis_mean, systemic_v = \
        Cal_Mean_Profile_PV(position_axis_record, bubbleObj.red_mean_record,
                           systemic_v, position_axis_mean, mean_line_num)

    weighted_velocity_mean_blue, weighted_velocity_mean_std_blue, position_axis_mean, systemic_v = \
        Cal_Mean_Profile_PV(position_axis_record, bubbleObj.blue_mean_record,
                           systemic_v, position_axis_mean, mean_line_num)

    weighted_velocity_means = [weighted_velocity_mean,
                              weighted_velocity_mean_red,
                              weighted_velocity_mean_blue]
    weighted_velocity_mean_stds = [weighted_velocity_mean_std,
                                  weighted_velocity_mean_std_red,
                                  weighted_velocity_mean_std_blue]

    # Analyze mean profiles
    expansion_analysis_mean, models_mean, best_model_key_mean = Analyze_Mean_PV_Profile(
        weighted_velocity_means[0], position_axis_mean, systemic_v,
        bubble_outer_radius, pix_scale_arcmin
    )
    expansion_analysis_mean_red, models_mean_red, best_model_key_mean_red = Analyze_Mean_PV_Profile(
        weighted_velocity_means[1], position_axis_mean, systemic_v,
        bubble_outer_radius, pix_scale_arcmin
    )
    expansion_analysis_mean_blue, models_mean_blue, best_model_key_mean_blue = Analyze_Mean_PV_Profile(
        weighted_velocity_means[2], position_axis_mean, systemic_v,
        bubble_outer_radius, pix_scale_arcmin
    )

    # Compute delta metrics between max and mean expansions
    Cal_Expansion_V_Delta(bubbleObj, expansion_analysis_mean,
                          expansion_analysis_mean_red, expansion_analysis_mean_blue)

    # Central velocity from mean profile (closest to x=0 with non-zero)
    for i in range(len(position_axis_mean)):
        central_v_mean = weighted_velocity_means[0][np.argsort(np.abs(position_axis_mean))[i]]
        if central_v_mean != 0:
            break
    bubbleObj.central_v_mean = central_v_mean

    # Store outputs
    bubbleObj.systemic_v = systemic_v
    bubbleObj.exp_vs_systemic_v = np.array(exp_vs_systemic_v)
    bubbleObj.exp_vs_central_v = np.array(exp_vs_central_v)
    bubbleObj.exp_central_delta_v_arg_max = exp_central_delta_v_arg_max

    bubbleObj.weighted_velocity_record = weighted_velocity_record
    bubbleObj.position_axis_record = position_axis_record
    bubbleObj.intensity_weights_record = intensity_weights_record
    bubbleObj.velocity_dispersion_record = velocity_dispersion_record

    bubbleObj.weighted_velocity_means = weighted_velocity_means
    bubbleObj.weighted_velocity_mean_stds = weighted_velocity_mean_stds
    bubbleObj.position_axis_mean = position_axis_mean

    bubbleObj.expansion_analysis_mean = expansion_analysis_mean
    bubbleObj.models_mean = models_mean
    bubbleObj.best_model_key_mean = best_model_key_mean

    bubbleObj.expansion_analysis_mean_red = expansion_analysis_mean_red
    bubbleObj.models_mean_red = models_mean_red
    bubbleObj.best_model_key_mean_red = best_model_key_mean_red

    bubbleObj.expansion_analysis_mean_blue = expansion_analysis_mean_blue
    bubbleObj.models_mean_blue = models_mean_blue
    bubbleObj.best_model_key_mean_blue = best_model_key_mean_blue

    bubbleObj.pv_data_record = pv_data_record

    # Expansion sign summary + environmental sigma estimation + significance scores
    Analyze_Exp_Vs(bubbleObj, exp_base='central_v')
    sigma_env = Estimate_Sigma_Env(bubbleObj, systemic_velocity=bubbleObj.systemic_v)
    exp_results, max_Sexp, max_Sexp_index = Compute_Exp_Significance(bubbleObj, sigma_env=sigma_env)

    bubbleObj.sigma_env = sigma_env
    bubbleObj.exp_results = exp_results
    bubbleObj.max_Sexp = max_Sexp
    bubbleObj.max_Sexp_index = max_Sexp_index


def Cal_Radial_Velocity_Profile_Mean(bubbleObj, pv_type=1, mean_line_num=4, width=4):
    """
    Wrapper for computing mean PV profiles under different PV center definitions.

    pv_type options
    --------------
    1: Use bubble morphology center (default, bubbleObj.bubble_com_item)
    2: Use gas center (bubbleObj.bubble_gas_com_1) as bubble center for PV centering
    None: Try both types and keep the one with "better" separation
          (based on |exp_max - exp_mean| heuristic)

    Side Effects
    ------------
    Updates bubbleObj in-place with chosen results.
    """
    if pv_type == 1:
        Cal_Radial_Velocity_Profile_Mean_I(bubbleObj, mean_line_num=mean_line_num, width=width)
        bubbleObj.pv_type = pv_type

    elif pv_type == 2:
        bubbleObj.bubble_com_item = bubbleObj.bubble_gas_com_1 - bubbleObj.start_coords
        Cal_Radial_Velocity_Profile_Mean_I(bubbleObj, mean_line_num=mean_line_num, width=width)
        bubbleObj.pv_type = pv_type

    elif pv_type is None:
        # Try pv_type=1
        Cal_Radial_Velocity_Profile_Mean_I(bubbleObj, mean_line_num=mean_line_num, width=width)
        bubbleObj.pv_type = 1
        bubbleObj_pv_type_1 = copy.copy((bubbleObj))

        exp_left = bubbleObj_pv_type_1.expansion_analysis_mean['expansion_left']
        exp_right = bubbleObj_pv_type_1.expansion_analysis_mean['expansion_right']
        exp_mean_1 = (exp_left + exp_right) / 2
        exp_left_max = bubbleObj_pv_type_1.expansion_analysis_mean['expansion_left_max']
        exp_right_max = bubbleObj_pv_type_1.expansion_analysis_mean['expansion_right_max']
        exp_max_1 = (exp_left_max + exp_right_max) / 2

        # Try pv_type=2
        bubbleObj.bubble_com_item = bubbleObj.bubble_gas_com_1 - bubbleObj.start_coords
        Cal_Radial_Velocity_Profile_Mean_I(bubbleObj, mean_line_num=mean_line_num, width=width)
        bubbleObj.pv_type = 2
        bubbleObj_pv_type_2 = copy.copy(bubbleObj)

        exp_left = bubbleObj_pv_type_2.expansion_analysis_mean['expansion_left']
        exp_right = bubbleObj_pv_type_2.expansion_analysis_mean['expansion_right']
        exp_mean_2 = (exp_left + exp_right) / 2
        exp_left_max = bubbleObj_pv_type_2.expansion_analysis_mean['expansion_left_max']
        exp_right_max = bubbleObj_pv_type_2.expansion_analysis_mean['expansion_right_max']
        exp_max_2 = (exp_left_max + exp_right_max) / 2

        # Heuristic: choose the result with larger |exp_max - exp_mean|
        if np.abs(exp_max_1 - exp_mean_1) > np.abs(exp_max_2 - exp_mean_2):
            best = bubbleObj_pv_type_1
        else:
            best = bubbleObj_pv_type_2
        bubbleObj.__dict__.update(best.__dict__)


def Estimate_Sigma_Env_From_Single_PV(
    pv_data, position_axis, velocity_axis, R_out,
    systemic_velocity=None, delta_v=0.0,
    clip_negative=True, min_weight=0.0,
    use_outer=True, outer_factor=1.5
):
    """
    Estimate environmental velocity dispersion (sigma_env) from a single PV slice.

    The idea is to use positions outside the bubble (|x| > outer_factor * R_out)
    to characterize background turbulence / dispersion.

    Parameters
    ----------
    pv_data : (Nv, Npos) array
        PV intensity T(v, x).
    position_axis : (Npos,) array
        Spatial axis (arcmin).
    velocity_axis : (Nv,) array
        Velocity channels (km/s).
    R_out : float
        Bubble outer radius in the same units as position_axis (arcmin).
    systemic_velocity, delta_v, clip_negative, min_weight :
        Passed to Extract_Velocity_Profile_From_PV.
    use_outer : bool
        If True, prefer using outside-bubble positions. If too few points, fall back to broader masks.
    outer_factor : float
        Defines "environment" as |x| > outer_factor * R_out (e.g., 1.5).

    Returns
    -------
    sigma_env_slice : float
        Intensity-weighted mean of sigma(x) in the chosen environment region,
        or NaN if no valid points.
    """
    mu, sigma, S, red_comp, blue_comp = Extract_Velocity_Profile_From_PV(
        pv_data, position_axis, velocity_axis,
        systemic_velocity=systemic_velocity,
        delta_v=delta_v, clip_negative=clip_negative, min_weight=min_weight
    )

    sigma = np.asarray(sigma)
    S = np.asarray(S)
    x = np.asarray(position_axis)

    valid = S > min_weight

    if use_outer:
        # Prefer outside-bubble region
        mask_env = valid & (np.abs(x) > outer_factor * R_out)

        # If too few points, relax the mask progressively
        if np.sum(mask_env) < 6:
            mask_env = valid & (np.abs(x) > outer_factor)
            if np.sum(mask_env) < 3:
                mask_env = valid
    else:
        mask_env = valid

    if np.sum(mask_env) == 0:
        return np.nan

    # Intensity-weighted average sigma as sigma_env estimate
    sigma_env_slice = np.average(sigma[mask_env], weights=S[mask_env])
    return sigma_env_slice


def Estimate_Sigma_Env(bubbleObj, systemic_velocity=None, delta_v=0.0,
                       clip_negative=True, min_weight=0.0,
                       use_outer=True, outer_factor=1.5):
    """
    Estimate sigma_env for a bubble by aggregating sigma_env estimates from all PV cuts.

    Requirements on bubbleObj
    -------------------------
    Must provide:
      - bubbleObj.pv_data_record : list of (Nv, Npos) arrays
      - bubbleObj.position_axis_record : list of (Npos,) arrays
      - bubbleObj.velocity_axis : (Nv,) array
      - bubbleObj.bubble_outer_radius : float (pixels)
      - bubbleObj.pix_scale_arcmin : float (arcmin/pixel)

    Returns
    -------
    sigma_env : float
        Median sigma_env across cuts (robust), or NaN if all failed.
    """
    sigma_list = []

    velocity_axis = bubbleObj.velocity_axis
    R_out = bubbleObj.bubble_outer_radius * bubbleObj.pix_scale_arcmin

    position_axis_record = bubbleObj.position_axis_record
    pv_data_record = bubbleObj.pv_data_record

    for pv_data, position_axis in zip(pv_data_record, position_axis_record):
        sigma_env_slice = Estimate_Sigma_Env_From_Single_PV(
            pv_data=pv_data,
            position_axis=position_axis,
            velocity_axis=velocity_axis,
            R_out=R_out,
            systemic_velocity=systemic_velocity,
            delta_v=delta_v,
            clip_negative=clip_negative,
            min_weight=min_weight,
            use_outer=use_outer,
            outer_factor=outer_factor
        )

        if not np.isnan(sigma_env_slice):
            sigma_list.append(sigma_env_slice)

    if len(sigma_list) == 0:
        return np.nan

    sigma_env = np.median(sigma_list)
    return sigma_env


def Compute_Exp_Significance_For_Line(position_axis, velocity_profile, R_out,
                                     sigma_env=None, env_factor=1.5):
    """
    Compute expansion (V-shaped) significance metrics for a single position–velocity cut.

    Three quantities are defined:
        EV   : depth significance (central velocity offset relative to environment)
        Ec   : curvature significance (quadratic curvature inside |x| <= R_out)
        S_exp: combined expansion significance, sqrt(EV^2 + Ec^2)

    Parameters
    ----------
    position_axis : 1D array
        Spatial coordinate along the cut (e.g. arcmin).
    velocity_profile : 1D array
        Corresponding velocity values (e.g. km/s).
    R_out : float
        Outer radius of the bubble (same units as position_axis).
    sigma_env : float or None
        Velocity dispersion of the environment (km/s).
        If None, it is estimated from linear-fit residuals within |x| <= R_out.
    env_factor : float
        Multiplicative factor defining the outer environment region.
        The environment is taken from R_out <= |x| <= env_factor * R_out.

    Returns
    -------
    EV : float
        Depth significance of the V-shape.
    Ec : float
        Curvature significance of the V-shape.
    S_exp : float
        Combined expansion significance.
    """

    # Convert inputs to numpy arrays
    x_all = np.asarray(position_axis, dtype=float)
    v_all = np.asarray(velocity_profile, dtype=float)

    # Remove invalid points (NaN or zero velocity)
    m_valid = ~np.isnan(v_all) & (v_all != 0)
    x_all = x_all[m_valid]
    v_all = v_all[m_valid]

    # Require a minimum number of points
    if len(x_all) < 5:
        return 0,0,0
        # return np.nan, np.nan, np.nan

    # ------------------------------------------------------------------
    # 1. Select inner region |x| <= R_out for curvature fitting
    # ------------------------------------------------------------------
    mask_in = np.abs(x_all) <= R_out
    x = x_all[mask_in]
    v = v_all[mask_in]

    if len(x) < 5:
        return 0,0,0
        # return np.nan, np.nan, np.nan

    # ------------------------------------------------------------------
    # 2. Estimate environmental velocity dispersion if not provided
    #    (use residuals from a linear fit within |x| <= R_out)
    # ------------------------------------------------------------------
    if sigma_env is None:
        X1 = sm.add_constant(x)
        model_lin = sm.OLS(v, X1).fit()
        sigma_env = np.std(v - model_lin.fittedvalues, ddof=1)
        sigma_env = max(sigma_env, 1e-6)  # avoid division by zero

    # ------------------------------------------------------------------
    # 3. Compute EV: central velocity vs. outer environmental velocity
    # ------------------------------------------------------------------
    # Central velocity: point closest to x = 0
    center_idx = np.argmin(np.abs(x))
    Vc = v[center_idx]

    # Define environmental regions: R_out <= |x| <= env_factor * R_out
    R_env_max = env_factor * R_out
    mask_left_env  = (x_all <= -R_out) & (x_all >= -R_env_max)
    mask_right_env = (x_all >=  R_out) & (x_all <=  R_env_max)

    # Mean velocity on each side if available
    if np.any(mask_left_env):
        V_l_env = np.nanmean(v_all[mask_left_env])
    else:
        V_l_env = v[0]

    if np.any(mask_right_env):
        V_r_env = np.nanmean(v_all[mask_right_env])
    else:
        V_r_env = v[-1]

    # Choose the environmental baseline depending on available sides
    if np.isfinite(V_l_env) and np.isfinite(V_r_env):
        V_env = 0.5 * (V_l_env + V_r_env)
    elif np.isfinite(V_l_env):
        V_env = V_l_env
    elif np.isfinite(V_r_env):
        V_env = V_r_env
    else:
        # Fallback: use the inner endpoints
        Vl_inner = v[0]
        Vr_inner = v[-1]
        V_env = 0.5 * (Vl_inner + Vr_inner)

    # Depth significance
    dV = V_env - Vc
    EV = dV / sigma_env

    # ------------------------------------------------------------------
    # 4. Compute Ec: quadratic curvature significance within |x| <= R_out
    # ------------------------------------------------------------------
    # Fit v(x) = a + b*x + c*x^2
    X2 = sm.add_constant(np.column_stack([x, x**2]))
    model_quad = sm.OLS(v, X2).fit()
    a, b, c = model_quad.params

    # Curvature significance scaled by R_out
    Ec = np.abs(c) * R_out**2 / sigma_env

    # ------------------------------------------------------------------
    # 5. Combined expansion significance
    # ------------------------------------------------------------------
    S_exp = np.sqrt(EV**2 + Ec**2)

    return EV, Ec, S_exp


def Compute_Exp_Significance(bubbleObj, sigma_env=None):
    """
    Compute expansion significance for multiple velocity profiles
    stored in a bubble object.

    This function evaluates Compute_Exp_Significance_For_Line for:
        - Three mean velocity profiles
        - One profile corresponding to the maximum central velocity offset

    Parameters
    ----------
    bubbleObj : object
        Bubble object containing:
            - position_axis_mean
            - bubble_outer_radius
            - pix_scale_arcmin
            - weighted_velocity_means
            - position_axis_record
            - weighted_velocity_record
            - exp_central_delta_v_arg_max
    sigma_env : float or None
        Environmental velocity dispersion passed to the lower-level function.

    Returns
    -------
    results : dict
        Dictionary keyed by curve name ("black", "red", "blue", etc.),
        each containing EV, Ec, and S_exp.
    max_Sexp : float
        Maximum expansion significance among the main three curves.
    max_Sexp_index : int
        Index (0, 1, or 2) of the curve with the maximum significance.
    """

    # Mean position axis and outer radius in physical units
    position_axis_mean = bubbleObj.position_axis_mean
    R_out = bubbleObj.bubble_outer_radius * bubbleObj.pix_scale_arcmin

    # Extract the velocity profile with maximum central delta-v
    position_axis_record = bubbleObj.position_axis_record
    weighted_velocity_record = bubbleObj.weighted_velocity_record
    exp_central_delta_v_arg_max = bubbleObj.exp_central_delta_v_arg_max

    position_axis_delta_v_arg_max = position_axis_record[exp_central_delta_v_arg_max]
    weighted_velocity_delta_v_arg_max = weighted_velocity_record[exp_central_delta_v_arg_max]

    results = {}

    # Labels used as keys (historically treated as colors)
    colors = ["black", "red", "blue", "light black"]

    # Assemble axes and velocity arrays to loop over
    position_axiss = [
        position_axis_mean,
        position_axis_mean,
        position_axis_mean,
        position_axis_delta_v_arg_max
    ]

    vel_arrays = [
        bubbleObj.weighted_velocity_means[0],
        bubbleObj.weighted_velocity_means[1],
        bubbleObj.weighted_velocity_means[2],
        weighted_velocity_delta_v_arg_max
    ]

    # Compute expansion significance for each curve
    for name, position_axis, v in zip(colors, position_axiss, vel_arrays):
        EV, Ec, S_exp = Compute_Exp_Significance_For_Line(
            position_axis=position_axis,
            velocity_profile=v,
            R_out=R_out,
            sigma_env=sigma_env
        )
        results[name] = dict(EV=EV, Ec=Ec, S_exp=S_exp)

    # Identify the maximum significance among the three main curves
    colors_main = ["black", "red", "blue"]
    max_Sexp = max([results[c]["S_exp"] for c in colors_main])
    max_Sexp_index = np.argmax([results[c]["S_exp"] for c in colors_main])

    return results, max_Sexp, max_Sexp_index







    


