import time
import numpy as np
import copy
from skimage import filters, measure, morphology
from scipy import optimize, linalg
import scipy.ndimage as ndimage
from scipy.interpolate import splprep, splev, RegularGridInterpolator
import networkx as nx
from collections import defaultdict
from pvextractor import Path, extract_pv_slice
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def Profile_Builder(image, mask, point, derivative, shift=True, fold=False):
    """
    Build a 1D intensity profile perpendicular to the spine at a given point.

    This function is adapted from RadFil. It computes the perpendicular cut line
    through a given spine point and samples the image intensity along that cut.

    Parameters
    ----------
    image : ndarray
        2D image containing intensity.
    mask : ndarray
        Binary mask of the region.
    point : array-like
        (x, y) coordinates of the spine point.
    derivative : array-like
        Tangent direction vector (dx, dy) of the spine at 'point'.
    shift : bool
        If True, center the distance axis at the local peak along the cut
        (instead of centering at the input spine point).
    fold : bool
        If True, fold distances to positive side (useful for symmetric averaging).

    Returns
    -------
    final_dist : ndarray
        Signed distance array along the cut (centered at peak or spine point).
    image_line_T : ndarray
        Intensity values sampled along the cut; values outside mask are set to 0.
    mask_line : ndarray
        Boolean mask along the sampled cut indicating points inside filament mask.
    line_coords : ndarray
        (x, y) coordinates of sampling points along the cut (pixel coordinate system).
    peak : ndarray
        Integer (x, y) coordinate of the detected peak inside mask.
    (start, end) : tuple
        Two endpoints of the cut segment within the mask.
    """
    # Fill holes in the mask to ensure continuous regions
    mask = ndimage.binary_fill_holes(mask)

    # Read the target point on the spine
    x0, y0 = point

    # Create pixel-edge grids (cell boundaries) used to compute line-grid intersections
    shapex, shapey = image.shape[1], image.shape[0]
    edgex, edgey = np.arange(.5, shapex - .5, 1.), np.arange(.5, shapey - .5, 1.)

    # Handle degenerate cases where derivative is axis-aligned
    if (derivative[0] == 0) or (derivative[1] == 0):
        # If both are zero, tangent direction is undefined
        if (derivative[0] == 0) and (derivative[1] == 0):
            raise ValueError("Both components of the derivative are zero; unable to derive a tangent.")
        # Vertical tangent -> perpendicular is horizontal; intersection with vertical edges is trivial
        elif (derivative[0] == 0):
            y_edgex = []
            edgex = []
            x_edgey = np.ones(len(edgey)) * x0
        # Horizontal tangent -> perpendicular is vertical
        elif (derivative[1] == 0):
            y_edgex = np.ones(len(edgex)) * y0
            x_edgey = []
            edgey = []

    else:
        # Compute slope of the perpendicular line.
        # If tangent direction is (dx,dy), a perpendicular direction is (-dy,dx).
        # Using slope form: slope_perp = -1 / (dy/dx) = -dx/dy (equivalent).
        slope = -1. / (derivative[1] / derivative[0])

        # Intersections of the perpendicular line with vertical grid lines x = edgex
        y_edgex = slope * (edgex - x0) + y0

        # Intersections with horizontal grid lines y = edgey
        x_edgey = (edgey - y0) / slope + x0

        # Mask out intersection points falling outside the image boundaries
        pts_maskx = ((np.round(x_edgey) >= 0.) & (np.round(x_edgey) < shapex))
        pts_masky = ((np.round(y_edgex) >= 0.) & (np.round(y_edgex) < shapey))

        edgex, edgey = edgex[pts_masky], edgey[pts_maskx]
        y_edgex, x_edgey = y_edgex[pts_masky], x_edgey[pts_maskx]

    # Combine all intersection points and sort them.
    # Sorting is needed to reconstruct ordered segments inside each pixel.
    stack = sorted(list(set(zip(np.concatenate([edgex, x_edgey]),
                                np.concatenate([y_edgex, edgey])))))

    # Midpoints between consecutive intersections => sampling points inside each pixel segment
    coords_total = stack[:-1] + .5 * np.diff(stack, axis=0)

    # Setup bilinear interpolation over pixel centers for more accurate sampling
    xgrid = np.arange(0.5, image.shape[1] + 0.5, 1.0)
    ygrid = np.arange(0.5, image.shape[0] + 0.5, 1.0)
    interpolator = RegularGridInterpolator((xgrid, ygrid), image.T, bounds_error=False, fill_value=None)

    # Sample intensity along the cut line
    image_line = interpolator(coords_total)

    # Sample mask membership along the cut line using nearest pixel indexing
    mask_line = mask[np.round(coords_total[:, 1]).astype(int), np.round(coords_total[:, 0]).astype(int)]

    # Identify the sampling index that corresponds to the original spine point (nearest pixel)
    mask_p0 = (np.round(coords_total[:, 0]).astype(int) == int(round(x0))) & \
              (np.round(coords_total[:, 1]).astype(int) == int(round(y0)))

    # Extract profile values within mask to locate the peak robustly
    if derivative[1] < 0.:
        image_line0 = image_line[mask_line][::-1]
    else:
        image_line0 = image_line[mask_line]

    # Build a “masked intensity line”: outside-mask values are set to 0
    image_line_T = np.zeros_like(image_line)
    image_line_T[mask_line] = image_line[mask_line]

    # Ensure consistent ordering depending on derivative direction
    if derivative[1] < 0.:
        coords_total = coords_total[::-1]
        mask_line = mask_line[::-1]
        mask_p0 = mask_p0[::-1]
        image_line_T = image_line_T[::-1]

    # Candidate coordinates inside mask (used for endpoints/peak detection)
    peak_finder = coords_total[mask_line]
    if len(peak_finder) != 0:

        # Endpoints of the cut segment that lies inside the mask
        start, end = peak_finder[0], peak_finder[-1]
    
        # Peak position: choose the coordinate corresponding to maximum intensity inside mask
        xpeak, ypeak = peak_finder[image_line0 == np.nanmax(image_line0)][0]
        peak = np.around([xpeak, ypeak]).astype(int)
    
        # Boolean mask of the sampling coordinate that corresponds to the peak position
        mask_peak = (np.round(coords_total[:, 0]).astype(int) == int(round(xpeak))) & \
                    (np.round(coords_total[:, 1]).astype(int) == int(round(ypeak)))
    
        # Build distance axis:
        # - if shift=True, center distances at peak
        # - else, center distances at original spine point
        if shift:
            final_dist = np.hypot(coords_total[:, 0] - xpeak, coords_total[:, 1] - ypeak)
            pos0 = np.where(mask_peak)[0][0]
            # Make one side negative so profile has signed distances
            final_dist[:pos0] = -final_dist[:pos0]
        else:
            final_dist = np.hypot(coords_total[:, 0] - x0, coords_total[:, 1] - y0)
            pos0 = np.where(mask_p0)[0][0]
            final_dist[:pos0] = -final_dist[:pos0]
    
        # Optionally fold distances for symmetric averaging
        if fold:
            final_dist = abs(final_dist)
    else:
        final_dist = None
        peak = None
        start, end = None, None
    return final_dist, image_line_T, mask_line, coords_total, peak, (start, end)


def Get_Sub_Mask(point, regions_data, related_ids_T, connected_ids_dict, clump_coords_dict, start_coords):
    """
    Extract a sub-mask containing the clumps connected to a given point.

    The function:
    1) Finds which labeled clump the input point belongs to.
    2) Collects all clumps connected to that clump.
    3) Keeps only those connected clumps that are part of the filament (related_ids_T).
    4) Builds a binary mask for these clumps.

    Parameters
    ----------
    point : array-like
        (x, y) coordinates to locate the clump label.
    regions_data : ndarray
        2D or 3D array of region labels (1-based IDs assumed).
    related_ids_T : list
        Clump IDs that are considered part of the filament.
    connected_ids_dict : dict
        Mapping clump_id -> list of connected clump IDs.
    clump_coords_dict : dict
        Mapping clump_id -> array of pixel coordinates in that clump.
    start_coords : ndarray
        Origin offset for local coordinate conversion.

    Returns
    -------
    fil_mask_sub_profiles : ndarray
        2D binary mask of selected connected clumps.
    """
    # 2D case: regions_data[y, x] gives label
    if len(regions_data.shape) == 2:
        # Convert point to nearest integer pixel and convert label to 0-based ID
        clump_id = regions_data[np.int64(np.around(point))[1], np.int64(np.around(point))[0]] - 1

        # Include itself + its connected neighbors
        connected_ids_i = [clump_id] + connected_ids_dict[clump_id]

        # Build an empty mask and fill selected clumps
        fil_mask_sub_profiles = np.zeros_like(regions_data)

        for connected_id in connected_ids_i:
            if connected_id in related_ids_T:
                coords = clump_coords_dict[connected_id]
                # Convert global coords to local coords by subtracting start_coords
                fil_mask_sub_profiles[(coords[:, 0] - start_coords[0],
                                       coords[:, 1] - start_coords[1])] = 1

    # 3D case: multiple labels along LOS can exist; we project to 2D mask
    elif len(regions_data.shape) == 3:
        clump_ids = regions_data[:, np.int64(np.around(point))[1], np.int64(np.around(point))[0]] - 1
        clump_ids = list(set(clump_ids))

        # Remove invalid background (-1)
        if -1 in clump_ids:
            clump_ids.remove(-1)

        # Gather all connected clumps from all IDs found
        connected_ids_i = []
        for clump_id in clump_ids:
            connected_ids_i += [clump_id] + connected_ids_dict[clump_id]

        # Projected 2D mask (sum over axis 0)
        fil_mask_sub_profiles = np.zeros_like(regions_data.sum(0))

        for connected_id in connected_ids_i:
            if connected_id in related_ids_T:
                coords = clump_coords_dict[connected_id]
                # coords expected as (z, y, x); project onto (y, x)
                fil_mask_sub_profiles[(coords[:, 1] - start_coords[1],
                                       coords[:, 2] - start_coords[2])] = 1

    return fil_mask_sub_profiles


def Cal_Dictionary_Cuts(regions_data, related_ids_T, connected_ids_dict, clump_coords_dict,
                        points, points_b, fprime, image, mask, dictionary_cuts,
                        start_coords=None, CalSub=False, shift=True):
    """
    Compute perpendicular profiles for a list of spine points and store them into dictionary_cuts.

    Parameters
    ----------
    points : ndarray
        Spine points where profiles will be measured.
    points_b : ndarray
        Auxiliary points (often used as secondary endpoints / ellipse sample points).
    fprime : ndarray
        Derivative (tangent) vectors at each point.
    CalSub : bool
        If True, compute a sub-mask for each point based on connected clumps.
    shift : bool
        Whether to center the profile distance axis at the peak intensity.

    Returns
    -------
    dictionary_cuts : dict
        Updated dictionary containing distances, profiles, mask lines, etc.
    """
    # Only meaningful if we have enough sample points
    if len(points) > 3:
        points_updated = points.copy().tolist()
        fprime_updated = fprime.copy().tolist()
        dictionary_cuts['empty_logic'] = False
        
        for point_id in range(len(points)):
            image_shape = image.shape

            # Check boundary condition (avoid out-of-bounds indexing)
            if np.round(points[point_id][0]) > 0 and np.round(points[point_id][1]) > 0 \
                    and np.round(points[point_id][0]) < image_shape[1] - 1 and \
                    np.round(points[point_id][1]) < image_shape[0] - 1:

                # Optionally compute local sub-mask (connected clumps) for this point
                if CalSub:
                    mask = Get_Sub_Mask(points[point_id], regions_data, related_ids_T, connected_ids_dict,
                                        clump_coords_dict, start_coords)

                # Compute perpendicular intensity profile at this spine point
                profile = Profile_Builder(image, mask, points[point_id], fprime[point_id],
                                          shift=shift, fold=False)

                if profile[0] is not None:
                    # Store results
                    dictionary_cuts['distance'].append(profile[0])
                    dictionary_cuts['profile'].append(profile[1])
                    dictionary_cuts['mask_lines'].append(profile[2])
                    dictionary_cuts['lines_coords'].append(profile[3])
                    dictionary_cuts['plot_peaks'].append(profile[4])
                    dictionary_cuts['plot_cuts'].append(profile[5])
                else:
                    # Drop points that are empty
                    points_updated.remove(points[point_id].tolist())
                    fprime_updated.remove(fprime[point_id].tolist())
                    dictionary_cuts['empty_logic'] = True

            else:
                # Drop points that are too close to boundary
                points_updated.remove(points[point_id].tolist())
                fprime_updated.remove(fprime[point_id].tolist())

        # Save filtered spine points and derivatives
        dictionary_cuts['points'].append(np.array(points_updated))
        dictionary_cuts['fprime'].append(np.array(fprime_updated))
        dictionary_cuts['points_b'] = [points_b]

    return dictionary_cuts


def Cal_Derivative(ellipse_x0, ellipse_y0, ellipse_coords_sample, start_coords):
    """
    Compute derivatives (tangent/perpendicular vectors) for ellipse sample points.

    The function builds a set of identical 'point_a' (center) and varies 'point_b'
    (sample points), then computes a perpendicular direction at each pair.

    Returns
    -------
    points : ndarray
        Repeated center point array.
    fprime : ndarray
        Perpendicular direction vectors for each sample.
    points_b : ndarray
        Sample points (relative coordinates).
    """
    points_b = []
    fprime = []

    for i in range(len(ellipse_coords_sample)):
        # Convert global center to local coords (swap x/y order as needed)
        point_a = [ellipse_y0, ellipse_x0] - start_coords[1:][::-1]
        # Convert each ellipse sample point to local coords
        point_b = ellipse_coords_sample[i][::-1] - start_coords[1:][::-1]

        # Vector from center to sample point
        der = point_b - point_a

        # Perpendicular vector (rotate by 90 degrees)
        derivative = [-der[1], der[0]]

        points_b.append(point_b)
        fprime.append(derivative)

    points = [point_a] * len(fprime)
    return np.array(points), np.array(fprime), np.array(points_b)


def Cal_Mean_Profile(filamentObj, mean_line_num=0, mean_line_range=[-100, 100], ExtendRange=0):
    """
    Compute the mean cross-sectional profile of a filament from stored cuts.

    Workflow:
    1) Collect all distances and intensity samples from filamentObj.dictionary_cuts.
    2) Build symmetric bins across the maximum extent.
    3) For each bin, compute mean intensity (only if enough valid samples).
    4) Store mean profile and left/right components for symmetry analysis.

    Parameters
    ----------
    filamentObj : object
        Must provide pix_scale_arcmin and dictionary_cuts with 'distance' and 'profile'.
    mean_line_num : int
        Minimum number of non-zero points required per bin to compute mean.
    mean_line_range : list
        Range (in arcmin before scaling) to keep for the mean profile.
    ExtendRange : int
        Additional pixels to extend beyond the max sampled distance.

    Returns
    -------
    None
        Results saved into filamentObj fields.
    """
    # Convert mean_line_range from arcmin to pixel units (or internal coordinate units)
    mean_line_range = np.array(mean_line_range) / filamentObj.pix_scale_arcmin

    # Work on a copy so we don't modify original dictionary
    dictionary_cuts = copy.deepcopy(filamentObj.dictionary_cuts)

    mean_arr = []
    std_arr = []
    xall_peak, yall_peak = np.array([]), np.array([])

    # Concatenate all cut distances and profiles
    for i in range(0, len(dictionary_cuts['distance'])):
        xall_peak = np.concatenate([xall_peak, dictionary_cuts['distance'][i]])
        yall_peak = np.concatenate([yall_peak, dictionary_cuts['profile'][i]])

    # Keep only non-zero samples (outside-mask values are 0)
    xall_peak_eff = xall_peak[np.where(yall_peak != 0)]
    yall_peak_eff = yall_peak[np.where(yall_peak != 0)]

    # Determine the max half-range needed
    max_range = np.int32(
        np.max([np.abs(np.min(xall_peak_eff) - ExtendRange),
                np.int32(np.max(xall_peak_eff)) + ExtendRange])
    )

    # Create 1-pixel bins centered around integer distances
    bins = np.linspace(-max_range, max_range, 2 * max_range + 1)
    axis_coords = bins[:-1]

    # Compute per-bin mean/std
    for axis_coords_i in axis_coords:
        # Select samples falling into current bin
        yall_bin_i = yall_peak[((xall_peak >= (axis_coords_i - .5 * np.diff(bins)[0])) &
                                (xall_peak < (axis_coords_i + .5 * np.diff(bins)[0])))]
        coords_bin_i = np.where(yall_bin_i != 0)[0]

        # Only compute mean if enough points and within requested range
        if len(coords_bin_i) > mean_line_num and axis_coords_i > mean_line_range[0] and axis_coords_i < mean_line_range[1]:
            mean_arr.append(np.mean(yall_bin_i[coords_bin_i]))
            std_arr.append(np.std(yall_bin_i))
        else:
            mean_arr.append(0)
            std_arr.append(0)

    mean_profile = np.nan_to_num(mean_arr, 0)
    std_arr = np.nan_to_num(std_arr, 0)

    # Save results back to filament object
    filamentObj.xall_peak = xall_peak
    filamentObj.yall_peak = yall_peak
    filamentObj.xall_peak_eff = xall_peak_eff
    filamentObj.yall_peak_eff = yall_peak_eff
    filamentObj.max_range = max_range
    filamentObj.axis_coords = axis_coords
    filamentObj.mean_profile = mean_profile
    filamentObj.std_arr = std_arr

    # Split mean profile into left/right halves for symmetry checks
    mean_profile_left = mean_profile[1:max_range + 1]
    mean_profile_left_r = mean_profile_left[::-1]
    mean_profile_right = mean_profile[max_range:]
    mean_profile_right_r = mean_profile_right[::-1]

    axis_coords_left = np.linspace(-max_range + 1, 0, max_range)
    axis_coords_right = np.linspace(0, max_range - 1, max_range)

    filamentObj.mean_profile_left = mean_profile_left
    filamentObj.mean_profile_left_r = mean_profile_left_r
    filamentObj.mean_profile_right = mean_profile_right
    filamentObj.mean_profile_right_r = mean_profile_right_r
    filamentObj.axis_coords_left = axis_coords_left
    filamentObj.axis_coords_right = axis_coords_right


def Double_Gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, baseline):
    """Double-Gaussian model: two Gaussians + constant baseline."""
    gauss1 = A1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    gauss2 = A2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    return gauss1 + gauss2 + baseline


def Estimate_Initial_Params(x, y):
    """
    Estimate initial parameters for a constrained double-Gaussian fit.

    Strategy:
    - Estimate baseline from both ends of the profile.
    - Detect peaks using scipy.signal.find_peaks on baseline-subtracted data.
    - Force one peak on left (x<-1) and one on right (x>1) if possible.
    - Provide fallback choices if peak detection fails.

    Returns
    -------
    initial_params : list
        [A1, mu1, sigma1, A2, mu2, sigma2, baseline]
    peak1_idx, peak2_idx : int
        Indices of the chosen left/right peaks in x,y arrays.
    left_peaks, right_peaks : ndarray
        Candidate peak indices on each side.
    """
    baseline = np.mean([np.mean(y[:5]), np.mean(y[-5:])])
    y_corrected = y - baseline

    # Detect peaks above 50% of maximum corrected signal; distance enforces separation
    peaks, properties = find_peaks(y_corrected, height=np.max(y_corrected) * 0.5, distance=6)

    # Split peaks by side
    left_peaks = peaks[x[peaks] < -1]
    right_peaks = peaks[x[peaks] > 1]

    # Choose a left peak (prefer the one closest to center on the left)
    if len(left_peaks) > 0:
        peak1_idx = left_peaks[-1]
    else:
        # Fallback: maximum in x<0 region
        left_mask = x < 0
        peak1_idx = np.where(left_mask)[0][np.argmax(y_corrected[left_mask])] if np.any(left_mask) else 0

    # Choose a right peak (prefer the one closest to center on the right)
    if len(right_peaks) > 0:
        peak2_idx = right_peaks[0]
    else:
        # Fallback: maximum in x>0 region
        right_mask = x > 0
        peak2_idx = np.where(right_mask)[0][np.argmax(y_corrected[right_mask])] if np.any(right_mask) else len(x) - 1

    # Initialize amplitudes/centers
    A1 = y_corrected[peak1_idx]
    mu1 = x[peak1_idx]
    A2 = y_corrected[peak2_idx]
    mu2 = x[peak2_idx]

    # Initial sigma: quarter of peak separation (heuristic)
    sigma = abs(mu2 - mu1) / 4

    return [A1, mu1, sigma, A2, mu2, sigma, baseline], peak1_idx, peak2_idx, left_peaks, right_peaks


def Find_Valley_Between_Peaks(x, y, peak1_idx, peak2_idx):
    """
    Find the minimum (valley) intensity between two peak indices.

    Returns
    -------
    valley_intensity : float
        Minimum y value between two peaks.
    valley_idx : int
        Index location of that valley.
    """
    # Ensure peak1 is on the left of peak2
    if x[peak1_idx] > x[peak2_idx]:
        peak1_idx, peak2_idx = peak2_idx, peak1_idx

    valley_start = min(peak1_idx, peak2_idx)
    valley_end = max(peak1_idx, peak2_idx)

    if valley_end > valley_start:
        valley_region = y[valley_start:valley_end]
        valley_idx = valley_start + np.argmin(valley_region)
        valley_intensity = y[valley_idx]
    else:
        # If peaks overlap/too close, use heuristic fallback
        valley_intensity = min(y[peak1_idx], y[peak2_idx]) * 0.8
        valley_idx = np.int32(np.around((peak1_idx + peak1_idx)) / 2)

    return valley_intensity, valley_idx


def Fit_Double_Gaussian(bubbleObj, ref_ellipse='cavity',
                        left_constraint_width=2.0, right_constraint_width=2.0, bounds_per=3,
                        local_fit=False, intensity_threshold=None):
    """
    Fit a constrained double-Gaussian to ``bubbleObj.mean_profile``.

    Parameters
    ----------
    bubbleObj : object
        Object containing the profile to fit. It must provide:
        - ``axis_coords``  : x coordinates of the profile
        - ``mean_profile`` : y values of the profile
        If ``ref_ellipse='skeleton'``, it must also contain
        ``bubbleObj.fit_results['params']`` from a previous fit.

    ref_ellipse : {'cavity', 'skeleton'}, optional
        Controls how the fit is initialized.
        - 'cavity'   : detect peaks directly from the current profile
        - 'skeleton' : use parameters from a previous fit as prior and
          restrict the new fit around them

    left_constraint_width : float, optional
        Allowed offset of the left Gaussian center (mu1) from the detected
        left peak position.

    right_constraint_width : float, optional
        Allowed offset of the right Gaussian center (mu2) from the detected
        right peak position.

    bounds_per : float, optional
        Only used when ``ref_ellipse='skeleton'``. The new parameter bounds are
        set to ``initial_params ± initial_params / bounds_per``.
        Larger values give tighter bounds.

    local_fit : bool, optional
        If True, fit only the main high-intensity region near the two peaks
        instead of the full profile.

    intensity_threshold : float or None, optional
        Threshold used when ``local_fit=True``.
        - If None, it is estimated from the valley between the two main peaks.
        - If given, it is used directly to define the fitting region.

    Returns
    -------
    results : dict
        Dictionary containing fit parameters, uncertainties, fit quality,
        symmetry score, and the fitted profile. The same result is also saved
        to ``bubbleObj.fit_results``.
    """
    x_full = np.array(bubbleObj.axis_coords)
    y_full = np.array(bubbleObj.mean_profile)

    # Remove invalid samples before peak detection and fitting
    valid_mask = ~(np.isnan(x_full) | np.isnan(y_full))
    x_full, y_full = x_full[valid_mask], y_full[valid_mask]

    if len(x_full) < 7:
        raise ValueError("Insufficient data points for double Gaussian fit")

    # Detect the two main peaks and build initial parameters from the full profile
    initial_params, peak1_idx, peak2_idx, left_peaks, right_peaks = \
        Estimate_Initial_Params(x_full, y_full)

    mu1_detected = x_full[peak1_idx]
    mu2_detected = x_full[peak2_idx]

    # Optional local fit: keep only the main high-intensity region around the peaks
    if local_fit:
        if intensity_threshold is None:
            # Estimate threshold from the valley between the two main peaks
            valley_intensity, valley_idx = Find_Valley_Between_Peaks(
                x_full, y_full, peak1_idx, peak2_idx
            )
            intensity_threshold = valley_intensity * 0.9
        else:
            valley_idx = int(len(y_full))

        threshold_mask = y_full >= intensity_threshold
        threshold_label = measure.label(threshold_mask, connectivity=1)

        # Keep the connected component containing the main peak pair
        if threshold_mask[valley_idx]:
            threshold_mask[np.where(threshold_label != threshold_label[valley_idx])] = False

        # If one side has multiple peaks, trim away remote components
        if len(left_peaks) > 1:
            valley_intensity, valley_idx = Find_Valley_Between_Peaks(
                x_full, y_full, left_peaks[-1], left_peaks[-2]
            )
            threshold_mask[:valley_idx] = False
        if len(right_peaks) > 1:
            valley_intensity, valley_idx = Find_Valley_Between_Peaks(
                x_full, y_full, right_peaks[0], right_peaks[1]
            )
            threshold_mask[valley_idx + 1:] = False

        x = x_full[threshold_mask]
        y = y_full[threshold_mask]

        # Fall back to the full profile if too few points remain
        if len(x) < 7:
            print(f"Warning: Only {len(x)} points above threshold. Using full dataset.")
            x, y = x_full, y_full
            local_fit = False
    else:
        x, y = x_full, y_full
        intensity_threshold = None

    # Rebuild initial parameters if fitting only a local subset
    if local_fit and len(x) != len(x_full):
        left_peaks = np.where((x < 0) & (y >= intensity_threshold))[0]
        right_peaks = np.where((x > 0) & (y >= intensity_threshold))[0]

        peak1_idx_local = left_peaks[np.argmax(y[left_peaks])] if len(left_peaks) > 0 else 0
        peak2_idx_local = right_peaks[np.argmax(y[right_peaks])] if len(right_peaks) > 0 else len(x) - 1

        baseline = intensity_threshold * 0.9
        A1 = y[peak1_idx_local] - baseline
        mu1 = x[peak1_idx_local]
        A2 = y[peak2_idx_local] - baseline
        mu2 = x[peak2_idx_local]
        sigma = abs(mu2 - mu1) / 4
        initial_params = [A1, mu1, sigma, A2, mu2, sigma, baseline]

    # Parameter bounds for [A1, mu1, sigma1, A2, mu2, sigma2, baseline]
    lower_bounds = [
        0,
        max(mu1_detected - left_constraint_width, np.min(x)),
        0.1,
        0,
        max(mu2_detected - right_constraint_width, 0.1),
        0.1,
        0 if not local_fit else intensity_threshold * 0.5
    ]

    upper_bounds = [
        np.inf,
        min(mu1_detected + left_constraint_width, -0.1),
        abs(mu2_detected - mu1_detected),
        np.inf,
        min(mu2_detected + right_constraint_width, np.max(x)),
        abs(mu2_detected - mu1_detected),
        np.max(y)
    ]
    
    # Skeleton mode: use a previous fit as prior and shrink the search range
    if ref_ellipse == 'skeleton':
        if bubbleObj.fit_results['success']:
            initial_params = np.array([
                bubbleObj.fit_results['params'][k]
                for k in ['A1', 'mu1', 'sigma1', 'A2', 'mu2', 'sigma2', 'baseline']
            ])
            lower_bounds = initial_params - initial_params / bounds_per
            upper_bounds = initial_params + initial_params / bounds_per
    
            # Keep mu1 on the left side after updating bounds
            mu1 = lower_bounds[1]
            lower_bounds[1] = upper_bounds[1]
            upper_bounds[1] = mu1

    try:
        # Constrained nonlinear fit
        popt, pcov = curve_fit(
            Double_Gaussian, x, y,
            p0=initial_params,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000
        )

        # Standard fit-quality metrics
        y_pred = Double_Gaussian(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        param_errors = np.sqrt(np.diag(pcov))

        # Symmetry metrics for peak height, position, and width
        symmetry_index_peak_value = np.around(
            1 - np.abs((np.abs(popt[0]) - np.abs(popt[3]))) / (np.abs(popt[0]) + np.abs(popt[3])),3)
        symmetry_index_peak_position = np.around(
            1 - np.abs((np.abs(popt[1]) - np.abs(popt[4]))) / (np.abs(popt[1]) + np.abs(popt[4])),3)
        symmetry_index_fwhm = np.around(
            1 - np.abs((np.abs(popt[2]) - np.abs(popt[5]))) / (np.abs(popt[2]) + np.abs(popt[5])),3)
        symmetry_score = np.around(
            np.mean([symmetry_index_peak_value, symmetry_index_peak_position, symmetry_index_fwhm]),3)

        results = {
            'success': True,
            'params': dict(zip(['A1', 'mu1', 'sigma1', 'A2', 'mu2', 'sigma2', 'baseline'], popt)),
            'errors': dict(zip(['A1_err', 'mu1_err', 'sigma1_err', 'A2_err', 'mu2_err', 'sigma2_err', 'baseline_err'], param_errors)),
            'r_squared': r_squared,
            'rmse': np.sqrt(ss_res / len(y)),
            'detected_peaks': {'left_peak': mu1_detected, 'right_peak': mu2_detected},
            'constraints': {'left_width': left_constraint_width, 'right_width': right_constraint_width},
            'local_fit': local_fit,
            'intensity_threshold': intensity_threshold,
            'x': x,
            'y_original': y,
            'y_fitted': y_pred,
            'x_full': x_full,
            'y_full': y_full,
            'symmetry_score': symmetry_score
        }

        bubbleObj.fit_results = results
        return results

    except Exception as e:
        print(f"Fitting failed: {str(e)}")
        bubbleObj.fit_results = {
            'success': False,
            'local_fit': False,
            'error': str(e),
            'intensity_threshold': intensity_threshold,
            'x': x,
            'y_original': y,
            'x_full': x_full,
            'y_full': y_full,
            'symmetry_score': 0
        }
        return {'success': False, 'error': str(e)}
        

def Print_Fit_Summary(results):
    """
    Print a human-readable summary of the double-Gaussian fit results.

    Includes:
    - Local fit info (how many points used, threshold)
    - Detected peak positions and constraint widths
    - Best-fit parameters with uncertainties
    - Fit quality metrics (R^2, RMSE)
    - Constraint validation (whether centers stayed within allowed ranges)
    """
    if not results['success']:
        print(f"Fitting failed: {results['error']}")
        return

    p = results['params']
    e = results['errors']
    local_fit = results.get('local_fit', False)
    intensity_threshold = results.get('intensity_threshold', None)

    print("=" * 60)
    print("LOCAL CONSTRAINED DOUBLE GAUSSIAN FIT SUMMARY" if local_fit else "CONSTRAINED DOUBLE GAUSSIAN FIT SUMMARY")
    print("=" * 60)

    if local_fit:
        print(f"\nLocal Fit Information:")
        if 'x_full' in results:
            print(f"  Total data points:     {len(results['x_full'])}")
            print(f"  Fitted data points:    {len(results['x'])}")
            print(f"  Data usage:            {len(results['x'])/len(results['x_full'])*100:.1f}%")
        if intensity_threshold is not None:
            print(f"  Intensity threshold:   {intensity_threshold:.3f}")
            print(f"  Threshold method:      {'Auto (valley between peaks)' if intensity_threshold else 'Manual'}")

    if 'detected_peaks' in results and 'constraints' in results:
        detected = results['detected_peaks']
        constraints = results['constraints']
        print(f"\nDetected Peak Positions:")
        print(f"  Left peak detected at:  {detected['left_peak']:.3f} arcmin")
        print(f"  Right peak detected at: {detected['right_peak']:.3f} arcmin")
        print(f"  Left constraint width:  ±{constraints['left_width']:.1f} arcmin")
        print(f"  Right constraint width: ±{constraints['right_width']:.1f} arcmin")

    print(f"\nFitted Peak 1 (Left):")
    print(f"  Amplitude:  {p['A1']:.3f} ± {e['A1_err']:.3f}")
    print(f"  Center:     {p['mu1']:.3f} ± {e['mu1_err']:.3f} arcmin")
    print(f"  Width:      {p['sigma1']:.3f} ± {e['sigma1_err']:.3f} arcmin")
    print(f"  FWHM:       {2.355 * p['sigma1']:.3f} arcmin")

    print(f"\nFitted Peak 2 (Right):")
    print(f"  Amplitude:  {p['A2']:.3f} ± {e['A2_err']:.3f}")
    print(f"  Center:     {p['mu2']:.3f} ± {e['mu2_err']:.3f} arcmin")
    print(f"  Width:      {p['sigma2']:.3f} ± {e['sigma2_err']:.3f} arcmin")
    print(f"  FWHM:       {2.355 * p['sigma2']:.3f} arcmin")

    print(f"\nBaseline:     {p['baseline']:.3f} ± {e['baseline_err']:.3f}")
    print(f"R²:           {results['r_squared']:.6f}")
    print(f"RMSE:         {results['rmse']:.3f}")
    print(f"Peak separation: {abs(p['mu2'] - p['mu1']):.3f} arcmin")
    print(f"Intensity ratio (A1/A2): {p['A1']/p['A2']:.3f}")

    if 'detected_peaks' in results and 'constraints' in results:
        detected = results['detected_peaks']
        constraints = results['constraints']

        left_deviation = abs(p['mu1'] - detected['left_peak'])
        right_deviation = abs(p['mu2'] - detected['right_peak'])

        print(f"\nConstraint Validation:")
        print(f"  Left peak deviation:  {left_deviation:.3f} arcmin (max: {constraints['left_width']:.1f})")
        print(f"  Right peak deviation: {right_deviation:.3f} arcmin (max: {constraints['right_width']:.1f})")
        print(f"  Left peak position:   {p['mu1']:.3f} arcmin (< 0: {'✓' if p['mu1'] < 0 else '✗'})")
        print(f"  Right peak position:  {p['mu2']:.3f} arcmin (> 0: {'✓' if p['mu2'] > 0 else '✗'})")

        constraint_violated = (left_deviation > constraints['left_width'] or right_deviation > constraints['right_width'])
        position_violated = (p['mu1'] >= 0 or p['mu2'] <= 0)

        if constraint_violated or position_violated:
            print("  ⚠️  WARNING: Constraint violations detected!")
        else:
            print("  ✓ All constraints satisfied")

    print(f"\nFitting Quality Assessment:")
    if results['r_squared'] > 0.95:
        print("  ✓ Excellent fit (R² > 0.95)")
    elif results['r_squared'] > 0.90:
        print("  ✓ Good fit (R² > 0.90)")
    elif results['r_squared'] > 0.80:
        print("  ⚠️ Fair fit (R² > 0.80)")
    else:
        print("  ⚠️ Poor fit (R² < 0.80) - consider adjusting parameters")

    if local_fit and intensity_threshold is not None:
        print("  Local fitting helped focus on peak regions")


def Cal_Bubble_RFWHM(bubbleObj, ref_ellipse='cavity', SymScore=0.5, thickness_min=4, distance_pc_item=None):
    """
    Estimate bubble radius and shell thickness from the double-Gaussian fit
    to the mean radial profile.

    The function first tries to use the fitted double-Gaussian parameters
    (when the fit is successful and sufficiently symmetric). If the fit is
    unreliable or produces unphysical values, it falls back to a geometry-based
    estimate derived from the fitted ellipse.

    Definitions
    ----------
    radius
        Mean of the absolute positions of the two Gaussian centers:
        (|mu1| + |mu2|) / 2

    thickness
        Mean of the FWHM values of the two Gaussian components:
        (FWHM1 + FWHM2) / 2, where FWHM = 2.355 * sigma

    outer radius
        Approximate outer boundary of the shell:
        radius + thickness / 2

    Parameters
    ----------
    bubbleObj : object
        Object containing at least:
        - pix_scale_arcmin : angular size per pixel
        - fit_results : output of the double-Gaussian fit
        - ellipse_infor : ellipse parameters, where indices [3] and [4]
          correspond to the semi-major and semi-minor axes
    SymScore : float, optional
        Minimum symmetry score required for accepting the fitted profile-based
        radius/thickness estimate.
    thickness_min : float, optional
        Lower bound imposed on the shell thickness (and also on the fallback
        radius scale) to avoid unrealistically small bubbles.
    distance_pc_item : float or None, optional
        Source distance in parsec. If provided, store it in
        `bubbleObj.distance_pc_item` and convert angular measurements into
        physical units.

    Returns
    -------
    bubble_params : dict
        Dictionary containing angular radius/thickness estimates and, if a
        distance is available, the corresponding physical scales in parsec.

        `bubble_params_type` indicates which branch was used:
        - 1 : accepted fit-based estimate
        - 2 : fit existed but failed sanity checks, so fallback was used
        - 3 : fit unsuccessful or too asymmetric, so fallback was used
    """
    pix_scale_arcmin = bubbleObj.pix_scale_arcmin
    fit_results = bubbleObj.fit_results

    # Distance used for optional angular-to-physical conversion.
    # If the caller provides a new distance, also store it back into bubbleObj.
    distance_pc = 0
    if distance_pc_item is not None:
        distance_pc = distance_pc_item
        bubbleObj.distance_pc_item = distance_pc_item

    # Extract ellipse size information.
    # ellipse_ra / ellipse_rb are the larger / smaller ellipse semi-axes.
    ellipse_ra = np.max([bubbleObj.ellipse_infor[3], bubbleObj.ellipse_infor[4]])
    ellipse_rb = np.min([bubbleObj.ellipse_infor[3], bubbleObj.ellipse_infor[4]])

    # Fallback characteristic radius scale.
    # Although the geometric-mean radius is another reasonable choice,
    # the current implementation uses the semi-major axis as a conservative
    # shell-size proxy.
    # ellipse_equivalent_r = np.sqrt(ellipse_ra * ellipse_rb)
    ellipse_equivalent_r = ellipse_ra

    # ------------------------------------------------------------------
    # Branch 1: use profile-fit results when the fit is successful and
    # sufficiently symmetric.
    # ------------------------------------------------------------------

    if fit_results['symmetry_score'] > SymScore and ref_ellipse == 'cavity':
        logic = True
    elif ref_ellipse == 'skeleton':
        logic = True
    else:
        logic = False
        # print('ref_ellipse = cavity or skeleton')
        
    if fit_results['success'] and logic:
        p = fit_results['params']
        e = fit_results['errors']

        # Radius = mean absolute distance of the two shell peaks from center.
        bubble_radius = (abs(p['mu1']) + abs(p['mu2'])) / 2

        # Propagate center uncertainties in quadrature, then average.
        radius_error = np.sqrt(e['mu1_err'] ** 2 + e['mu2_err'] ** 2) / 2

        # Convert Gaussian widths to FWHM and average them as shell thickness.
        fwhm1 = 2.355 * p['sigma1']
        fwhm2 = 2.355 * p['sigma2']
        bubble_thickness = (fwhm1 + fwhm2) / 2

        # Propagate width uncertainties in quadrature and convert to FWHM units.
        thickness_error = 2.355 * np.sqrt(e['sigma1_err'] ** 2 + e['sigma2_err'] ** 2) / 2

        # Useful dimensionless diagnostic: shell thickness relative to radius.
        thickness_to_radius_ratio = bubble_thickness / bubble_radius

        # Approximate outer edge of the shell.
        bubble_outer_radius = np.around(bubble_radius + bubble_thickness / 2, 2)

        # Type 1 = accepted fit-based result.
        bubble_params_type = 1

        # Sanity checks for unphysical or unstable fit-derived parameters.
        #
        # Reject fit-based values if:
        # - the inferred outer radius is smaller than the ellipse size,
        # - the shell is excessively thick compared with the radius,
        # - the formal uncertainties exceed the estimated values,
        # - or the shell thickness falls below the imposed minimum.
        if bubble_outer_radius < ellipse_equivalent_r or thickness_to_radius_ratio > 2 or \
           radius_error > bubble_radius or thickness_error > bubble_thickness or \
           np.around(bubble_thickness) < thickness_min:

            # Fallback to geometry-based estimates.
            bubble_radius = np.max([ellipse_equivalent_r, thickness_min])

            # Negative errors are used here as status flags rather than
            # statistical uncertainties:
            # -1 means "fit existed but fallback was enforced".
            radius_error = -1

            # Empirical outer-radius proxy used in fallback mode.
            bubble_outer_radius = 3 / 2 * ellipse_equivalent_r

            bubble_thickness = np.max([ellipse_equivalent_r, thickness_min])
            thickness_error = -1
            thickness_to_radius_ratio = bubble_thickness / bubble_radius

            # Type 2 = fit available, but rejected by sanity checks.
            bubble_params_type = 2

    # ------------------------------------------------------------------
    # Branch 2: fit failed or symmetry is too poor; directly use fallback.
    # ------------------------------------------------------------------
    else :
        bubble_radius = np.max([ellipse_equivalent_r, thickness_min])

        # -2 means "no trustworthy fit-based estimate was available".
        radius_error = -2

        bubble_outer_radius = 3 / 2 * ellipse_equivalent_r
        bubble_thickness = np.max([ellipse_equivalent_r, thickness_min])
        thickness_error = -2
        thickness_to_radius_ratio = bubble_thickness / bubble_radius

        # Type 3 = direct fallback because fit was invalid or too asymmetric.
        bubble_params_type = 3

    # Collect angular-scale results.
    bubble_params = {
        'bubble_params_type': bubble_params_type,
        'radius': bubble_radius,
        'radius_error': radius_error,
        'thickness': bubble_thickness,
        'thickness_error': thickness_error,
        'thickness_to_radius_ratio': thickness_to_radius_ratio,
        'bubble_outer_radius': bubble_outer_radius
    }

    # Convert angular quantities to physical scales when a distance is available.
    # The conversion uses:
    #   angle[arcmin] * pix_scale_arcmin -> arcmin
    #   arcmin * 60 -> arcsec
    #   arcsec / 206265 -> radians (small-angle approximation via tan)
    if distance_pc is not None:
        radius_pc = distance_pc * np.tan(bubble_radius * pix_scale_arcmin * 60 / 206265)
        radius_error_pc = distance_pc * np.tan(radius_error * pix_scale_arcmin * 60 / 206265)
        thickness_pc = distance_pc * np.tan(bubble_thickness * pix_scale_arcmin * 60 / 206265)
        thickness_error_pc = distance_pc * np.tan(thickness_error * pix_scale_arcmin * 60 / 206265)

        bubble_params.update({
            'distance_pc': distance_pc,
            'radius_pc': radius_pc,
            'radius_error_pc': radius_error_pc,
            'thickness_pc': thickness_pc,
            'thickness_error_pc': thickness_error_pc,
            'diameter_pc': radius_pc * 2
        })

    # Save results back to the bubble object for later reuse.
    bubbleObj.bubble_params_RFWHM = bubble_params
    bubbleObj.bubble_outer_radius = bubble_outer_radius
    return bubble_params












