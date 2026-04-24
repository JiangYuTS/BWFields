import time
import numpy as np

from matplotlib.path import Path
from astropy import units as u
from skimage import filters, measure, morphology
import scipy.ndimage as ndimage
from scipy.optimize import least_squares
from tqdm import tqdm
import warnings

# =========================
# Project-specific modules
# =========================
import FacetClumps
import DPConCFil
from DPConCFil import Filament_Class_Funs_Analysis as FCFA

# Local utilities (ellipse/circle helpers, distance helpers, etc.)
from . import Bubble_Funs_Tools as BFTools


def GNC_FacetClumps(core_data, cores_coordinate):
    """
    Gradient-ascent step for FacetClumps-like peak tracing.

    Given a pixel coordinate (x_center, y_center), look at a 3x3 neighborhood
    and move one step towards the neighbor with the maximum positive gradient.

    Parameters
    ----------
    core_data : 2D np.ndarray
        A 2D intensity map (or a padded map).
    cores_coordinate : array-like
        [x, y] coordinate (0-based in the original region coords convention).

    Returns
    -------
    gradients : np.ndarray
        The difference between neighbor pixels and center pixel (flattened neighborhood).
    new_center : list
        Suggested next coordinate [x, y] (0-based), one step towards the steepest ascent.
    """
    xres, yres = core_data.shape

    # NOTE: "+1" here implies core_data may be padded by 1 pixel border elsewhere.
    x_center = cores_coordinate[0] + 1
    y_center = cores_coordinate[1] + 1

    # Build valid neighborhood ranges (clamped to image bounds)
    x_arange = np.arange(max(0, x_center - 1), min(xres, x_center + 2))
    y_arange = np.arange(max(0, y_center - 1), min(yres, y_center + 2))

    # Enumerate all neighborhood coordinates
    x, y = np.meshgrid(x_arange, y_arange)
    xy = np.column_stack([x.flat, y.flat])

    # Gradient relative to the current center pixel
    gradients = core_data[xy[:, 0], xy[:, 1]] - core_data[x_center, y_center]

    # Choose the neighbor with maximum gradient (steepest ascent)
    g_step = np.where(gradients == gradients.max())[0][0]

    # Convert back to original (unpadded) coordinate system
    new_center = list(xy[g_step] - 1)
    return gradients, new_center


def Build_MPR_Dict(origin_data, regions):
    """
    Build "Mountain-Peak-Region" dictionaries by tracing local gradient ascent.

    For each voxel/pixel coordinate inside each region:
      - Follow gradient-ascent steps until no positive ascent is possible.
      - The endpoint is treated as a "peak" for that path.
      - The set of visited coordinates is considered a "mountain" for an ID k.

    Outputs:
      - mountain_array: label map with mountain id for traced coordinates
      - mountain_dict:  mountain_id -> list of coordinates belonging to that mountain
      - peak_dict:      mountain_id -> list of peak coordinate(s)
      - region_mp_dict: region_index -> list of mountain ids appearing in this region

    Notes
    -----
    - This logic assumes a 2D data layout (origin_data is 2D).
    - temp_origin_data is padded by 1 pixel on each side to simplify boundary access.
    """
    k = 1               # mountain id counter
    reg = -1            # region index counter (incremented per region)
    peak_dict = {k: []}
    mountain_dict = {k: []}
    region_mp_dict = {}

    # Keep original_data as-is (commented noise injection removed)
    origin_data = origin_data

    # Label map: 0 means unassigned; positive integers are mountain ids
    mountain_array = np.zeros_like(origin_data)

    # Padded version of origin_data to prevent boundary issues in 3x3 neighborhood
    temp_origin_data = np.zeros(tuple(np.array(origin_data.shape) + 2))
    temp_origin_data[1:temp_origin_data.shape[0] - 1, 1:temp_origin_data.shape[1] - 1] = origin_data

    # Prepare region -> mountain list mapping
    for i in range(len(regions)):
        region_mp_dict[i] = []

    for region in regions:
        reg += 1
        coordinates = region.coords

        for i in range(coordinates.shape[0]):
            temp_coords = []

            # Only trace if this coordinate hasn't been assigned to any mountain yet
            if mountain_array[coordinates[i][0], coordinates[i][1]] == 0:
                temp_coords.append(coordinates[i].tolist())
                mountain_array[coordinates[i][0], coordinates[i][1]] = k

                # One-step ascent
                gradients, new_center = GNC_FacetClumps(temp_origin_data, coordinates[i])

                # If we can ascend, include the next step
                if gradients.max() > 0 and mountain_array[new_center[0], new_center[1]] == 0:
                    temp_coords.append(new_center)

                # Keep ascending while improvement exists and new_center is unassigned
                while gradients.max() > 0 and mountain_array[new_center[0], new_center[1]] == 0:
                    mountain_array[new_center[0], new_center[1]] = k
                    gradients, new_center = GNC_FacetClumps(temp_origin_data, new_center)

                    if gradients.max() > 0 and mountain_array[new_center[0], new_center[1]] == 0:
                        temp_coords.append(new_center)

                # Assign all traced coords the final mountain id we ended at
                mountain_array[np.stack(temp_coords)[:, 0], np.stack(temp_coords)[:, 1]] = \
                    mountain_array[new_center[0], new_center[1]]

                # Record coords into that mountain's coordinate list
                mountain_dict[mountain_array[new_center[0], new_center[1]]] += temp_coords

                # If no further ascent is possible, treat new_center as a peak
                if gradients.max() <= 0:
                    peak_dict[k].append(new_center)
                    region_mp_dict[reg].append(k)

                    # Start a new mountain id bucket
                    k += 1
                    mountain_dict[k] = []
                    peak_dict[k] = []

    # Remove the last empty bucket created after the final increment
    del (mountain_dict[k])
    del (peak_dict[k])

    return mountain_array, mountain_dict, peak_dict, region_mp_dict


def Fit_Ellipse_Algebraic(points, center=None):
    """
    Algebraic ellipse fitting with (optional) fixed center using least squares.

    Model:
        A*x^2 + B*x*y + C*y^2 = 1   (after shifting by center)

    Parameters
    ----------
    points : (N,2) array
        Input contour points (x,y).
    center : tuple or None
        Fixed (xc,yc). If None, uses mean of points.

    Returns
    -------
    coords_fit : np.ndarray or list
        Fitted ellipse contour coordinates (depends on Generate_Ellipse_Points implementation).
    ellipse_infor : list
        Typically [xc, yc, angle_deg or rad, a, b] depending on helper implementation.
    """
    if center is None:
        center = np.mean(points, axis=0)

    xc, yc = center
    x = points[:, 0] - xc
    y = points[:, 1] - yc

    # Solve D @ [A,B,C] = 1 in least squares sense
    D = np.column_stack([x ** 2, x * y, y ** 2])
    params = np.linalg.lstsq(D, np.ones(len(points)), rcond=None)[0]

    A, B, C = params

    # Rotation angle from quadratic form
    theta = 0.5 * np.arctan2(B, A - C)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Rotate quadratic form to get principal coefficients
    A_rot = A * cos_t ** 2 + B * cos_t * sin_t + C * sin_t ** 2
    C_rot = A * sin_t ** 2 - B * cos_t * sin_t + C * cos_t ** 2

    # Semi-axes (in pixels)
    a = 1.0 / np.sqrt(A_rot)
    b = 1.0 / np.sqrt(C_rot)

    # Ensure a>=b and adjust theta accordingly
    if a < b:
        a, b = b, a
        theta += np.pi / 2

    theta = theta % (2 * np.pi) 
    if theta > np.pi/2 and theta <= 3*np.pi/2:
        theta = theta - np.pi
    elif theta > 3*np.pi/2:
        theta = theta - 2*np.pi
    theta = max(min(theta, np.pi/2), -np.pi/2)

    # NOTE: This calls Generate_Ellipse_Points; in your repo it may be in BFTools.
    coords_fit, ellipse_infor = BFTools.Generate_Ellipse_Points(a, b, theta, center)
    return coords_fit, ellipse_infor


def Fit_Ellipse_Geometric(points, center=None, initial_guess=None,
                                 axis_ratio_min=0.2, axis_scale_max=1.2,
                                 return_params_only=False):
    """
    Stable geometric ellipse fitting with optional fixed center.

    Residual for each point:
        (x_rot/a)^2 + (y_rot/b)^2 - 1

    Improvements over the original version
    --------------------------------------
    1) Add bounds on a, b, theta to prevent degenerate solutions.
    2) Use 'trf' solver because 'lm' does not support bounds.
    3) Use data-driven initialization.
    4) Force a >= b and normalize theta into [-pi/2, pi/2].

    Parameters
    ----------
    points : (N,2) ndarray
        Input contour/skeleton points (x, y).
    center : tuple or None
        Fixed ellipse center (xc, yc). If None, uses mean of points.
    initial_guess : list or None
        Initial guess [a, b, theta].
    axis_ratio_min : float
        Minimum allowed b/a ratio. Helps avoid very thin degenerate ellipses.
    axis_scale_max : float
        Maximum allowed semi-axis relative to data span.
        Example: 1.2 means axes cannot exceed 1.2 * max(data_span).
    return_params_only : bool
        If True, return only ellipse_infor and None.

    Returns
    -------
    ellipse_infor : list
        [xc, yc, theta, a, b]
    coords_fit : ndarray or None
        Fitted ellipse coordinates if BFTools.Generate_Ellipse_Points exists,
        otherwise None.
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an (N,2) array")

    if len(points) < 5:
        raise ValueError("At least 5 points are required to fit an ellipse")

    # Fixed center: use mean if not provided
    if center is None:
        center = np.mean(points, axis=0)
    center = np.asarray(center, dtype=float)

    # Shift points to local coordinates
    x = points[:, 0] - center[0]
    y = points[:, 1] - center[1]

    # Estimate data scale
    x_span = np.ptp(x)   # max-min
    y_span = np.ptp(y)
    max_span = max(x_span, y_span, 1e-6)

    # Sensible bounds for semi-axes
    a_min = max(0.5, 0.10 * max_span)
    a_max = max(2.0, axis_scale_max * max_span)

    b_min = max(0.5, axis_ratio_min * a_min)
    b_max = max(2.0, axis_scale_max * max_span)

    # Data-driven initial guess
    if initial_guess is None:
        cov = np.cov(np.c_[x, y].T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Rough axis estimate from variance
        a0 = np.sqrt(max(eigvals[0], 1e-6)) * 2.0
        b0 = np.sqrt(max(eigvals[1], 1e-6)) * 2.0

        # Clamp into bounds
        a0 = np.clip(a0, a_min, a_max)
        b0 = np.clip(b0, b_min, min(b_max, a0))

        theta0 = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        initial_guess = [a0, b0, theta0]

    def residuals(params):
        a, b, theta = params
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Rotate points into ellipse frame
        x_rot = cos_t * x + sin_t * y
        y_rot = -sin_t * x + cos_t * y

        # Implicit ellipse residual
        res = (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1.0

        # Mild regularization against extreme axis separation
        # This is soft, not mandatory, and mainly helps stability.
        reg = 0.01 * np.log(a / b)
        return np.r_[res, reg]

    # Bounds: enforce positive axes and restrict theta
    lower_bounds = [a_min, b_min, -np.pi-0.1]
    upper_bounds = [a_max, b_max,  np.pi+0.1]

    result = least_squares(
        residuals,
        x0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        method='trf',
        loss='soft_l1')

    a, b, theta = result.x

    # Ensure a >= b
    if a < b:
        a, b = b, a
        theta += np.pi / 2

    # Normalize theta to [-pi/2, pi/2]
    theta = theta % (2 * np.pi)
    if theta > np.pi / 2 and theta <= 3 * np.pi / 2:
        theta -= np.pi
    elif theta > 3 * np.pi / 2:
        theta -= 2 * np.pi
    theta = np.clip(theta, -np.pi / 2, np.pi / 2)

    ellipse_infor = [center[0], center[1], theta, a, b]

    coords_fit = None
    if not return_params_only:
        try:
            coords_fit, ellipse_infor = BFTools.Generate_Ellipse_Points(a, b, theta, center)
        except Exception:
            coords_fit = None

    return ellipse_infor, coords_fit


def Get_Ellipse_Coords(bubble_com_bl, bubble_region, bubble_weight_data):
    """
    Fit an ellipse to a bubble region contour on the (b,l) plane.

    Steps
    -----
    1) Project region voxel coords to a 2D mask (b,l).
    2) Extract contour points via skimage.measure.find_contours.
    3) Fit ellipse geometrically with a fixed center (bubble_com_bl).
       If it fails, fall back to auto center.

    Parameters
    ----------
    bubble_com_bl : tuple
        Expected ellipse center in (b,l) pixel coordinates (note: uses contour coordinate convention).
    bubble_region : skimage regionprops-like object
        Contains .coords (N,3) = (v,b,l) coords.
    bubble_weight_data : 3D np.ndarray
        Bubble weight cube used to infer shape/extent.

    Returns
    -------
    ellipse_infor : list
        [x0, y0, angle, a, b] in 2D (contour coordinate system).
    ellipse_coords : np.ndarray
        Ellipse contour coordinates.
    """
    # Voxel coordinates for this bubble
    coords_i = bubble_region.coords

    # # Build 2D mask over (b,l) plane (shape: [B, L])
    # contour_data = np.zeros((bubble_weight_data.shape[1], bubble_weight_data.shape[2]), dtype='uint16')
    # contour_data[coords_i[:, 1], coords_i[:, 2]] = 1

    # # Slight morphology to smooth boundary before contouring
    # contour_data = morphology.dilation(contour_data, morphology.disk(1))
    # contour_data = morphology.erosion(contour_data, morphology.disk(1))

    # # Extract contours; concatenate all contour segments
    # contour = measure.find_contours(contour_data, 0.5)
    # contour_temp = []
    # for i in range(len(contour)):
    #     contour_temp += list(contour[i])
    # contour = np.array(contour_temp)
    
    _, _, _, contour, _ = Cal_2D_Region_From_3D_Coords(coords_i, cal_contours=True)

    # Try ellipse fitting with fixed center first
    try:
        ellipse_infor, ellipse_coords = Fit_Ellipse_Geometric(contour, center=bubble_com_bl, initial_guess=None)

        # Fail fast if NaNs appear
        if np.isnan(ellipse_infor[3]) or np.isnan(ellipse_infor[4]):
            raise ValueError("Fitting result contains NaN")

    except Exception as e:
        warnings.warn(f"Fitting with fixed center failed: {str(e)}, trying automatic center calculation")

        # Fallback: use contour mean as center (note: original code uses contour; kept as-is)
        bubble_com_bl = np.mean(contour, axis=0)
        ellipse_infor, ellipse_coords = Fit_Ellipse_Geometric(contour, center=bubble_com_bl, initial_guess=None)

    # Optional sampling (currently unused in return)
    delta_sample = 1
    ellipse_coords_sample = ellipse_coords[::delta_sample]

    return ellipse_infor, ellipse_coords


def Get_Bubble_Ellipse_Infor_By_Provide(bubbleObj, bubble_infor_provided):
    """
    Build a "default" ellipse/circle description from provided (or partially missing) bubble info.

    bubble_infor_provided is expected as:
        [bubble_clump_ids, bubble_com_pi, bubble_com_wcs_pi, bubble_radius_p]

    If center/radius is None, it is derived from clump centers.

    Returns
    -------
    bubble_com_pi : (v,b,l) center in pixel
    bubble_com_wcs_pi : (l,b,v) or similar in WCS units (depends on upstream conventions)
    bubble_radius_p : float
    bubble_contour_p : np.ndarray
        Circle points for the provided radius
    ellipse_infor : list
        [b_center, l_center, angle=0, a=radius, b=radius]
    """
    bubble_clump_ids = bubble_infor_provided[0]
    bubble_com_pi = bubble_infor_provided[1]
    bubble_com_wcs_pi = bubble_infor_provided[2]
    bubble_radius_p = bubble_infor_provided[3]

    # If center not provided: use mean of the clump centers (pixel space)
    if bubble_com_pi is None:
        clump_centers_p = bubbleObj.clumpsObj.centers[bubble_clump_ids]
        bubble_com_pi = clump_centers_p.mean(axis=0)

    # If WCS center not provided: use mean of WCS centers
    if bubble_com_wcs_pi is None:
        bubble_com_wcs_pi = bubbleObj.clumpsObj.centers_wcs[bubble_clump_ids].mean(axis=0)

    # If radius not provided: estimate from RMS spread of clump centers around bubble center
    if bubble_radius_p is None:
        bubble_radius_p = np.sqrt(((clump_centers_p - bubble_com_pi) ** 2)[:, 1:].mean(axis=0).sum())

    # Build a circle contour (note: expects arguments in (x,y,r) order per helper)
    bubble_contour_p, circumference = BFTools.Generate_Circle_Points(
        bubble_com_pi[1], bubble_com_pi[2], bubble_radius_p
    )

    # Add a tiny noise to avoid degenerate geometry issues (kept from original code)
    bubble_contour_p += np.random.random(bubble_contour_p.shape) / 10 ** 5

    # Represent as a "zero-rotation" ellipse with a=b=radius
    ellipse_infor = [bubble_com_pi[1], bubble_com_pi[2], 0, bubble_radius_p, bubble_radius_p]
    return bubble_com_pi, bubble_com_wcs_pi, bubble_radius_p, bubble_contour_p, ellipse_infor


def Cal_2D_Region_From_3D_Coords(coords, cal_contours=False):
    """
    Project a set of 3D voxel coordinates (v,b,l) to a 2D mask on (b,l) plane,
    and optionally compute outer contours and boundary coordinates.

    Parameters
    ----------
    coords : (N,3) np.ndarray
        Voxel coordinates [v,b,l].
    cal_contours : bool
        If True, compute contour polyline and boundary coordinates.

    Returns
    -------
    box_data : 2D np.ndarray
        2D mask in (b,l) sub-box with padding.
    coords_range : np.ndarray
        [vmin,vmax,bmin,bmax,lmin,lmax] for the original 3D coords.
    box_region_max : regionprops object
        Largest connected component in the 2D mask.
    contours_max : np.ndarray
        Longest contour polyline coordinates in global (b,l) frame (if requested).
    boundary_coords : np.ndarray
        Boundary coords in global (b,l) frame (if requested).
    """
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()

    # v extent (currently computed but not used downstream here)
    v_delta = x_max - x_min + 1
    coords_range = np.array([x_min, x_max, y_min, y_max, z_min, z_max])

    # Build 2D mask (b,l) with a small padding
    box_data = np.zeros([y_max - y_min + 5, z_max - z_min + 5])
    box_data[coords[:, 1] - y_min + 2, coords[:, 2] - z_min + 2] = 1

    # Connected component labeling in 2D
    box_label = measure.label(box_data, connectivity=2)
    box_regions = measure.regionprops(box_label)

    # Choose the largest region if multiple components exist
    box_region_max = box_regions[0]
    if len(box_regions) != 1:
        region_sizes = [len(region.coords) for region in box_regions]
        box_region_max = box_regions[np.argmax(region_sizes)]

    contours_max = np.array([0, 0])
    boundary_coords = np.array([0, 0])

    if cal_contours:
        # Extract outer contours of the labeled map
        contours = measure.find_contours(box_label, level=0.5, fully_connected='high')
        if len(contours) != 0:
            for i in range(len(contours)):
                if len(contours_max) < len(contours[i]):
                    contours_max = contours[i]
            if len(contours_max) != 0:
                # Map contour back to global (b,l) plane coordinates
                contours_max = np.array(contours_max) + np.array([y_min - 2, z_min - 2])

        # Build boundary coords by subtracting eroded interior from mask
        bub_mask_erosion_boundary = morphology.binary_erosion(box_data, morphology.disk(1))
        contour_data = box_data * ~bub_mask_erosion_boundary

        regions = measure.regionprops(morphology.label(contour_data))
        boundary_coords = []
        for region in regions:
            boundary_coords += region.coords.tolist()
        boundary_coords = np.array(boundary_coords) + np.array([y_min - 2, z_min - 2])

    return box_data, coords_range, box_region_max, contours_max, boundary_coords


def Get_Singals(origin_data, Sigma=None, threshold=1):
    """
    Build a binary "signal" mask from a 3D cube using threshold + morphology (+ optional Gaussian smoothing).

    Parameters
    ----------
    origin_data : 3D np.ndarray
        Intensity cube.
    Sigma : float or None
        If provided, apply Gaussian smoothing after thresholding.
    threshold : float
        Base threshold for detecting voxels.

    Returns
    -------
    dilation_data : 3D np.ndarray
        Binary mask (0/1) after opening + dilation (and optional Gaussian constraint).
    """
    kernal_radius = 1

    # Initial binary mask then opening to remove small noise
    open_data = morphology.opening(origin_data > threshold, morphology.ball(kernal_radius))

    # Dilation to reconnect near components
    dilation_data = morphology.dilation(open_data, morphology.ball(kernal_radius))

    if Sigma is not None:
        # Zero out below-threshold before smoothing (preserves strong cores)
        origin_data[origin_data < threshold] = 0
        filter_data = filters.gaussian(origin_data, Sigma)

        # Require smoothed signal to be above threshold too
        dilation_data &= (filter_data > threshold)

    dilation_data[dilation_data > 0] = 1
    return dilation_data


def Get_Bubble_2D_Coords(label_sum_i, erosion_logic, margin=5):
    """
    Given a 2D binary mask (typically sum over velocity slices),
    detect "holes" inside connected components (potential bubble interiors).

    The function:
      - labels connected regions
      - for each region, fills holes and extracts hole components
      - optionally erodes holes to avoid thin artifacts
      - returns hole coordinates in the original mask frame

    Parameters
    ----------
    label_sum_i : 2D np.ndarray
        Binary 2D mask.
    erosion_logic : bool
        Whether to erode the holes (morphology.erosion) before labeling them.
    margin : int
        Extra padding around a region bbox when extracting a submask.

    Returns
    -------
    bubble_regions_coords : list of np.ndarray
        Each entry is (Nhole,2) coords for a detected hole (bubble interior) in global coordinates.
    """
    mask = label_sum_i.astype(bool)
    if not mask.any():
        return []

    h, w = mask.shape
    label_all = measure.label(mask, connectivity=2)
    regions_all = measure.regionprops(label_all)

    bubble_regions_coords = []

    for reg in regions_all:
        # Euler number == 1 often indicates no holes (depends on topology); skip those
        if reg.euler_number == 1:
            continue

        # Extract submask around this region for local hole filling
        minr, minc, maxr, maxc = reg.bbox
        minr_e = max(0, minr - margin)
        minc_e = max(0, minc - margin)
        maxr_e = min(h, maxr + margin)
        maxc_e = min(w, maxc + margin)

        sub_label = label_all[minr_e:maxr_e, minc_e:maxc_e]
        sub_mask = (sub_label == reg.label)
        if not sub_mask.any():
            continue

        # Fill holes and isolate hole pixels
        filled = ndimage.binary_fill_holes(sub_mask)
        holes = filled & (~sub_mask)

        if erosion_logic:
            holes = morphology.erosion(holes, morphology.disk(1))

        if not holes.any():
            continue

        # Label holes and record each hole component's coords
        holes_label = measure.label(holes.astype(np.uint8), connectivity=2)
        sub_regions = measure.regionprops(holes_label)
        for sreg in sub_regions:
            coords = sreg.coords.copy()
            coords[:, 0] += minr_e  # map to global row
            coords[:, 1] += minc_e  # map to global col
            bubble_regions_coords.append(coords)

    return bubble_regions_coords


def Get_Bubble_2D(label_sum):
    """
    Run Get_Bubble_2D_Coords on the original mask and two morphological variants
    (eroded and dilated), then combine results.

    This helps stabilize hole detection across slightly different boundaries.
    """
    bubble_regions_record = []
    label_sum[label_sum > 0] = 1

    label_sum_erosion = morphology.erosion(label_sum, morphology.disk(1))
    label_sum_dilation = morphology.dilation(label_sum, morphology.disk(1))

    erosion_logics = [False, False, True]
    for label_sum_i, erosion_logic in zip([label_sum, label_sum_erosion, label_sum_dilation], erosion_logics):
        bubble_regions_i = Get_Bubble_2D_Coords(label_sum_i, erosion_logic)
        bubble_regions_record += bubble_regions_i

    return bubble_regions_record


def Get_Bubble_Weighted_Data(RMS, Threshold, Sigma, SlicedV, sr_data_i):
    """
    Build a 3D "bubble weight" cube inside a single SR (signal region) sub-cube.

    The idea (as implemented):
      - iterate SNR thresholds (Threshold_i = SNR_i * RMS)
      - create a 3D binary signal mask (optionally Gaussian-constrained)
      - slice the velocity axis into chunks of various widths (delta_v = 1..SlicedV)
      - project each chunk to 2D, detect hole regions (bubbles) via Get_Bubble_2D
      - add contributions to bubble_weight_data using:
          * a discrete term SNR_i/delta_v
          * a small local Gaussian weight (weight_gauss_local)

    Parameters
    ----------
    RMS, Threshold, Sigma : float
        Noise / thresholding parameters.
    SlicedV : int
        Maximum velocity slicing factor (try multiple chunk sizes).
    sr_data_i : 3D np.ndarray
        Sub-cube (v,b,l) extracted around a signal region.

    Returns
    -------
    bubble_weight_data : 3D np.ndarray
        Accumulated bubble weights for this SR sub-cube.
    """
    SNR_min = np.int64(Threshold / RMS)
    SNR_max = np.int64(sr_data_i.max() / RMS)

    bubble_weight_data = np.zeros_like(sr_data_i)
    len_v = sr_data_i.shape[0]
    end_id = len_v + 1

    # Try delta_v = 1..SlicedV
    SlicedVS = np.arange(1, SlicedV + 1)

    for SNR_i in range(SNR_min, SNR_max):
        Threshold_i = SNR_i * RMS

        # 3D binary signal mask under the current threshold setting
        singal_mask = Get_Singals(sr_data_i, Sigma, Threshold_i)

        for delta_v in SlicedVS:
            # Choose slice alignment (centered)
            start_id = np.int64((len_v % delta_v) / 2)
            sliced_ids = np.arange(start_id, end_id, delta_v)

            for i in range(len(sliced_ids) - 1):
                # Sum signal presence across velocity slab -> 2D mask
                singal_mask_sum = singal_mask[sliced_ids[i]:sliced_ids[i + 1], :, :].sum(0)

                # Detect bubble-like holes on this 2D projection
                bubble_regions_coords = Get_Bubble_2D(singal_mask_sum)

                for rb_coords in bubble_regions_coords:
                    # Approximate bubble "center" in (v,b,l)
                    bubble_mor_center = (
                        (sliced_ids[i] + sliced_ids[i + 1]) // 2,
                        rb_coords[:, 0].mean().astype(np.int64),
                        rb_coords[:, 1].mean().astype(np.int64)
                    )

                    # Build a local Gaussian weighting in this slab around the bubble center
                    z_coords = np.arange(sliced_ids[i], sliced_ids[i + 1], dtype=np.int64)
                    r_sq = (z_coords.reshape(len(z_coords), 1) - bubble_mor_center[0]) ** 2 + \
                           (rb_coords[:, 0].reshape(1, len(rb_coords)) - bubble_mor_center[1]) ** 2 + \
                           (rb_coords[:, 1].reshape(1, len(rb_coords)) - bubble_mor_center[2]) ** 2

                    weight_gauss_local = np.exp(-r_sq / (2 * SNR_i ** 2)) / 1000

                    # Add discrete + Gaussian weights into the bubble_weight_data slab
                    bubble_weight_data[sliced_ids[i]:sliced_ids[i + 1], rb_coords[:, 0], rb_coords[:, 1]] += SNR_i / delta_v
                    bubble_weight_data[sliced_ids[i]:sliced_ids[i + 1], rb_coords[:, 0], rb_coords[:, 1]] += weight_gauss_local

    return bubble_weight_data


def Update_Bubble_Weight_Data(bubble_weight_data, BubbleSize, BubbleWeight):
    """
    Filter/cleanup bubble_weight_data and extract final bubble voxel coordinate sets.

    Main operations:
      1) Find connected components of (bubble_weight_data > 0) in 3D.
      2) For each component, threshold by BubbleWeight (remove weak voxels).
      3) Apply morphology opening + dilation for cleanup.
      4) Re-label subcomponents and keep only those with projected 2D area > BubbleSize.

    Parameters
    ----------
    bubble_weight_data : 3D np.ndarray
        Input weight cube.
    BubbleSize : int or float
        Minimum projected 2D area (pixels) to keep a bubble.
    BubbleWeight : float
        Minimum voxel weight threshold.

    Returns
    -------
    bubble_weight_data_out : 3D np.ndarray
        Output cube with weights only at kept bubble voxels (else 0).
    bubbles_coords : list of (N,3) arrays
        Each entry is the voxel coords of a kept bubble region.
    """
    # Output: keep weights only for accepted bubbles
    bubble_weight_data_out = np.zeros_like(bubble_weight_data)
    bubbles_coords = []

    # Structuring element for 3D morphology
    selem = morphology.ball(1)

    # Candidate voxels
    mask_all = bubble_weight_data > 0
    if not mask_all.any():
        return bubble_weight_data_out, bubbles_coords

    # Global connected components (3D)
    label_all = measure.label(mask_all.astype(np.uint8), connectivity=2)
    regions_all = measure.regionprops(label_all)

    for reg in regions_all:
        # Work within bounding box for efficiency
        zmin, ymin, xmin, zmax, ymax, xmax = reg.bbox
        sub_slices = np.s_[zmin:zmax, ymin:ymax, xmin:xmax]

        sub_data = bubble_weight_data[sub_slices]

        # Weight thresholding
        sub_mask = sub_data > BubbleWeight
        if not sub_mask.any():
            continue

        # Morphology cleanup
        sub_mask = morphology.opening(sub_mask, selem)
        sub_mask = morphology.dilation(sub_mask, selem)
        if not sub_mask.any():
            continue

        # Re-label subcomponents within this bbox
        sub_label = measure.label(sub_mask.astype(np.uint8), connectivity=2)
        sub_regions = measure.regionprops(sub_label)

        for sreg in sub_regions:
            sub_coords = sreg.coords  # local coords (z,y,x) within bbox

            # Map back to global coords
            g_coords = np.empty_like(sub_coords)
            g_coords[:, 0] = sub_coords[:, 0] + zmin
            g_coords[:, 1] = sub_coords[:, 1] + ymin
            g_coords[:, 2] = sub_coords[:, 2] + xmin

            # Project to 2D and apply size threshold
            box_data, coords_range, box_region, contours, boundary_coords = Cal_2D_Region_From_3D_Coords(g_coords)
            if box_region.area <= BubbleSize:
                continue

            # Keep this bubble region
            bubbles_coords.append(g_coords)
            idx = (g_coords[:, 0], g_coords[:, 1], g_coords[:, 2])
            bubble_weight_data_out[idx] = bubble_weight_data[idx]

    return bubble_weight_data_out, bubbles_coords


def Bubble_Weight_Data_Detect_By_SR(srs_list, origin_data, parameters):
    """
    Build bubble weight cubes (raw and filtered) by iterating over SR regions.

    For each SR region:
      - Extract a padded sub-cube around that SR
      - Compute bubble_weight_data inside that sub-cube
      - Save two versions:
          (1) no filter (BubbleSize=0, BubbleWeight=0)
          (2) filtered using BubbleSize and BubbleWeight from parameters

    Parameters
    ----------
    srs_list : list of regionprops
        Signal region objects. Each has .coords (N,3) in (v,b,l).
    origin_data : 3D np.ndarray
        Full intensity cube.
    parameters : list/array
        [RMS, Threshold, Sigma, SlicedV, BubbleSize, BubbleWeight]

    Returns
    -------
    bubble_weight_data_array_no_filter : 3D np.ndarray
    bubble_weight_data_array : 3D np.ndarray
    bubbles_coords_record : list of arrays
        Kept bubble voxel coords in full-cube coordinates.
    delta_times : list of float
        Per-SR processing time (seconds).
    """
    RMS, Threshold, Sigma, SlicedV, BubbleSize, BubbleWeight = parameters

    delta_times = []
    bubbles_coords_record = []

    bubble_weight_data_array_no_filter = np.zeros_like(origin_data)
    bubble_weight_data_array = np.zeros_like(origin_data)

    regions_array_shape = origin_data.shape

    for index in tqdm(range(len(srs_list))):
        start_1_i = time.time()

        coords_item = srs_list[index].coords

        # Extract a slightly expanded bbox around the SR
        l_max = np.min([coords_item[:, 2].max() + 3, regions_array_shape[2]])
        l_min = np.max([0, coords_item[:, 2].min() - 3])
        b_max = np.min([coords_item[:, 1].max() + 3, regions_array_shape[1]])
        b_min = np.max([0, coords_item[:, 1].min() - 3])
        v_max = np.min([coords_item[:, 0].max() + 3, regions_array_shape[0]])
        v_min = np.max([0, coords_item[:, 0].min() - 3])

        # Build SR sub-cube with values only where SR coords exist
        sr_data_i = np.zeros([v_max - v_min, b_max - b_min, l_max - l_min])
        sr_data_i[coords_item[:, 0] - v_min, coords_item[:, 1] - b_min, coords_item[:, 2] - l_min] = \
            origin_data[coords_item[:, 0], coords_item[:, 1], coords_item[:, 2]]

        # Compute bubble weights in SR sub-cube
        bubble_weight_data = Get_Bubble_Weighted_Data(RMS, Threshold, Sigma, SlicedV, sr_data_i)

        # Save "no-filter" version
        bubble_weight_data_nf, bubbles_coords_nf = Update_Bubble_Weight_Data(
            bubble_weight_data, BubbleSize=0, BubbleWeight=0
        )
        if len(bubbles_coords_nf) != 0:
            for bubble_coords in bubbles_coords_nf:
                bubble_coords = (bubble_coords[:, 0], bubble_coords[:, 1], bubble_coords[:, 2])
                bubble_weight_data_array_no_filter[
                    bubble_coords[0] + v_min, bubble_coords[1] + b_min, bubble_coords[2] + l_min
                ] = bubble_weight_data_nf[bubble_coords]

        # Save filtered version
        bubble_weight_data_f, bubbles_coords_f = Update_Bubble_Weight_Data(
            bubble_weight_data, BubbleSize, BubbleWeight
        )
        if len(bubbles_coords_f) != 0:
            for bubble_coords in bubbles_coords_f:
                bubble_coords = (bubble_coords[:, 0], bubble_coords[:, 1], bubble_coords[:, 2])
                bubble_weight_data_array[
                    bubble_coords[0] + v_min, bubble_coords[1] + b_min, bubble_coords[2] + l_min
                ] = bubble_weight_data_f[bubble_coords]

                # Record full-cube coords (N,3)
                bubble_coords_array = np.c_[bubble_coords[0] + v_min, bubble_coords[1] + b_min, bubble_coords[2] + l_min]
                bubbles_coords_record.append(bubble_coords_array)

        end_1_i = time.time()
        delta_times.append(np.around(end_1_i - start_1_i, 2))

    return bubble_weight_data_array_no_filter, bubble_weight_data_array, bubbles_coords_record, delta_times


def Get_Bubble_Regions_By_FacetClumps(srs_array, bubble_weight_data_no_filter, bubble_weight_data, par_FacetClumps_bub):
    """
    Run FacetClumps on candidate bubble-weight regions to segment them into bubble clumps.

    Workflow:
      1) Label candidate voxels (bubble_weight_data_no_filter > 0)
      2) For each labeled component, extract sub-cube and run FacetClumps detection
      3) Merge all detected clumps into a global clump_mask_array
      4) Post-process merging using Update_Bubble_Regions_Data

    Parameters
    ----------
    srs_array : 3D np.ndarray
        Original SR label array for the full cube (used later in merging).
    bubble_weight_data_no_filter : 3D np.ndarray
        Candidate bubble voxels (looser selection).
    bubble_weight_data : 3D np.ndarray
        Filtered/weighted bubble cube.
    par_FacetClumps_bub : list/array
        Parameter set for FacetClumps post-detection / merging behavior.

    Returns
    -------
    bubble_regions_data : 3D np.ndarray (uint32)
        Final bubble region label cube.
    bubbles_coords_record : list of arrays
        Voxel coords per bubble region.
    """
    # Fixed FacetClumps parameters (as in original code)
    SWindow = 3
    KBins = 35
    FwhmBeam = 2
    VeloRes = 2

    # Parameters packed in input
    RMS_bub = par_FacetClumps_bub[1]
    Threshold_bub = par_FacetClumps_bub[1]
    SRecursionLBV_bub = par_FacetClumps_bub[2]
    MergeArea = par_FacetClumps_bub[3]

    # Label candidate bubble volumes (3D connected components)
    srs_array_bub = measure.label(bubble_weight_data_no_filter > 0, connectivity=3)
    srs_list_bub = measure.regionprops(srs_array_bub)

    numbers = []
    bubble_peaks = []
    bubble_coms = []

    # This will store the combined FacetClumps labels across all components
    clump_mask_array = np.zeros_like(bubble_weight_data, dtype='uint32')

    regions_array_shape = bubble_weight_data.shape

    for index in tqdm(range(len(srs_list_bub))):
        coords_item = srs_list_bub[index].coords

        # Component bbox extraction with small padding
        l_max = np.min([coords_item[:, 2].max() + 3, regions_array_shape[2]])
        l_min = np.max([0, coords_item[:, 2].min() - 3])
        b_max = np.min([coords_item[:, 1].max() + 3, regions_array_shape[1]])
        b_min = np.max([0, coords_item[:, 1].min() - 3])
        v_max = np.min([coords_item[:, 0].max() + 3, regions_array_shape[0]])
        v_min = np.max([0, coords_item[:, 0].min() - 3])

        # Sub-cube containing weights only for this component
        signal_region_data_i = np.zeros([v_max - v_min, b_max - b_min, l_max - l_min])
        signal_region_data_i[coords_item[:, 0] - v_min, coords_item[:, 1] - b_min, coords_item[:, 2] - l_min] = \
            bubble_weight_data[coords_item[:, 0], coords_item[:, 1], coords_item[:, 2]]

        # Run FacetClumps detection on this sub-cube
        detect_infor_dict = FacetClumps.FacetClumps_3D_Funs.Detect_FacetClumps(
            RMS_bub, Threshold_bub, SWindow, KBins, FwhmBeam, VeloRes, SRecursionLBV_bub, signal_region_data_i
        )

        # Accumulate results if any peaks were found
        if len(detect_infor_dict['peak_location']) != 0:
            number = len(detect_infor_dict['peak_location'])
            regions_data = detect_infor_dict['regions_data']
            regions_list = measure.regionprops(regions_data)

            # Write sub-labels into global clump_mask_array with proper id offset
            if len(numbers) == 0:
                base = 0
            else:
                base = int(np.sum(numbers))

            for i in range(len(regions_list)):
                coords = regions_list[i].coords
                clump_mask_array[coords[:, 0] + v_min, coords[:, 1] + b_min, coords[:, 2] + l_min] = i + 1 + base

            bubble_peaks += (np.array(detect_infor_dict['peak_location']) + np.array([v_min, b_min, l_min])).tolist()
            bubble_coms += (np.array(detect_infor_dict['clump_com']) + np.array([v_min, b_min, l_min])).tolist()
            numbers.append(number)

    bubble_peaks = np.array(bubble_peaks)
    bubble_coms = np.array(bubble_coms)

    # Merge / clean up bubble clumps using overlap/connectivity logic
    bubble_regions_data, bubbles_coords_record = Update_Bubble_Regions_Data(
        bubble_weight_data, srs_array, clump_mask_array,
        srs_array_bub, srs_list_bub, bubble_peaks, bubble_coms, MergeArea
    )
    return bubble_regions_data, bubbles_coords_record, srs_array_bub, srs_list_bub


def Update_Bubble_Regions_Data(bubble_weight_data, srs_array, bubble_regions_data,
    srs_array_bub, srs_list_bub, bubble_peaks, bubble_coms, MergeArea=0.4, MergePeak=3):
    """
    Merge bubble clumps if they overlap strongly in (b,l) and are connected.

    Key ideas:
      - Build RC (region-cluster) dict that groups clumps inside each candidate component
      - For clumps in the same group: compare their 2D projected overlap (IOU-like)
      - If overlap > MergeArea and connectivity==1, merge smaller into larger label

    Parameters
    ----------
    bubble_weight_data : 3D np.ndarray
        Weight cube, used for peak value comparisons.
    srs_array : 3D np.ndarray
        (Unused in this function’s current body; kept for signature consistency)
    bubble_regions_data : 3D np.ndarray
        Initial clump labels to be merged.
    srs_array_bub, srs_list_bub : outputs from candidate labeling
    bubble_peaks, bubble_coms : arrays
        Peak voxel coords and center-of-mass coords for clumps.
    MergeArea : float
        Overlap threshold used to decide merge.
    MergePeak : float
        Peak ratio threshold (currently not enforced; logic commented in code).

    Returns
    -------
    bubble_regions_data : 3D np.ndarray
        Relabeled/merged bubble regions.
    bubbles_coords_record : list of arrays
        Final coords for each region.
    """
    # Build mapping from candidate component -> list of clump ids
    rc_dict_bub = DPConCFil.Clump_Class_Funs.Build_RC_Dict_Simplified(bubble_peaks, srs_array_bub, srs_list_bub)

    # clump_id -> coords
    clump_coords_dict_bub = {}
    clump_coords_dict_bub_T = {}  # temporary merged coords

    clumps_list_bub = measure.regionprops(bubble_regions_data)
    for i in range(len(clumps_list_bub)):
        clump_coords_dict_bub[i] = clumps_list_bub[i].coords

    # Sort clumps inside each component by nearest-neighbor distances (for stable merging order)
    for key in rc_dict_bub.keys():
        if len(rc_dict_bub[key]) > 1:
            sorted_indices, sorted_coms, min_distances = BFTools.Sort_By_Nearest_Neighbor_Dists(
                bubble_coms[:, 1:][rc_dict_bub[key]]
            )
            rc_dict_bub[key] = np.array(rc_dict_bub[key])[sorted_indices]

    # Pairwise merge decisions inside each component group
    for key in rc_dict_bub.keys():
        if len(rc_dict_bub[key]) > 1:
            for key_i in rc_dict_bub[key]:
                dists = BFTools.Dists_Array([bubble_coms[key_i][1:]], bubble_coms[rc_dict_bub[key]][:, 1:])
                dists_index_sort = np.argsort(dists[0])
                key_js = np.array(rc_dict_bub[key])[dists_index_sort]

                # Try merging key_i into its nearest neighbors key_j
                for key_j in key_js[1:]:
                    if key_i != key_j and key_i in clump_coords_dict_bub.keys() and key_j in clump_coords_dict_bub.keys():
                        coords_bl_i = np.unique(clump_coords_dict_bub[key_i][:, 1:], axis=0)
                        coords_bl_j = np.unique(clump_coords_dict_bub[key_j][:, 1:], axis=0)

                        bubble_peak_value_i = bubble_weight_data[
                            bubble_peaks[key_i][0], bubble_peaks[key_i][1], bubble_peaks[key_i][2]
                        ]
                        bubble_peak_value_j = bubble_weight_data[
                            bubble_peaks[key_j][0], bubble_peaks[key_j][1], bubble_peaks[key_j][2]
                        ]

                        # Merge heuristic: compare overlap from smaller region’s perspective
                        if len(clump_coords_dict_bub[key_i]) < len(clump_coords_dict_bub[key_j]):
                        # if len(coords_bl_i) < len(coords_bl_j):
                            match = np.all(coords_bl_i[:, np.newaxis] == coords_bl_j, axis=2)
                            a_indices = np.where(np.any(match, axis=1))[0]
                            common_rows = np.unique(coords_bl_i[a_indices], axis=0)

                            # Overlap ratio (smaller region overlap)
                            region_IOU = len(common_rows) / np.min([len(coords_bl_i),len(coords_bl_j)])

                            # Connectivity check in 2D (returns number of connected components between regions)
                            region_num = BFTools.Judge_Connectivity(coords_bl_i, coords_bl_j, dimension=2)

                            merge_logic_1 = region_IOU > MergeArea and region_num == 1
                            merge_logic_2 = bubble_peak_value_i > MergePeak * bubble_peak_value_j  # (currently not used)
                            merge_logic = merge_logic_1  # and not merge_logic_2

                            if merge_logic:
                                # Reassign label of key_i voxels to key_j+1 (labels are 1-based in array)
                                if key_i in clump_coords_dict_bub_T.keys():
                                    bubble_regions_data[
                                        clump_coords_dict_bub_T[key_i][:, 0],
                                        clump_coords_dict_bub_T[key_i][:, 1],
                                        clump_coords_dict_bub_T[key_i][:, 2]
                                    ] = key_j + 1
                                else:
                                    bubble_regions_data[
                                        clump_coords_dict_bub[key_i][:, 0],
                                        clump_coords_dict_bub[key_i][:, 1],
                                        clump_coords_dict_bub[key_i][:, 2]
                                    ] = key_j + 1

                                # Accumulate coords for key_j (merged set)
                                clump_coords_dict_bub_T[key_j] = np.r_[
                                    clump_coords_dict_bub[key_j],
                                    clump_coords_dict_bub[key_i]
                                ]
                                del clump_coords_dict_bub[key_i]

    # Final relabeling: compress labels to 1..N
    bubbles_coords_record = []
    clumps_list_bub = measure.regionprops(bubble_regions_data)

    for i in range(len(clumps_list_bub)):
        clump_coords_bub = clumps_list_bub[i].coords

        # In original code, optional filtering by Otsu/connected max-subregion is commented.
        clump_coords_bub_used = clump_coords_bub

        bubble_regions_data[
            clump_coords_bub_used[:, 0],
            clump_coords_bub_used[:, 1],
            clump_coords_bub_used[:, 2]
        ] = i + 1
        bubbles_coords_record.append(clump_coords_bub_used)

    return bubble_regions_data, bubbles_coords_record
    

def Bubble_Infor_Morphology(bubble_weight_data, bubbles_coords_record):
    """
    Compute basic morphology/statistics for each detected bubble (voxel coords list).

    For each bubble:
      - Project to 2D and compute regionprops (area, eccentricity, equivalent diameter)
      - Compute weight-mass-weighted center-of-mass in 3D (v,b,l)
      - Compute velocity range, volume, and a confidence metric (mean weight)
      - Store a contour polyline of the projected 2D boundary

    Returns
    -------
    bubble_infor : dict
        Keys: bubble_coms, radius_lb, areas_lb, eccentricities_lb, contours, ranges_v, volume, confidences
    bubble_regions_data : 3D uint32 array
        Bubble label map (1..N).
    bubble_regions : list
        regionprops list over bubble_regions_data.
    """
    bubble_coms = []
    bubble_peaks = []
    radius_lb = []
    areas_lb = []
    eccentricities_lb = []
    ranges_v = []
    volume = []
    confidences = []
    contours = []
    ellipses_infor = []  # (not filled in this function; kept from original code)

    bubble_infor = {}
    bubble_regions_data = np.zeros_like(bubble_weight_data, dtype='uint32')

    # Sort bubbles by minimum l (descending) to enforce a consistent ordering
    coords_l_min = [coords_i[:, 2].min() for coords_i in bubbles_coords_record]
    coords_l_min_argsort = np.argsort(coords_l_min)[::-1]

    k = 1
    for l_min_i in coords_l_min_argsort:
        coords_i = bubbles_coords_record[l_min_i]

        # 3D->2D projection and contour extraction
        box_data, coords_range, box_region, contour, boundary_coords = Cal_2D_Region_From_3D_Coords(
            coords_i, cal_contours=True
        )

        

        # Weight-mass (use bubble weights as "mass")
        od_mass = bubble_weight_data[coords_i[:, 0], coords_i[:, 1], coords_i[:, 2]]
        mass_array = np.c_[od_mass, od_mass, od_mass]

        # 3D center-of-mass in (v,b,l)
        bubble_com = np.around((mass_array * np.c_[coords_i[:, 0], coords_i[:, 1], coords_i[:, 2]]).sum(0) / od_mass.sum(), 3).tolist()
        bubble_coms.append(bubble_com)

        bubble_peaks.append(coords_i[np.where(od_mass==od_mass.max())][0])
        
        radius_lb.append(box_region.equivalent_diameter / 2)  # 2D equivalent radius
        areas_lb.append(box_region.area)
        eccentricities_lb.append(np.around(box_region.eccentricity, 2))
        contours.append(contour)

        # Velocity range of the bubble voxels
        range_v = [coords_i[:, 0].min(), coords_i[:, 0].max()]
        ranges_v.append(range_v)

        volume.append(len(coords_i))

        # Confidence: mean weight within bubble voxels
        confidences.append(np.around(np.mean(bubble_weight_data[coords_i[:, 0], coords_i[:, 1], coords_i[:, 2]]), 3))

        # Write label into output label cube
        bubble_regions_data[coords_i[:, 0], coords_i[:, 1], coords_i[:, 2]] = k
        k += 1

    bubble_regions = measure.regionprops(bubble_regions_data)

    bubble_infor['bubble_coms'] = bubble_coms
    bubble_infor['bubble_peaks'] = bubble_peaks
    bubble_infor['radius_lb'] = radius_lb
    bubble_infor['areas_lb'] = areas_lb
    bubble_infor['eccentricities_lb'] = eccentricities_lb
    bubble_infor['contours'] = contours
    bubble_infor['ranges_v'] = ranges_v
    bubble_infor['volume'] = volume
    bubble_infor['confidences'] = confidences

    return bubble_infor, bubble_regions_data, bubble_regions


def Bubble_Infor_Morphology_WCS(data_wcs, bubble_infor):
    """
    Convert bubble center and velocity ranges from pixel coordinates to WCS units.

    This function supports WCS objects with naxis==3 or naxis==4. The actual axis order
    used by `all_pix2world` depends on your WCS definition (here it’s used consistently
    with l,b,v ordering in calls).

    Parameters
    ----------
    data_wcs : astropy.wcs.WCS
        WCS of the cube.
    bubble_infor : dict
        Must contain 'bubble_coms' and 'ranges_v'.

    Returns
    -------
    bubble_infor : dict
        Adds:
          - 'bubble_coms_wcs' : array of WCS centers
          - 'ranges_v_wcs'    : array of WCS velocity ranges in km/s
    """
    bubble_vs_wcs = []
    bubble_coms = np.array(bubble_infor['bubble_coms'])
    ranges_v = np.array(bubble_infor['ranges_v'])

    bubble_infor['bubble_coms_wcs'] = []
    bubble_infor['ranges_v_wcs'] = []

    if len(bubble_coms) != 0:
        for index in range(len(bubble_coms)):
            if data_wcs.naxis == 3:
                # Convert ALL bubble COMs at once (l,b,v) from (pix_l, pix_b, pix_v)
                bubble_coms_wcs = data_wcs.all_pix2world(bubble_coms[:, 2], bubble_coms[:, 1], bubble_coms[:, 0], 0)
                bubble_coms_wcs = np.around(np.c_[bubble_coms_wcs[0], bubble_coms_wcs[1], bubble_coms_wcs[2]], 3)

                # Convert v-range endpoints for each bubble
                bubble_v0_wcs = data_wcs.all_pix2world(bubble_coms[index][2], bubble_coms[index][1], ranges_v[index][0], 0)
                bubble_v1_wcs = data_wcs.all_pix2world(bubble_coms[index][2], bubble_coms[index][1], ranges_v[index][1], 0)

            elif data_wcs.naxis == 4:
                bubble_coms_wcs = data_wcs.all_pix2world(bubble_coms[:, 2], bubble_coms[:, 1], bubble_coms[:, 0], 0, 0)
                bubble_coms_wcs = np.around(np.c_[bubble_coms_wcs[0], bubble_coms_wcs[1], bubble_coms_wcs[2]], 3)

                bubble_v0_wcs = data_wcs.all_pix2world(bubble_coms[index][2], bubble_coms[index][1], ranges_v[index][0], 0, 0)
                bubble_v1_wcs = data_wcs.all_pix2world(bubble_coms[index][2], bubble_coms[index][1], ranges_v[index][1], 0, 0)

            bubble_v_wcs = list(np.around([bubble_v0_wcs[2], bubble_v1_wcs[2]], 3))
            bubble_vs_wcs.append(bubble_v_wcs)

        # Convert velocity axis unit to km/s (assumes WCS third axis is velocity-like)
        bubble_coms_wcs[:, 2] = (bubble_coms_wcs[:, 2] * data_wcs.wcs.cunit[2]).to(u.km / u.s).value
        bubble_vs_wcs = (bubble_vs_wcs * data_wcs.wcs.cunit[2]).to(u.km / u.s).value

        bubble_infor['bubble_coms_wcs'] = np.around(bubble_coms_wcs, 3)
        bubble_infor['ranges_v_wcs'] = np.around(bubble_vs_wcs, 3)

    return bubble_infor


def Cal_Max_Sub_Region_Coords(coords, extend_len=1, dilation_r=None):
    """
    Extract the largest connected sub-region from a 3D coordinate set.

    This function:
      1) Builds a tight 3D bounding box around the input coordinates (with padding = extend_len),
      2) Labels connected components inside that box,
      3) Selects the largest component (by number of voxels),
      4) Optionally performs a binary dilation first (radius = dilation_r) and repeats step 2–3.

    Parameters
    ----------
    coords : (N, 3) ndarray
        Voxel coordinates in absolute/global index space. Each row is [x, y, z] (or [v, b, l] depending on convention).
    extend_len : int
        Padding added around the bounding box when creating the local cube.
        Helps preserve connectivity near the edges.
    dilation_r : int or None
        If provided, a morphological dilation is applied to the local mask before
        extracting the largest connected component again. Useful for bridging small gaps.

    Returns
    -------
    coords_range : (6,) ndarray
        [x_min, x_max, y_min, y_max, z_min, z_max] for the original coords (no padding applied here).
    box_region_max : skimage.measure._regionprops.RegionProperties
        RegionProperties object corresponding to the largest connected component
        (from the *non-dilated* local mask).
    max_sub_coords : (M, 3) ndarray
        Coordinates (in global index space) for the largest connected component from the non-dilated mask.
    max_sub_coords_dilation : (K, 3) ndarray or None
        Coordinates (in global index space) for the largest connected component after dilation,
        or None if dilation_r is None.
    """
    # Compute the tight bounding box of the input coordinates
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()

    # Record the original (global) bounding range
    coords_range = np.array([x_min, x_max, y_min, y_max, z_min, z_max])

    # Create a local cube (with padding) and rasterize the coordinates into a binary mask
    box_data = np.zeros([
        x_max - x_min + extend_len * 2 + 1,
        y_max - y_min + extend_len * 2 + 1,
        z_max - z_min + extend_len * 2 + 1
    ])
    box_data[coords[:, 0] - x_min + extend_len,
             coords[:, 1] - y_min + extend_len,
             coords[:, 2] - z_min + extend_len] = 1

    # Label connected components in the local cube
    box_label = measure.label(box_data, connectivity=2)
    box_regions = measure.regionprops(box_label)

    # Pick the largest connected component by voxel count
    region_sizes = []
    for region in box_regions:
        region_sizes.append(len(region.coords))
    box_region_max = box_regions[np.argmax(region_sizes)]

    # Convert the largest component’s local coords back to global coords
    max_sub_coords = box_region_max.coords + np.array([x_min - extend_len,
                                                       y_min - extend_len,
                                                       z_min - extend_len])

    # Optional: dilate to merge close components, then extract largest again
    max_sub_coords_dilation = None
    if dilation_r is not None:
        box_data_dilation = morphology.binary_dilation(box_data, morphology.ball(dilation_r))
        box_label = measure.label(box_data_dilation, connectivity=2)
        box_regions = measure.regionprops(box_label)

        region_sizes = []
        for region in box_regions:
            region_sizes.append(len(region.coords))
        box_region_max = box_regions[np.argmax(region_sizes)]

        max_sub_coords_dilation = box_region_max.coords + np.array([x_min - extend_len,
                                                                    y_min - extend_len,
                                                                    z_min - extend_len])

    return coords_range, box_region_max, max_sub_coords, max_sub_coords_dilation


def Clump_Item_By_Coords(real_data, clump_coords):
    """
    Extract a centered cubic sub-volume containing the given clump voxels.

    Strategy
    --------
    - Compute the bounding box of the clump coords.
    - Allocate a cubic array with side length = max extent + padding.
    - Place the clump voxels into the center of the cube.
    - Return the local cube and the global-to-local offset.

    Parameters
    ----------
    real_data : 3D ndarray
        Original data cube.
    clump_coords : (N,3) ndarray
        Clump voxel coordinates [v,b,l].

    Returns
    -------
    clump_item : 3D ndarray
        Centered local cube containing clump values.
    start_coords : list
        Offset for mapping: global_coord - start_coords = local_coord.
    """
    # Convert coords into separate arrays
    clump_coords = (clump_coords[:, 0], clump_coords[:, 1], clump_coords[:, 2])
    core_x, core_y, core_z = clump_coords

    # Bounding box
    x_min, x_max = core_x.min(), core_x.max()
    y_min, y_max = core_y.min(), core_y.max()
    z_min, z_max = core_z.min(), core_z.max()

    # Cube size (ensure minimum size)
    length = np.max([x_max - x_min, y_max - y_min, z_max - z_min]) + 5
    wish_len = 10
    if length < wish_len:
        length = wish_len + 5

    # Allocate cube
    clump_item = np.zeros([length, length, length])

    # Centering offsets
    start_x = np.int64((length - (x_max - x_min)) / 2)
    start_y = np.int64((length - (y_max - y_min)) / 2)
    start_z = np.int64((length - (z_max - z_min)) / 2)

    # Fill values into centered cube
    clump_item[
        core_x - x_min + start_x,
        core_y - y_min + start_y,
        core_z - z_min + start_z
    ] = real_data[core_x, core_y, core_z]

    # Offset to map back to global coords
    start_coords = [x_min - start_x, y_min - start_y, z_min - start_z]

    return clump_item, start_coords


def Cal_Ellipse_Coords_Values(bubbleObj, regions_data, bubble_clump_ids, dilation_r=1):
    """
    Evaluate whether a fitted ellipse is supported by clump regions.

    Strategy
    --------
    - Merge voxel coords of all clumps in `bubble_clump_ids`.
    - Extract a centered local cube using `Clump_Item_By_Coords`.
    - Project ellipse coords into the local (b,l) plane.
    - Build a binary mask of clump coverage (collapsed along v-axis).
    - Apply a small dilation to allow tolerance near boundaries.
    - Sample mask values along the ellipse to check coverage.

    Parameters
    ----------
    bubbleObj : object
        Must contain:
          - ellipse_coords : (N,2) array in (b,l)
          - clumpsObj.clump_coords_dict
    regions_data : 3D ndarray
        Clump label cube.
    bubble_clump_ids : array-like
        Clump ids (1-based).

    Returns
    -------
    clump_region_item : 3D ndarray
        Local cube containing clump labels.
    clump_region_values : ndarray
        Binary values (0/1) sampled along ellipse.
    clump_region_values_set : list
        Unique values along ellipse (used for validation).
    """
    if len(bubble_clump_ids) > 1:
        # Merge voxel coords from all clumps
        clump_coords = bubbleObj.clumpsObj.clump_coords_dict[bubble_clump_ids[0] - 1]
        for clump_id in bubble_clump_ids[1:]:
            clump_coords = np.r_[clump_coords,bubbleObj.clumpsObj.clump_coords_dict[clump_id - 1]]

        # Extract local cube centered on clump region
        clump_region_item, start_coords = Clump_Item_By_Coords(regions_data, clump_coords)

        # Map ellipse coords into local (b,l) coordinate system
        ellipse_coords_i = np.int64(np.around(np.c_[
            bubbleObj.ellipse_coords[:, 0] - start_coords[1],
            bubbleObj.ellipse_coords[:, 1] - start_coords[2]]))

        # Collapse along velocity axis → 2D mask
        clump_region_item_sum_mask = clump_region_item.sum(0)
        clump_region_item_sum_mask[clump_region_item_sum_mask > 0] = 1

        # Slight dilation to tolerate boundary mismatch
        clump_region_item_sum_mask_dilation = morphology.binary_dilation(clump_region_item_sum_mask, morphology.disk(dilation_r))
        
        # Sample binary mask values along ellipse
        H, W = clump_region_item_sum_mask_dilation.shape
        x, y = ellipse_coords_i.T
        inside = ((x>=0)&(x<H)&(y>=0)&(y<W))
        ellipse_coords_i = ellipse_coords_i[inside]
        if len(ellipse_coords_i)>0:
            clump_region_values = np.int32(clump_region_item_sum_mask_dilation[ellipse_coords_i[:, 0],ellipse_coords_i[:, 1]])
        else:
            clump_region_values = np.array([1])
        # Unique values (e.g. detect if 0 exists → ellipse crosses empty region)
        clump_region_values_set = list(set(clump_region_values))

    else:
        # Not enough clumps to evaluate
        clump_region_item, clump_region_values, clump_region_values_set = [], [], []
        
    return clump_region_item, clump_region_values, clump_region_values_set


def Get_Bubble_Clump_Ids(bubbleObj, bubble_region, clump_centers, regions_data, connected_ids_dict,
                         bubble_clump_ids=None, dilation_r=1):
    """
    Determine which clumps are associated with a bubble region.

    Strategy
    --------
    If `bubble_clump_ids` is not provided:
      1) Extract bubble voxels from `bubble_region.coords`.
      2) Keep largest connected component with optional dilation.
      3) Map voxels to `regions_data` to obtain clump labels.
      4) Validate clumps using ellipse sampling:
         - If ellipse crosses background (label=0), expand region.
      5) Increase dilation until enough clumps and valid ellipse coverage.

    Then:
      - Expand using `connected_ids_dict`.
      - Include connected clumps if their centers lie within (b,l) extent.

    Parameters
    ----------
    bubble_region : skimage RegionProperties
        Bubble region in 3D.
    clump_centers : (Nc,3) ndarray
        Clump centers [v,b,l].
    regions_data : 3D ndarray
        Clump label cube (label = id+1, 0=background).
    connected_ids_dict : dict
        Clump adjacency mapping.
    bubble_clump_ids : array-like or None
        Optional pre-defined clump ids (zero-based).
    dilation_r : int
        Initial dilation radius.

    Returns
    -------
    bubble_clump_ids : ndarray
        Zero-based clump ids inside bubble.
    bubble_clump_ids_con : ndarray
        Connected clumps not included in main set.
    """
    coords_i = np.array([])
    v_extend_logic = False

    if bubble_clump_ids is None:
        coords_i = bubble_region.coords

        # Largest connected component
        _, _, _, coords_i = Cal_Max_Sub_Region_Coords(
            coords_i, extend_len=10, dilation_r=dilation_r)

        # Boundary safety
        coords_i_valid = ((coords_i[:, 0] < regions_data.shape[0]) &
                          (coords_i[:, 1] < regions_data.shape[1]) &
                          (coords_i[:, 2] < regions_data.shape[2]) )
        coords_i = coords_i[coords_i_valid]

        # Initial clump ids
        bubble_clump_ids = list(set(
            regions_data[coords_i[:, 0], coords_i[:, 1], coords_i[:, 2]]))
        if 0 in bubble_clump_ids:
            bubble_clump_ids.remove(0)

        # Ellipse-based validation
        clump_region_item, clump_region_values, clump_region_values_set = \
            Cal_Ellipse_Coords_Values(bubbleObj, regions_data, bubble_clump_ids)

        # Expand region if insufficient or invalid
        while len(bubble_clump_ids) <= bubbleObj.ClumpNum or 0 in clump_region_values_set:
            dilation_r += 1
            _, _, _, coords_i = Cal_Max_Sub_Region_Coords(coords_i, extend_len=10, dilation_r=dilation_r)

            coords_i_valid = (
                (coords_i[:, 0] < regions_data.shape[0]) &
                (coords_i[:, 1] < regions_data.shape[1]) &
                (coords_i[:, 2] < regions_data.shape[2]))
            coords_i = coords_i[coords_i_valid]

            bubble_clump_ids = list(set(regions_data[coords_i[:, 0], coords_i[:, 1], coords_i[:, 2]]))
            if 0 in bubble_clump_ids:
                bubble_clump_ids.remove(0)

            clump_region_item, clump_region_values, clump_region_values_set = \
                Cal_Ellipse_Coords_Values(bubbleObj, regions_data, bubble_clump_ids)

            if dilation_r > 3:
                break

        bubble_clump_ids = np.array(bubble_clump_ids) - 1

    # Collect connected clumps
    bubble_clump_ids_con = []
    bubble_clump_ids_con_used = []

    for clump_id in bubble_clump_ids:
        bubble_clump_ids_con += connected_ids_dict[clump_id]

    bubble_clump_ids = bubble_clump_ids.tolist()
    bubble_clump_ids_con = list(set(bubble_clump_ids_con))

    # Spatial filtering in (b,l)
    for clump_id_con in bubble_clump_ids_con:
        if len(coords_i) > 0:
            v_extend_logic = (
                clump_centers[clump_id_con][1] > coords_i[:, 1].min() and
                clump_centers[clump_id_con][1] < coords_i[:, 1].max() and
                clump_centers[clump_id_con][2] > coords_i[:, 2].min() and
                clump_centers[clump_id_con][2] < coords_i[:, 2].max())

        if clump_id_con not in bubble_clump_ids and v_extend_logic:
            bubble_clump_ids.append(clump_id_con)
        elif clump_id_con not in bubble_clump_ids:
            bubble_clump_ids_con_used.append(clump_id_con)

    bubble_clump_ids = np.array(bubble_clump_ids)
    bubble_clump_ids_con = np.array(bubble_clump_ids_con_used)

    if len(bubble_clump_ids_con) == 0:
        bubble_clump_ids_con = bubble_clump_ids

    return bubble_clump_ids, bubble_clump_ids_con
    

def Get_Bubble_Clump_Ids_V2(bubble_region, clump_centers, regions_data, connected_ids_dict,
                            bubble_clump_ids=None, dilation_r=1, dilation_rv=5):
    """
    Determine which clumps are associated with a given bubble region.

    Strategy
    --------
    If `bubble_clump_ids` is not provided:
      1) Start from `bubble_region.coords`.
      2) Extract the largest connected sub-region with optional dilation.
      3) Query `regions_data` to get clump labels intersecting this region.
      4) If too few clumps are found, increase `dilation_r` and retry.
      5) Apply a second, larger dilation (`dilation_rv`) and keep only voxels
         whose projected (b,l) positions overlap the original set, in order to
         supplement clumps mainly along the velocity direction.

    Then:
      - Expand to include clumps connected through `connected_ids_dict`.
      - Return the main bubble clumps and the connected-but-not-included clumps.

    Parameters
    ----------
    bubble_region : skimage RegionProperties
        RegionProperties for a bubble candidate in 3D.
    clump_centers : (Nc, 3) ndarray
        Clump centers in pixel coordinates [v, b, l].
        (Kept for interface consistency; not used in the current logic.)
    regions_data : 3D ndarray
        Label cube where each voxel belongs to a clump (label = clump_id + 1),
        and 0 indicates background.
    connected_ids_dict : dict
        Mapping clump_id -> list of connected clump_ids.
    bubble_clump_ids : array-like or None
        If provided, skip voxel-to-label extraction and use these clump ids directly.
        Returned ids are zero-based.
    dilation_r : int
        Initial dilation radius used in `Cal_Max_Sub_Region_Coords`.
    dilation_rv : int
        Larger dilation radius used to extend the region along velocity while
        preserving the projected (b,l) footprint.

    Returns
    -------
    bubble_clump_ids : ndarray
        Zero-based clump ids considered part of the bubble.
    bubble_clump_ids_con : ndarray
        Zero-based clump ids connected to the bubble clumps but not included in
        `bubble_clump_ids`. If empty, falls back to `bubble_clump_ids`.
    """
    coords_i = np.array([])   # Representative bubble voxel coords in global space

    # Infer bubble clump ids from the bubble voxels if not provided
    if bubble_clump_ids is None:
        coords_i = bubble_region.coords

        # Keep the largest connected subset; optionally bridge small gaps
        _, _, _, coords_i = Cal_Max_Sub_Region_Coords(coords_i, extend_len=10, dilation_r=dilation_r)

        # Remove coords outside the valid cube bounds
        coords_i_valid = (coords_i[:, 0] < regions_data.shape[0]) & \
                         (coords_i[:, 1] < regions_data.shape[1]) & \
                         (coords_i[:, 2] < regions_data.shape[2])
        coords_i = coords_i[coords_i_valid]

        # Query clump labels overlapping the current voxel set
        bubble_clump_ids = list(set(regions_data[coords_i[:, 0], coords_i[:, 1], coords_i[:, 2]]))
        if 0 in bubble_clump_ids:
            bubble_clump_ids.remove(0)

        # Retry with larger dilation if too few clumps are found
        while len(bubble_clump_ids) < 4:
            dilation_r += 1
            _, _, _, coords_i = Cal_Max_Sub_Region_Coords(coords_i, extend_len=10, dilation_r=dilation_r)

            coords_i_valid = (coords_i[:, 0] < regions_data.shape[0]) & \
                             (coords_i[:, 1] < regions_data.shape[1]) & \
                             (coords_i[:, 2] < regions_data.shape[2])
            coords_i = coords_i[coords_i_valid]

            bubble_clump_ids = list(set(regions_data[coords_i[:, 0], coords_i[:, 1], coords_i[:, 2]]))
            if 0 in bubble_clump_ids:
                bubble_clump_ids.remove(0)
            if dilation_r > 4:
                break

        # Extend the region with a larger dilation and keep only voxels whose
        # projected (b,l) positions overlap the original set
        _, _, _, coords_i2 = Cal_Max_Sub_Region_Coords(coords_i, extend_len=10, dilation_r=dilation_rv)
        coords_i2_common_v = []
        for coords_i2_i in coords_i2:
            if (coords_i2_i[1:] == coords_i[:, 1:]).all(axis=1).any():
                coords_i2_common_v.append(coords_i2_i)
        coords_i2_common_v = np.array(coords_i2_common_v)

        # Query additional clumps from the velocity-extended voxel set
        coords_i_valid = (coords_i2_common_v[:, 0] < regions_data.shape[0]) & \
                         (coords_i2_common_v[:, 1] < regions_data.shape[1]) & \
                         (coords_i2_common_v[:, 2] < regions_data.shape[2])
        coords_i2_common_v = coords_i2_common_v[coords_i_valid]
        bubble_clump_ids_2 = list(set(
            regions_data[coords_i2_common_v[:, 0],coords_i2_common_v[:, 1],coords_i2_common_v[:, 2]]))
        if 0 in bubble_clump_ids_2:
            bubble_clump_ids_2.remove(0)

        bubble_clump_ids_2_used = [] 
        for clump_id in bubble_clump_ids: 
            for connected_id in connected_ids_dict[clump_id]: 
                if connected_id in bubble_clump_ids_2: 
                    bubble_clump_ids_2_used.append(connected_id)

        # Merge base and extended clump labels
        bubble_clump_ids = list(set(np.r_[bubble_clump_ids, bubble_clump_ids_2]))

        # Convert from label-space (1..N) to zero-based clump ids (0..N-1)
        bubble_clump_ids = np.int64(np.array(bubble_clump_ids) - 1)

    # Gather clumps connected to the bubble clumps
    bubble_clump_ids_con = []
    bubble_clump_ids_con_used = []

    for clump_id in bubble_clump_ids:
        bubble_clump_ids_con += connected_ids_dict[clump_id]

    # Keep connected clumps that are not already included in the main set
    bubble_clump_ids_con = list(set(bubble_clump_ids_con))
    for clump_id_con in bubble_clump_ids_con:
        if clump_id_con not in bubble_clump_ids:
            bubble_clump_ids_con_used.append(clump_id_con)

    bubble_clump_ids = np.array(bubble_clump_ids)
    bubble_clump_ids_con = np.array(bubble_clump_ids_con_used)

    # Fallback: if no extra connected clumps remain, reuse the main set
    if len(bubble_clump_ids_con) == 0:
        bubble_clump_ids_con = bubble_clump_ids

    return bubble_clump_ids, bubble_clump_ids_con


def Get_Bubble_Gas_Infor(bubbleObj, index, systemic_v_type):
    """
    Compute gas-related information for a specific bubble.

    This function aggregates clump voxels belonging to the bubble and (optionally)
    its connected clumps, then computes:
      - Gas center (pixel + WCS)
      - Systemic velocity (two possible definitions)
      - Gas emission line profiles integrated over (b,l) and over extracted sub-cubes
      - WCS ranges (l,b,v) bounding boxes for both sets (bubble-only and bubble+connected)

    Parameters
    ----------
    bubbleObj : BubbleInfor-like object
        Object holding clumpsObj, bubble_clump_ids, etc. Results are stored back into bubbleObj.
    index : int
        Bubble index (used to access bubble centers in WCS if needed).
    systemic_v_type : int
        1: systemic velocity from mean gas center in WCS (bubble+connected)
        2: systemic velocity from bubble inner clump center in WCS

    Notes
    -----
    - Assumes bubbleObj.clumpsObj provides:
        origin_data, regions_data, data_wcs, delta_v, centers, centers_wcs, clump_coords_dict
    - Uses FCFA.Filament_Coords(...) to extract a local cube around selected clump ids.
    """
    origin_data = bubbleObj.clumpsObj.origin_data
    regions_data = bubbleObj.clumpsObj.regions_data
    data_wcs = bubbleObj.clumpsObj.data_wcs
    delta_v = bubbleObj.clumpsObj.delta_v

    bubble_coms_wcs = bubbleObj.bubble_coms_wcs
    clump_coords_dict = bubbleObj.clumpsObj.clump_coords_dict

    bubble_clump_ids = bubbleObj.bubble_clump_ids
    bubble_clump_ids_con = bubbleObj.bubble_clump_ids_con

    # Gas centers in pixel space (v,b,l): three variants
    # 1) bubble-only clumps
    # 12) connected-only clumps
    # 2) bubble + connected clumps together
    bubble_gas_com_1 = np.around(bubbleObj.clumpsObj.centers[bubble_clump_ids].mean(axis=0), 3)
    bubble_gas_com_12 = np.around(bubbleObj.clumpsObj.centers[bubble_clump_ids_con].mean(axis=0), 3)
    bubble_gas_com_2 = np.around(bubbleObj.clumpsObj.centers[np.r_[bubble_clump_ids, bubble_clump_ids_con]].mean(axis=0), 3)

    # Gas centers in WCS space (l,b,v) or similar order depending on your upstream convention
    bubble_gas_com_wcs_1 = np.around(bubbleObj.clumpsObj.centers_wcs[bubble_clump_ids].mean(axis=0), 3)
    bubble_gas_com_wcs_12 = np.around(bubbleObj.clumpsObj.centers_wcs[bubble_clump_ids_con].mean(axis=0), 3)
    bubble_gas_com_wcs_2 = np.around(bubbleObj.clumpsObj.centers_wcs[np.r_[bubble_clump_ids, bubble_clump_ids_con]].mean(axis=0), 3)

    # Choose systemic velocity definition
    if systemic_v_type == 1:
        systemic_v = bubble_gas_com_wcs_2[2]
    elif systemic_v_type == 2:
        systemic_v = bubble_coms_wcs[index][2]

    # Build raw voxel coordinate sets for gas:
    # bubble-only (coords_1) and bubble+connected (coords_2)
    bubble_gas_coords_1 = clump_coords_dict[bubble_clump_ids[0]]
    bubble_gas_coords_2 = clump_coords_dict[bubble_clump_ids[0]]

    # Concatenate all bubble-only clump voxels
    for bubble_clump_id in np.r_[bubble_clump_ids[1:]]:
        bubble_gas_coords_1 = np.r_[bubble_gas_coords_1, clump_coords_dict[bubble_clump_id]]

    # Concatenate bubble + connected clump voxels
    for bubble_clump_id in np.r_[bubble_clump_ids[1:], bubble_clump_ids_con]:
        bubble_gas_coords_2 = np.r_[bubble_gas_coords_2, clump_coords_dict[bubble_clump_id]]

    # Compute WCS bounding boxes (min/max) for the two coordinate sets
    bubble_gas_ranges_lbv_mins = []
    bubble_gas_ranges_lbv_maxs = []

    for bubble_gas_coords in [bubble_gas_coords_1, bubble_gas_coords_2]:
        if data_wcs.naxis == 3:
            bubble_ranges_lbv_min = data_wcs.all_pix2world(
                bubble_gas_coords[:, 2].min(),
                bubble_gas_coords[:, 1].min(),
                bubble_gas_coords[:, 0].min(),
                0
            )
            bubble_ranges_lbv_max = data_wcs.all_pix2world(
                bubble_gas_coords[:, 2].max(),
                bubble_gas_coords[:, 1].max(),
                bubble_gas_coords[:, 0].max(),
                0
            )
        elif data_wcs.naxis == 4:
            bubble_ranges_lbv_min = data_wcs.all_pix2world(
                bubble_gas_coords[:, 2].min(),
                bubble_gas_coords[:, 1].min(),
                bubble_gas_coords[:, 0].min(),
                0, 0
            )
            bubble_ranges_lbv_max = data_wcs.all_pix2world(
                bubble_gas_coords[:, 2].max(),
                bubble_gas_coords[:, 1].max(),
                bubble_gas_coords[:, 0].max(),
                0, 0
            )

        # Convert velocity axis to km/s using WCS units
        bubble_ranges_lbv_min[2] = (bubble_ranges_lbv_min[2] * data_wcs.wcs.cunit[2]).to(u.km / u.s).value
        bubble_ranges_lbv_max[2] = (bubble_ranges_lbv_max[2] * data_wcs.wcs.cunit[2]).to(u.km / u.s).value

        bubble_gas_ranges_lbv_mins.append(np.around([bubble_ranges_lbv_min[0],
                                                    bubble_ranges_lbv_min[1],
                                                    bubble_ranges_lbv_min[2]], 3))
        bubble_gas_ranges_lbv_maxs.append(np.around([bubble_ranges_lbv_max[0],
                                                    bubble_ranges_lbv_max[1],
                                                    bubble_ranges_lbv_max[2]], 3))

    # Extract local cube for bubble-only clumps (via Filament_Coords helper)
    clump_ids = bubble_clump_ids
    bubble_gas_coords, bubble_gas_item, data_wcs_gas_item, regions_data_T, start_coords, clumps_item_mask_2D, lb_area = \
        FCFA.Filament_Coords(origin_data, regions_data, data_wcs, clump_coords_dict, clump_ids)

    # Store bubble-only extraction results in the object
    bubbleObj.bubble_gas_coords_1 = bubble_gas_coords
    bubbleObj.bubble_gas_item_1 = bubble_gas_item
    bubbleObj.data_wcs_gas_item_1 = data_wcs_gas_item
    bubbleObj.start_coords_gas_item_1 = start_coords

    # 1D spectrum (sum over b,l) from the extracted local cube
    bubbleObj.bubble_gas_item_1_line_v = bubble_gas_item.sum(2).sum(1) * delta_v

    # Also compute 1D spectrum directly from origin_data using unique (b,l) pixels
    # bubble_gas_coords_1_bl = np.unique(bubble_gas_coords_1[:, 1:], axis=0)
    # bubbleObj.bubble_gas_1_line_v = origin_data[:, bubble_gas_coords_1_bl[:, 0], bubble_gas_coords_1_bl[:, 1]].sum(1) * delta_v

    # Extract local cube for bubble + connected clumps
    clump_ids = np.r_[bubble_clump_ids, bubble_clump_ids_con]
    bubble_gas_coords, bubble_gas_item, data_wcs_gas_item, regions_data_T, start_coords, clumps_item_mask_2D, lb_area = \
        FCFA.Filament_Coords(origin_data, regions_data, data_wcs, clump_coords_dict, clump_ids)

    # Store bubble+connected extraction results
    bubbleObj.bubble_gas_item_2 = bubble_gas_item
    bubbleObj.data_wcs_gas_item_2 = data_wcs_gas_item
    bubbleObj.start_coords_gas_item_2 = start_coords

    # 1D spectrum from the extracted local cube
    bubbleObj.bubble_gas_item_2_line_v = bubble_gas_item.sum(2).sum(1) * delta_v

    # 1D spectrum directly from origin_data using unique (b,l) pixels for bubble+connected
    # bubble_gas_coords_2_bl = np.unique(bubble_gas_coords_2[:, 1:], axis=0)
    # bubbleObj.bubble_gas_2_line_v = origin_data[:, bubble_gas_coords_2_bl[:, 0], bubble_gas_coords_2_bl[:, 1]].sum(1) * delta_v

    # Store summary scalar quantities
    bubbleObj.bubble_gas_coords_2 = bubble_gas_coords
    bubbleObj.bubble_gas_com_1 = bubble_gas_com_1
    bubbleObj.bubble_gas_com_12 = bubble_gas_com_12
    bubbleObj.bubble_gas_com_2 = bubble_gas_com_2
    bubbleObj.bubble_gas_com_wcs_1 = bubble_gas_com_wcs_1
    bubbleObj.bubble_gas_com_wcs_12 = bubble_gas_com_wcs_12
    bubbleObj.bubble_gas_com_wcs_2 = bubble_gas_com_wcs_2
    bubbleObj.systemic_v = systemic_v

    # Store bounding ranges (min/max) for both cases
    bubbleObj.bubble_gas_ranges_lbv_min_1 = bubble_gas_ranges_lbv_mins[0]
    bubbleObj.bubble_gas_ranges_lbv_max_1 = bubble_gas_ranges_lbv_maxs[0]
    bubbleObj.bubble_gas_ranges_lbv_min_2 = bubble_gas_ranges_lbv_mins[1]
    bubbleObj.bubble_gas_ranges_lbv_max_2 = bubble_gas_ranges_lbv_maxs[1]


def Cal_Bub_Weights(bubbleObj, type='mean'):
    """
    Compute a scalar "bubble weight" per bubble by aggregating bubble_weight_data
    over the bubble voxel coordinates.

    Parameters
    ----------
    bubbleObj : object
        Must provide:
          - bubble_weight_data : 3D ndarray
          - bubbles_coords     : list of (N_i,3) coords for each bubble
          - bubble_coms_wcs    : used only for length (number of bubbles)
    type : {'mean', 'max', 'median'}
        Aggregation method.

    Side Effects
    ------------
    Sets bubbleObj.bub_weights : list of float
    """
    bub_weights = []

    if type == 'mean':
        for bub_id in range(len(bubbleObj.bubble_coms_wcs)):
            bub_weight = bubbleObj.bubble_weight_data[
                bubbleObj.bubble_regions[bub_id].coords[:, 0],
                bubbleObj.bubble_regions[bub_id].coords[:, 1],
                bubbleObj.bubble_regions[bub_id].coords[:, 2]
            ].mean()
            bub_weights.append(np.around(bub_weight, 3))

    elif type == 'max':
        for bub_id in range(len(bubbleObj.bubble_coms_wcs)):
            bub_weight = bubbleObj.bubble_weight_data[
                bubbleObj.bubble_regions[bub_id].coords[:, 0],
                bubbleObj.bubble_regions[bub_id].coords[:, 1],
                bubbleObj.bubble_regions[bub_id].coords[:, 2]
            ].max()
            bub_weights.append(np.around(bub_weight, 3))

    elif type == 'median':
        for bub_id in range(len(bubbleObj.bubble_coms_wcs)):
            bub_weight = np.median(bubbleObj.bubble_weight_data[
                bubbleObj.bubble_regions[bub_id].coords[:, 0],
                bubbleObj.bubble_regions[bub_id].coords[:, 1],
                bubbleObj.bubble_regions[bub_id].coords[:, 2]
            ])
            bub_weights.append(np.around(bub_weight, 3))

    bubbleObj.bub_weights = bub_weights


def Resort_Ellipse_Coords(bubbleObj,add_con=False,ellipse_start_type='max'):
    """
    Reorder ellipse contour coordinates so that the "starting point" is placed
    at a location with maximal or minimal nearby gas intensity, or is original coordinates.

    Idea:
      - Project ellipse coords into the extracted gas sub-cube coordinate system,
      - For each ellipse point, sum gas intensity in a small neighborhood,
      - Choose the ellipse point with the largest neighborhood sum as the new start,
      - Rotate the coordinate list accordingly.

    Side Effects
    ------------
    Updates bubbleObj.ellipse_coords in-place.
    """
    if add_con==False:
        bubble_gas_item = bubbleObj.bubble_gas_item_1
        start_coords_gas_item = bubbleObj.start_coords_gas_item_1
    else:
        bubble_gas_item = bubbleObj.bubble_gas_item_2
        start_coords_gas_item = bubbleObj.start_coords_gas_item_2

    # Convert global ellipse coords to local (b,l) coords inside gas sub-cube
    ellipse_coords_gas = np.int64(np.around(bubbleObj.ellipse_coords - start_coords_gas_item[1:]))

    ellipse_values = []
    for ellipse_coord_gas in ellipse_coords_gas:
        # Neighborhood sampling around an ellipse point (in 2D plane)
        neighbor_coords = BFTools.Generate_Neighbor_coords(
            np.int64(np.around(ellipse_coord_gas)),
            bubble_gas_item.sum(0).shape
        )
        if len(neighbor_coords) > 0:
            ellipse_value = bubble_gas_item.sum(0)[neighbor_coords[:, 0], neighbor_coords[:, 1]].sum()
        else:
            ellipse_value = 0
        ellipse_values.append(ellipse_value)

    # Rotate ellipse coordinate list so that max-value point becomes first
    if ellipse_start_type=='max':
        ellipse_coords_order = np.argsort(ellipse_values)
        ellipse_coords_updated = np.r_[
            bubbleObj.ellipse_coords[ellipse_coords_order[-1]:],
            bubbleObj.ellipse_coords[:ellipse_coords_order[-1] + 1]
        ]
    elif ellipse_start_type=='min':
        ellipse_coords_order = np.argsort(ellipse_values)
        ellipse_coords_updated = np.r_[
            bubbleObj.ellipse_coords[ellipse_coords_order[0]:],
            bubbleObj.ellipse_coords[:ellipse_coords_order[0] + 1]
        ]
    else:
        ellipse_coords_updated = ellipse_coords_gas
    bubbleObj.ellipse_coords = ellipse_coords_updated


def Get_Bubble_Inner_Item(bubble_regions_data, coords):
    """
    Extract a centered local cube containing the given bubble voxels.

    A cubic volume is allocated with side length:
        max(extent_x, extent_y, extent_z) + 5
    The bubble voxels are placed into the center of this cube.

    Parameters
    ----------
    bubble_regions_data : 3D ndarray
        Typically the labeled region cube.
    coords : (N, 3) ndarray
        Bubble voxel coordinates in global space.

    Returns
    -------
    bubble_inner_item : 3D ndarray
        Centered local cube containing the bubble labels/values.
    start_coords_inner : list [x0, y0, z0]
        Global-to-local offset: global_coord - start_coords_inner = local_coord
        (i.e., where the local cube starts in global coordinates).
    """
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()

    length = np.max([x_max - x_min, y_max - y_min, z_max - z_min]) + 5
    bubble_inner_item = np.zeros([length, length, length])

    # Center placement offsets
    start_x = np.int64((length - (x_max - x_min)) / 2)
    start_y = np.int64((length - (y_max - y_min)) / 2)
    start_z = np.int64((length - (z_max - z_min)) / 2)

    # Insert the bubble voxels into the centered cube (keeping original values)
    bubble_inner_item[
        coords[:, 0] - x_min + start_x,
        coords[:, 1] - y_min + start_y,
        coords[:, 2] - z_min + start_z
    ] = bubble_regions_data[coords[:, 0], coords[:, 1], coords[:, 2]]

    # Offset to map local coords back to global coords
    start_coords_inner = [x_min - start_x, y_min - start_y, z_min - start_z]
    return bubble_inner_item, start_coords_inner


def Cal_Contours_IOU(contour, ellipse_coords):
    """
    Calculate contour overlap ratio between the original contour and a fitted ellipse
    using Matplotlib Path point-in-polygon tests on a pixel grid.

    Parameters
    ----------
    contour : (N, 2) ndarray
        Original contour coordinates.
    ellipse_coords : (M, 2) ndarray
        Fitted ellipse contour coordinates.

    Returns
    -------
    contour_ellipse_IOU : float
        Intersection area / contour area (not symmetric IOU).
    visualization : (H, W, 3) uint8 ndarray
        RGB visualization:
          - Red   : pixels inside contour
          - Green : pixels inside ellipse
          - Blue  : pixels inside intersection
    contour_grid : (H, W) bool ndarray
        Pixel mask of contour interior.
    ellipse_grid : (H, W) bool ndarray
        Pixel mask of ellipse interior.
    """
    # Compute bounding box of both shapes to build a finite grid
    all_coords = np.vstack([contour, ellipse_coords])
    x_min, y_min = np.floor(all_coords.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_coords.max(axis=0)).astype(int)

    # Build grid points covering the bounding box
    x = np.linspace(x_min, x_max, x_max - x_min + 1)
    y = np.linspace(y_min, y_max, y_max - y_min + 1)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Convert both contours to polygon paths
    contour_path = Path(contour)
    ellipse_path = Path(ellipse_coords)

    # Determine whether each grid point lies inside each polygon
    contour_mask = contour_path.contains_points(grid_points)
    ellipse_mask = ellipse_path.contains_points(grid_points)

    # Compute overlap metrics
    contour_area = np.sum(contour_mask)
    intersection_area = np.sum(contour_mask & ellipse_mask)

    # NOTE: This is intersection / contour_area (not union-based IOU)
    contour_ellipse_IOU = intersection_area / contour_area if contour_area > 0 else 0
    contour_ellipse_IOU = np.around(contour_ellipse_IOU, 2)

    # Visualization
    visualization = np.zeros((len(y), len(x), 3), dtype=np.uint8)

    contour_grid = contour_mask.reshape(xx.shape)
    ellipse_grid = ellipse_mask.reshape(xx.shape)
    intersection_grid = (contour_mask & ellipse_mask).reshape(xx.shape)

    visualization[:, :, 0] = (contour_grid * 255).astype(np.uint8)       # Red channel
    visualization[:, :, 1] = (ellipse_grid * 255).astype(np.uint8)       # Green channel
    visualization[:, :, 2] = (intersection_grid * 255).astype(np.uint8)  # Blue channel

    return contour_ellipse_IOU, visualization, contour_grid, ellipse_grid



