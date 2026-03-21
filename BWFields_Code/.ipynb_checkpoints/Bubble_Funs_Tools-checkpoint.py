import time
import numpy as np
import astropy.io.fits as fits
import astropy.wcs as WCS
from astropy import units as u
from astropy.table import Table
from skimage import filters, measure, morphology
from scipy.spatial.distance import cdist


def Translate_Coords_LBV(coords_LBV, data_wcs, pix2world=False, world2pix=False):
    """
    Translate coordinates between pixel space and world (WCS) space.

    The function supports WCS with naxis = 2, 3, or 4.
    It calls:
      - all_pix2world(...) when pix2world=True
      - all_world2pix(...) when world2pix=True

    Notes / Assumptions
    -------------------
    - coords_LBV is assumed to be an array of points. For naxis>=3, it expects (x, y, v).
    - For naxis==2, it expects (x, y).
    - The code divides the 3rd coordinate by 1000 when pix2world=True,
      assuming the 3rd world axis is velocity in m/s and converting to km/s.
      (Be careful: if your WCS uses different units, this may be wrong.)
    - Potential pitfall: if data_wcs.naxis == 2 but pix2world=True, the function still later
      tries to access coords_LBV_Trans[:,2]. That will fail unless you guard it.

    Parameters
    ----------
    coords_LBV : array-like
        Input coordinates. Pixel mode expects (x,y) or (x,y,v). World mode expects (l,b,v) or (l,b).
    data_wcs : astropy.wcs.WCS
        WCS object describing the transformation.
    pix2world : bool
        If True, convert pixel -> world.
    world2pix : bool
        If True, convert world -> pixel.

    Returns
    -------
    coords_LBV_Trans : ndarray
        Transformed coordinates, rounded to 3 decimals.
    """
    coords_LBV = np.array(coords_LBV)

    # Choose transform based on flags and dimensionality of WCS
    if pix2world and data_wcs.naxis == 2:
        # 2D: (x, y) -> (lon, lat) or similar
        coords_T = data_wcs.all_pix2world(coords_LBV[:, 0], coords_LBV[:, 1], 0)

    elif pix2world and data_wcs.naxis == 3:
        # 3D: (x, y, v) -> (lon, lat, vel) or similar
        coords_T = data_wcs.all_pix2world(coords_LBV[:, 0], coords_LBV[:, 1], coords_LBV[:, 2], 0)

    elif pix2world and data_wcs.naxis == 4:
        # 4D: (x, y, v, stokes/time) -> world
        coords_T = data_wcs.all_pix2world(coords_LBV[:, 0], coords_LBV[:, 1], coords_LBV[:, 2], 0, 0)

    elif world2pix and data_wcs.naxis == 2:
        # 2D: (lon, lat) -> (x, y)
        coords_T = data_wcs.all_world2pix(coords_LBV[:, 0], coords_LBV[:, 1], 0)

    elif world2pix and data_wcs.naxis == 3:
        # 3D: (lon, lat, vel) -> (x, y, v)
        coords_T = data_wcs.all_world2pix(coords_LBV[:, 0], coords_LBV[:, 1], coords_LBV[:, 2], 0)

    elif world2pix and data_wcs.naxis == 4:
        # 4D: (lon, lat, vel, stokes/time) -> pixel
        coords_T = data_wcs.all_world2pix(coords_LBV[:, 0], coords_LBV[:, 1], coords_LBV[:, 2], 0, 0)

    else:
        print('Please check the naxis of data_wcs (2, 3 or 4) and the translation type (pix2world or world2pix).')

    # Pack output into an (N,2) or (N,3) matrix depending on returned tuple length
    if len(coords_T) > 2:
        coords_LBV_Trans = np.c_[coords_T[0], coords_T[1], coords_T[2]]
    elif len(coords_T) == 2:
        coords_LBV_Trans = np.c_[coords_T[0], coords_T[1]]
    else:
        print('Length of coords_T:', len(coords_T))

    # If converting to world coords and a velocity axis exists, convert m/s -> km/s
    # NOTE: This will error if coords_LBV_Trans has only 2 columns.
    if pix2world and coords_LBV_Trans.shape[1] >= 3:
        coords_LBV_Trans[:, 2] = coords_LBV_Trans[:, 2] / 1000

    coords_LBV_Trans = np.around(coords_LBV_Trans, 3)
    return coords_LBV_Trans


def Cal_Item_WCS_Range(data_item, data_wcs_item):
    """
    Compute the world-coordinate coverage (l,b,v) of a data cube and pixel scale.

    Steps
    -----
    1) Define pixel min/max for each axis based on data_item.shape.
    2) Convert corners in pixel space to world space using WCS.
    3) If a velocity axis exists, convert it to km/s and build a linear velocity axis array.
    4) Estimate pixel scale from WCS cdelt if celestial axes exist.

    Parameters
    ----------
    data_item : ndarray
        Data cube. Expected shape (nv, ny, nx) when velocity is included.
    data_wcs_item : astropy.wcs.WCS
        WCS for this cube.

    Returns
    -------
    lbv_item_start : list
        World coordinate at cube start corner (l,b[,v]) with v in km/s if present.
    lbv_item_end : list
        World coordinate at cube end corner (l,b[,v]) with v in km/s if present.
    velocity_axis : ndarray or None
        Velocity axis values (km/s) sampled linearly across nv channels; None for 2D WCS.
    pixel_scale : float
        Approx pixel scale in degrees/pixel (absolute cdelt[0]) if celestial WCS; else a fallback value.
    """
    data_item_shape = data_item.shape

    # Pixel-space bounds (x=lon axis index, y=lat axis index, v=velocity channel index)
    l_min, l_max = 0, data_item_shape[2] - 1
    b_min, b_max = 0, data_item_shape[1] - 1
    v_min, v_max = 0, data_item_shape[0] - 1

    # Convert pixel corners to world coords depending on WCS dimensionality
    if data_wcs_item.naxis == 4:
        lbv_start = data_wcs_item.all_pix2world(l_min, b_min, v_min, 0, 0)
        lbv_end = data_wcs_item.all_pix2world(l_max, b_max, v_max, 0, 0)
    elif data_wcs_item.naxis == 3:
        lbv_start = data_wcs_item.all_pix2world(l_min, b_min, v_min, 0)
        lbv_end = data_wcs_item.all_pix2world(l_max, b_max, v_max, 0)
    elif data_wcs_item.naxis == 2:
        lbv_start = data_wcs_item.all_pix2world(l_min, b_min, 0)
        lbv_end = data_wcs_item.all_pix2world(l_max, b_max, 0)

    # If WCS returns (l,b,v,...) then build start/end with velocity converted to km/s
    if len(lbv_start) > 2:
        # Convert third axis unit to km/s using header-defined unit (cunit[2])
        lbv_item_start = [
            np.around(lbv_start[0], 2),
            np.around(lbv_start[1], 2),
            np.around((lbv_start[2] * data_wcs_item.wcs.cunit[2]).to(u.km / u.s).value, 2)
        ]
        lbv_item_end = [
            np.around(lbv_end[0], 2),
            np.around(lbv_end[1], 2),
            np.around((lbv_end[2] * data_wcs_item.wcs.cunit[2]).to(u.km / u.s).value, 2)
        ]

        # Build a velocity axis with nv samples between start and end
        velocity_axis = np.linspace(lbv_item_start[2], lbv_item_end[2], data_item_shape[0])

    elif len(lbv_start) == 2:
        # Only l,b are present
        lbv_item_start = [np.around(lbv_start[0], 2), np.around(lbv_start[1], 2)]
        lbv_item_end = [np.around(lbv_end[0], 2), np.around(lbv_end[1], 2)]
        velocity_axis = None

    # Estimate pixel scale (degrees/pixel). If WCS lacks celestial axes, use a fallback.
    if data_wcs_item.has_celestial:
        pixel_scale = np.abs(data_wcs_item.wcs.cdelt[0])
    else:
        pixel_scale = 0.0083333333333333  # fallback (≈ 0.5 arcmin in degrees)

    return lbv_item_start, lbv_item_end, velocity_axis, pixel_scale


def Dists_Array(matrix_1, matrix_2):
    """
    Compute pairwise Euclidean distances between two point sets.

    Uses vectorized identity:
      ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y

    A small random noise term is added to avoid exactly identical distances
    (sometimes useful to break ties in sorting/assignment).

    Parameters
    ----------
    matrix_1 : ndarray
        Shape (n, d)
    matrix_2 : ndarray
        Shape (m, d)

    Returns
    -------
    dists : ndarray
        Distance matrix with shape (n, m)
    """
    matrix_1 = np.array(matrix_1)
    matrix_2 = np.array(matrix_2)

    # -2 * dot product term
    dist_1 = -2 * np.dot(matrix_1, matrix_2.T)

    # ||x||^2 term
    dist_2 = np.sum(np.square(matrix_1), axis=1, keepdims=True)

    # ||y||^2 term
    dist_3 = np.sum(np.square(matrix_2), axis=1)

    # Add tiny random component to avoid identical distances (tie-breaking)
    random_dist = np.random.random(dist_1.shape) / 1000000

    # Combine all pieces into squared distances
    dist_temp = dist_1 + dist_2 + dist_3 + random_dist

    # Square-root for Euclidean distances
    dists = np.sqrt(dist_temp)
    return dists


def Sort_By_Nearest_Neighbor_Dists(points):
    """
    Sort points by their nearest-neighbor distance (ascending).

    This is useful to rank points by local density:
    - points with smaller nearest-neighbor distance are in denser regions
    - points with larger nearest-neighbor distance are more isolated

    Parameters
    ----------
    points : ndarray
        Array of shape (N, d)

    Returns
    -------
    sorted_indices : ndarray
        Indices that sort points by min nearest-neighbor distance.
    sorted_points : ndarray
        Points sorted by that criterion.
    sorted_min_distances : ndarray
        The corresponding nearest-neighbor distances (sorted).
    """
    dist_matrix = cdist(points, points, metric='euclidean')
    np.fill_diagonal(dist_matrix, np.inf)  # ignore self-distance
    min_distances = np.min(dist_matrix, axis=1)
    sorted_indices = np.argsort(min_distances)
    return sorted_indices, points[sorted_indices], min_distances[sorted_indices]


def Generate_Ellipse_Points(a_res, b_res, alpha_res, center=None):
    """
    Generate perimeter points of a rotated ellipse.

    Parameters
    ----------
    a_res : float
        Semi-major axis length (in pixel units).
    b_res : float
        Semi-minor axis length (in pixel units).
    alpha_res : float
        Rotation angle in radians.
    center : array-like
        Ellipse center [x0, y0].

    Returns
    -------
    coords_fit : ndarray
        (N,2) array of ellipse perimeter points.
    ellipse_infor : list
        [center_x, center_y, angle_deg, a_res, b_res]
    """
    # Approximate ellipse perimeter (Ramanujan-like approximation)
    ellipse_perimeter = 2 * np.pi * np.sqrt((a_res**2 + b_res**2) / 2)

    # Choose number of samples proportional to perimeter length
    num_points = int(ellipse_perimeter)
    theta_res = np.linspace(0.0, 2 * np.pi, num_points)

    # Parametric ellipse with rotation
    x_res = a_res * np.cos(theta_res) * np.cos(alpha_res) \
            - b_res * np.sin(theta_res) * np.sin(alpha_res)
    y_res = b_res * np.sin(theta_res) * np.cos(alpha_res) \
            + a_res * np.cos(theta_res) * np.sin(alpha_res)

    # Translate to center
    coords_fit = np.c_[x_res + center[0], y_res + center[1]]

    ellipse_angle = np.around(np.rad2deg(alpha_res), 2)
    ellipse_infor = [center[0], center[1], ellipse_angle, a_res, b_res]
    return coords_fit, ellipse_infor


def Generate_Circle_Points(center_x, center_y, radius):
    """
    Generate perimeter points of a circle.

    Parameters
    ----------
    center_x, center_y : float
        Circle center.
    radius : float
        Circle radius (pixel units).

    Returns
    -------
    points : ndarray
        (N,2) circle perimeter points.
    circumference : float
        Circumference = 2*pi*radius.
    """
    circumference = 2 * np.pi * radius

    # Use approximately one point per pixel along circumference
    num_points = int(round(circumference))
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=True)

    x = center_x + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)
    points = np.column_stack((x, y))
    return points, circumference


def Generate_Neighbor_coords(center, data_shape, range_x=3, range_y=3, step=1):
    """
    Generate neighbor coordinates around a center point within array bounds.

    Notes
    -----
    - This function uses `H, W = data_shape` and then checks `0 <= nx < H` and `0 <= ny < W`.
      That means it treats the first coordinate as "row" (y-like) and the second as "col" (x-like).
      However the docstring says center: (x, y). Be careful with your coordinate convention.

    Parameters
    ----------
    center : tuple
        Center coordinate (x, y) or (row, col) depending on your convention.
    data_shape : tuple
        (H, W) array shape for boundary checks.
    range_x, range_y : int
        Neighborhood window size in each direction (odd numbers recommended).
    step : int
        Step size for offsets (in pixels).

    Returns
    -------
    neighbor_coords : ndarray
        Array of neighbor coordinates inside bounds.
    """
    H, W = data_shape

    half_x = (range_x - 1) // 2
    half_y = (range_y - 1) // 2

    x_offsets = [dx * step for dx in range(-half_x, half_x + 1)]
    y_offsets = [dy * step for dy in range(-half_y, half_y + 1)]

    neighbor_coords = []
    for dx in x_offsets:
        for dy in y_offsets:
            nx = center[0] + dx
            ny = center[1] + dy

            # Boundary check: keep only valid indices
            if 0 <= nx < H and 0 <= ny < W:
                neighbor_coords.append((nx, ny))

    return np.array(neighbor_coords)


def Judge_Connectivity(region_coords_i, region_coords_j, dimension=2):
    """
    Judge connectivity between two regions by putting them into a common bounding box,
    labeling connected components, and counting how many components exist.

    Interpretation:
    - If region_coords_i and region_coords_j are connected (touching under the chosen connectivity),
      then labeling will produce 1 connected component.
    - If they are separate, labeling produces 2 components.

    Parameters
    ----------
    region_coords_i : ndarray
        Coordinates for region i, shape (Ni, 2) or (Ni, 3).
    region_coords_j : ndarray
        Coordinates for region j, shape (Nj, 2) or (Nj, 3).
    dimension : int
        2 for 2D connectivity, 3 for 3D connectivity.

    Returns
    -------
    region_num : int
        Number of connected components in the combined box (1 means connected).
    """
    if dimension == 2:
        # Build a minimal bounding box covering both regions
        x_min = np.r_[region_coords_i[:, 0], region_coords_j[:, 0]].min()
        x_max = np.r_[region_coords_i[:, 0], region_coords_j[:, 0]].max()
        y_min = np.r_[region_coords_i[:, 1], region_coords_j[:, 1]].min()
        y_max = np.r_[region_coords_i[:, 1], region_coords_j[:, 1]].max()

        box_data = np.zeros([x_max - x_min + 2, y_max - y_min + 2])
        box_data[region_coords_i[:, 0] - x_min, region_coords_i[:, 1] - y_min] = 1
        box_data[region_coords_j[:, 0] - x_min, region_coords_j[:, 1] - y_min] = 1

    elif dimension == 3:
        # 3D bounding box
        x_min = np.r_[region_coords_i[:, 0], region_coords_j[:, 0]].min()
        x_max = np.r_[region_coords_i[:, 0], region_coords_j[:, 0]].max()
        y_min = np.r_[region_coords_i[:, 1], region_coords_j[:, 1]].min()
        y_max = np.r_[region_coords_i[:, 1], region_coords_j[:, 1]].max()
        z_min = np.r_[region_coords_i[:, 2], region_coords_j[:, 2]].min()
        z_max = np.r_[region_coords_i[:, 2], region_coords_j[:, 2]].max()

        box_data = np.zeros([x_max - x_min + 2, y_max - y_min + 2, z_max - z_min + 2])
        box_data[region_coords_i[:, 0] - x_min, region_coords_i[:, 1] - y_min, region_coords_i[:, 2] - z_min] = 1
        box_data[region_coords_j[:, 0] - x_min, region_coords_j[:, 1] - y_min, region_coords_j[:, 2] - z_min] = 1

    # Connected-component labeling; connectivity=dimension means 4/8-connectivity (2D) or 6/18/26-ish (3D)
    box_label = measure.label(box_data, connectivity=dimension)
    box_region = measure.regionprops(box_label)

    # Number of labeled components
    region_num = len(box_region)
    return region_num


def Get_Item_Data_By_Coords(origin_data, coords):
    """
    Extract a local 3D sub-cube (with padding) containing the given voxel coordinates.

    The function creates a new cube 'item_data' that covers the coordinate bounding box
    (plus padding), and fills those voxels with corresponding values from origin_data.

    Parameters
    ----------
    origin_data : ndarray
        Original 3D data cube.
    coords : ndarray
        Voxel coordinates array of shape (N,3) with columns [x, y, z].

    Returns
    -------
    item_data : ndarray
        Local cube containing the selected voxels (others are 0).
    """
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()

    # v_delta is computed but not used; kept from earlier versions
    v_delta = x_max - x_min + 1

    coords_range = np.array([x_min, x_max, y_min, y_max, z_min, z_max])

    # Allocate local cube with padding of 2 voxels each side (+5 total in each axis)
    item_data = np.zeros([x_max - x_min + 5, y_max - y_min + 5, z_max - z_min + 5])

    # Place values into the padded local cube
    item_data[coords[:, 0] - x_min + 2,
              coords[:, 1] - y_min + 2,
              coords[:, 2] - z_min + 2] = origin_data[coords[:, 0], coords[:, 1], coords[:, 2]]

    return item_data


def Normalize_For_RGB(img, p_lo=1, p_hi=99.5, stretch="asinh"):
    """
    Normalize an image to [0,1] for RGB visualization using percentile clipping and stretching.

    Steps:
    1) Replace non-finite values with 0.
    2) Compute lower/upper percentiles using only positive pixels.
    3) Clip and scale into [0,1].
    4) Apply an optional stretch function (asinh/sqrt/log) to enhance contrast.

    Parameters
    ----------
    img : ndarray
        Input image.
    p_lo, p_hi : float
        Percentiles used for robust min/max scaling (computed on x[x>0]).
    stretch : str
        One of {"asinh", "sqrt", "log"}.

    Returns
    -------
    x : ndarray
        Normalized image in [0,1] with chosen stretch applied.
    """
    x = np.array(img, dtype=float)
    x[~np.isfinite(x)] = 0.0

    # Percentiles computed on positive pixels only (common in astronomical images)
    lo, hi = np.percentile(x[x > 0], [p_lo, p_hi])

    # If dynamic range is invalid, return zeros
    if hi <= lo:
        return np.zeros_like(x)

    # Normalize into [0,1]
    x = np.clip((x - lo) / (hi - lo), 0, 1)

    # Apply contrast stretch
    if stretch == "asinh":
        # Normalize by median before asinh to avoid overly dark images
        median = np.nanmedian(x[x > 0]) if np.any(x > 0) else 1.0
        x = x / median
        x = np.arcsinh(x)

    elif stretch == "sqrt":
        x = np.sqrt(x)

    elif stretch == "log":
        # log1p scaling with a fixed factor to enhance faint structures
        x = np.log1p(100 * x) / np.log1p(100)

    return x


def Save_Fits(data,data_header,file_name):
    hdu_sub = fits.PrimaryHDU()
    hdu_sub.data = data
    hdu_sub.header = data_header
    hdu_sub.writeto(file_name, overwrite=True)



