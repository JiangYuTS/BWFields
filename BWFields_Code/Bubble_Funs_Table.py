import time
import numpy as np
import astropy.io.fits as fits
import astropy.wcs as WCS
from astropy import units as u
from astropy.table import Table
from skimage import filters, measure, morphology


def Bubble_Infor_WCS(filename, bubble_infor):
    """
    Convert bubble information from pixel coordinates to WCS (world) coordinates.

    This function reads the FITS header, builds a WCS object, and converts:
      - bubble centers (pixel indices) -> (GLon, GLat, V) in world coordinates
      - velocity ranges (pixel indices) -> V range in world coordinates

    Notes
    -----
    - Assumes bubble_coms is an array-like of shape (N, 3) in (v, y, x) order.
    - WCS conversion uses: wcs.all_pix2world(x, y, z, origin)
      so the input order is (xpix, ypix, vpix).
    - The code divides the 3rd world coordinate by 1000, likely converting m/s -> km/s,
      depending on the FITS cube convention.

    Parameters
    ----------
    filename : str
        Path to the FITS cube (contains WCS information in header).
    bubble_infor : dict
        Dictionary that must include:
          - 'bubble_coms': list/array of bubble centers in pixel coords (v,y,x)
          - 'ranges_v'  : list/array of velocity ranges in pixel coords (v0,v1)

    Returns
    -------
    bubble_infor : dict
        Updated dictionary with:
          - 'bubble_coms_wcs': Nx3 array-like [GLon, GLat, VLSR(km/s)] rounded to 3 decimals
          - 'ranges_v_wcs'   : list of [Vlow, Vup] in km/s for each bubble
    """
    MWISP_bubble_vs_wcs = []

    # Read FITS header and build WCS transformation
    data_header = fits.getheader(filename)
    wcs = WCS.WCS(data_header)

    # Extract bubble centers and velocity ranges
    bubble_coms = np.array(bubble_infor['bubble_coms'])
    ranges_v = np.array(bubble_infor['ranges_v'])

    # Prepare output fields
    bubble_infor['bubble_coms_wcs'] = []
    bubble_infor['ranges_v_wcs'] = []

    if len(bubble_coms) != 0:
        # Convert all bubble centers from pixel -> world
        # Input pixel ordering for WCS: (x, y, v) = (com[:,2], com[:,1], com[:,0])
        MWISP_bubble_coms_wcs = wcs.all_pix2world(
            bubble_coms[:, 2], bubble_coms[:, 1], bubble_coms[:, 0], 0
        )

        # Stack into Nx3 and convert velocity unit scale (e.g., m/s -> km/s) by /1000
        MWISP_bubble_coms_wcs = np.around(
            np.c_[MWISP_bubble_coms_wcs[0], MWISP_bubble_coms_wcs[1], MWISP_bubble_coms_wcs[2] / 1000],
            3
        )

        # Convert velocity range for each bubble (keeping x,y fixed at bubble center)
        for index in range(len(bubble_coms)):
            # Lower bound velocity in pixel index -> world
            MWISP_bubble_v0_wcs = wcs.all_pix2world(
                bubble_coms[index][2], bubble_coms[index][1], ranges_v[index][0], 0
            )
            # Upper bound velocity in pixel index -> world
            MWISP_bubble_v1_wcs = wcs.all_pix2world(
                bubble_coms[index][2], bubble_coms[index][1], ranges_v[index][1], 0
            )

            # Store as [Vlow, Vup] in km/s (divide by 1000)
            MWISP_bubble_v_wcs = list(np.around(
                [MWISP_bubble_v0_wcs[2] / 1000, MWISP_bubble_v1_wcs[2] / 1000],
                3
            ))
            MWISP_bubble_vs_wcs.append(MWISP_bubble_v_wcs)

        # Save conversions back to dict
        bubble_infor['bubble_coms_wcs'] = MWISP_bubble_coms_wcs
        bubble_infor['ranges_v_wcs'] = MWISP_bubble_vs_wcs

    return bubble_infor


def Table_Interface_Pix(bubble_infor):
    """
    Build an Astropy Table of bubble parameters in pixel coordinates.

    This is mainly used for catalog output in pixel space:
      - center location (CenL, CenB, CenV) in pixel indices
      - velocity bounds (VLow, VUp) in pixel indices
      - geometric measurements (radius, area, eccentricity) in pixel units
      - volume and confidence (unitless or predefined)

    Parameters
    ----------
    bubble_infor : dict
        Must contain:
          - 'bubble_coms'        : list/array of centers (v,y,x)
          - 'ranges_v'           : list/array of velocity ranges in pixel
          - 'radius_lb'          : radius estimates in pixel
          - 'areas_lb'           : areas in pixel^2 (or pixel count)
          - 'eccentricities_lb'  : eccentricity values
          - 'volume'             : volume estimates (definition depends on pipeline)
          - 'confidences'        : confidence score per bubble

    Returns
    -------
    Bubble_Table_Pix : astropy.table.Table
        Table with standardized columns and formatting.
    """
    # Extract center coordinates in pixel units
    CenL = list(np.array(bubble_infor['bubble_coms'])[:, 2])  # x index
    CenB = list(np.array(bubble_infor['bubble_coms'])[:, 1])  # y index
    CenV = list(np.array(bubble_infor['bubble_coms'])[:, 0])  # v index

    # Extract velocity bounds in pixel units
    VLow = list(np.array(bubble_infor['ranges_v'])[:, 0])
    VUp = list(np.array(bubble_infor['ranges_v'])[:, 1])

    # Extract morphology/statistics
    RadiusLB = list(bubble_infor['radius_lb'])
    AreaLB = list(bubble_infor['areas_lb'])
    Eccentricity = list(bubble_infor['eccentricities_lb'])
    Volume = bubble_infor['volume']
    Confidence = bubble_infor['confidences']

    # Auto-generate 1-based IDs for catalog
    index_id = list(np.arange(1, len(CenL) + 1, 1))

    # Stack into a 2D array, then transpose into row-wise table format
    d_outcat = np.hstack([[index_id, CenL, CenB, CenV, VLow, VUp,
                           RadiusLB, AreaLB, Eccentricity, Volume, Confidence]]).T

    columns = ['ID', 'CenL', 'CenB', 'CenV', 'VLow', 'VUp',
               'RadiusLB', 'AreaLB', 'Eccentricity', 'Volume', 'Confidence']

    # Units: pixel-based for positions/velocities/radius/area, others unitless
    units = [None, 'pix', 'pix', 'pix', 'pix', 'pix', 'pix', 'pix', None, None, None]

    # Data types for compact storage
    dtype = ['int', 'float32', 'float32', 'float32', 'int', 'int',
             'int', 'int', 'float32', 'int', 'float32']

    Bubble_Table_Pix = Table(d_outcat, names=columns, dtype=dtype, units=units)

    # Set numeric formatting for float columns
    for i in range(len(dtype)):
        if dtype[i] == 'float32':
            Bubble_Table_Pix[columns[i]].info.format = '.3f'

    return Bubble_Table_Pix


def Table_Interface_WCS(bubbleObj, bub_ids=None):
    """
    Build an Astropy Table of bubble parameters in WCS/world coordinates.

    This outputs a catalog-like table including:
      - Bubble center in Galactic coordinates and VLSR
      - Skeleton ellipse center (SE) in Galactic coordinates and systemic velocity
      - Velocity bounds (VLow, VUp) in km/s
      - Ellipse geometry (angle, width, RA/RB) in arcmin
      - Expansion velocity estimate (ExpV)
      - Confidence score (Conf)

    Parameters
    ----------
    bubbleObj : object
        Object that must provide:
          - bubble_used_ids
          - bubble_coms_wcs
          - skeleton_ellipse_coms_wcs
          - thicknesss
          - pix_scale_arcmin
          - ranges_v_wcs
          - skeleton_ellipse_angle_abs  (angle + ellipse radii)
          - exp_maxs
          - bub_weights
    bub_ids : array-like or None
        Optional explicit ID list. If None, uses bubble_used_ids + 1.

    Returns
    -------
    Bubble_Table_WCS : astropy.table.Table
        Table with standardized columns, units, and formatting.
    """
    bubble_used_ids = bub_ids
    if bub_ids is None:
        bubble_used_ids = bubbleObj.bubble_used_ids

    # Bubble center in WCS (galactic lon/lat + velocity)
    CenL = list(np.array(bubbleObj.bubble_coms_wcs)[:, 0][bubble_used_ids])
    CenB = list(np.array(bubbleObj.bubble_coms_wcs)[:, 1][bubble_used_ids])
    CenV = list(np.array(bubbleObj.bubble_coms_wcs)[:, 2][bubble_used_ids])

    # Skeleton ellipse center in WCS for the same subset
    skeleton_ellipse_coms_wcs = np.array([bubbleObj.skeleton_ellipse_coms_wcs[i].tolist() for i in bubble_used_ids])
    CenSEL = skeleton_ellipse_coms_wcs[:, 0]
    CenSEB = skeleton_ellipse_coms_wcs[:, 1]
    CenSEV = skeleton_ellipse_coms_wcs[:, 2]  # systemic velocity

    # Shell thickness (pixel -> arcmin)
    Width = np.array([bubbleObj.thicknesss[i] for i in bubble_used_ids]) * bubbleObj.pix_scale_arcmin

    # Velocity bounds in km/s
    VLow = bubbleObj.ranges_v_wcs[:, 0][bubble_used_ids]
    VUp = bubbleObj.ranges_v_wcs[:, 1][bubble_used_ids]

    # Ellipse parameters: [angle, radiusA, radiusB] (pixel -> arcmin for radii)
    skeleton_ellipse_angle_abs = np.array([bubbleObj.skeleton_ellipse_angle_abs[i] for i in bubble_used_ids])
    Angle = skeleton_ellipse_angle_abs[:, 0]
    Radius_A = skeleton_ellipse_angle_abs[:, 1] * bubbleObj.pix_scale_arcmin
    Radius_B = skeleton_ellipse_angle_abs[:, 2] * bubbleObj.pix_scale_arcmin

    # Expansion velocity estimate and confidence
    Exp_maxs = np.array([bubbleObj.exp_maxs[i] for i in bubble_used_ids])
    Sexps = np.array([bubbleObj.exp_significances[i] for i in bubble_used_ids])
    Ssyms = np.array([bubbleObj.sym_scores[i] for i in bubble_used_ids])
    
    Confidence = np.array(bubbleObj.bub_weights)[bubble_used_ids]

    MWISP_ids = []
    for bubble_used_id in bubble_used_ids:
        MWISP_ids.append("MWISP{:05}".format(bubble_used_id+1))

    # Stack outputs and transpose into row format
    d_outcat = np.hstack([[MWISP_ids, CenL, CenB, CenV,
                           CenSEL, CenSEB, CenSEV,
                           VLow, VUp,
                           Angle, Radius_A, Radius_B, Width, 
                           Exp_maxs, Ssyms, Sexps, Confidence]]).T

    columns = ['MBID', 'GLon', 'GLat', 'VLSR', 'GLonSE', 'GLatSE', 'SysV',
               'VLow', 'VUp', 'Angle', 'RA', 'RB', 'Width', 'ExpV', 'Ssym', 'Sexp', 'Conf']

    units = [None, 'deg', 'deg', 'km/s', 'deg', 'deg', 'km/s',
             'km/s', 'km/s', None, 'arcmin', 'arcmin', 'arcmin', 'km/s', None, None, None]

    dtype = ['str', 'float32', 'float32', 'float32',
             'float32', 'float32', 'float32',
             'float32', 'float32',
             'float16', 'float16', 'float16', 'float16',
             'float32', 'float16','float16', 'float16']

    Bubble_Table_WCS = Table(d_outcat, names=columns, dtype=dtype, units=units)

    # Set formatting: more decimals for float32, fewer for float16
    for i in range(len(dtype)):
        if dtype[i] == 'float32':
            Bubble_Table_WCS[columns[i]].info.format = '.3f'
        if dtype[i] == 'float16':
            Bubble_Table_WCS[columns[i]].info.format = '.2f'

    return Bubble_Table_WCS




