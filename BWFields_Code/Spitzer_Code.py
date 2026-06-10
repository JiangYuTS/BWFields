import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.wcs as WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import WCSAxes
from astropy.visualization import make_lupton_rgb
from reproject import reproject_interp
from astropy.nddata import Cutout2D
import warnings
import os

warnings.filterwarnings('ignore')

from . import Bubble_Funs_Tools as BFTools   


def Cal_Data_Range_LB(data_wcs, data_cube):
    """
    Calculate the Galactic longitude/latitude coverage of a FITS cube/map.

    The function:
    1) Takes the four spatial corners in pixel space.
    2) Converts them to world coordinates using WCS.
    3) Converts to Galactic coordinates and returns min/max ranges.

    Notes
    -----
    - data_cube is assumed to have shape (nv, ny, nx), i.e., spectral axis first.
    - Only spatial corners are used; velocity is fixed at 0 in pixel_to_world.
    - WCS object may have naxis=2, 3, or 4 depending on cube definition.

    Parameters
    ----------
    data_wcs : astropy.wcs.WCS
        WCS object describing the data.
    data_cube : ndarray
        Data cube array with shape (nv, ny, nx).

    Returns
    -------
    data_ranges_lb : list
        [[l_max, l_min], [b_min, b_max]] in degrees.
        Maintained in this order to match later selection logic.
    """
    if data_cube.ndim == 3:
        nv, ny, nx = data_cube.shape
    elif data_cube.ndim == 2:
        ny, nx = data_cube.shape
    else:
        print('Check data_cube.ndim.')

    # Pixel coordinates for the 4 spatial corners
    corners = np.array([
        [0, 0],
        [nx - 1, 0],
        [0, ny - 1],
        [nx - 1, ny - 1]
    ])

    # Convert pixel corners to world coordinates
    if data_wcs.naxis == 2:
        sky_coords = data_wcs.pixel_to_world(corners[:, 0], corners[:, 1])
        galactic_coords = sky_coords.galactic
    elif data_wcs.naxis == 3:
        sky_coords = data_wcs.pixel_to_world(corners[:, 0], corners[:, 1], 0)
        galactic_coords = sky_coords[0].galactic
    elif data_wcs.naxis == 4:
        sky_coords = data_wcs.pixel_to_world(corners[:, 0], corners[:, 1], 0, 0)
        galactic_coords = sky_coords[0].galactic
    else:
        print('Check data_wcs.naxis.')

    # Extract longitude/latitude bounds in degrees
    l_min = galactic_coords.l.deg.min()
    l_max = galactic_coords.l.deg.max()
    b_min = galactic_coords.b.deg.min()
    b_max = galactic_coords.b.deg.max()

    # Return format consistent with later selection tests
    data_ranges_lb = [[l_max, l_min], [b_min, b_max]]
    return data_ranges_lb


def Read_Spitzer_Files_All_Red(file_names_folder,end_str='p005_024.fits',sort_logic=False):
    """
    Scan a folder and collect WCS coverage for all red FITS files.

    Parameters
    ----------
    file_names_folder : str
        Folder path that contains *.fits files.

    Returns
    -------
    data_ranges_lb_record : list
        List of [[l_max,l_min],[b_min,b_max]] for each file.
    file_names : list
        Full paths to the FITS files in the same order.
    """
    data_ranges_lb_record = []

    # Select red positive and negative FITS files
    file_names_p = [file_names_folder + f for f in os.listdir(file_names_folder) if f.endswith("{}".format(end_str))]
    file_names_n = [file_names_folder + f for f in os.listdir(file_names_folder) if f.endswith("n005_024.fits")]

    # Sort files
    if sort_logic:
        file_ids = Sort_Files(file_names_p, str_1='G', str_2='p')
        file_names_p = np.array(file_names_p)[np.argsort(file_ids)]
        file_ids = Sort_Files(file_names_n, str_1='G', str_2='n')
        file_names_n = np.array(file_names_n)[np.argsort(file_ids)]

    file_names_pn = np.r_[file_names_p, file_names_n]
    for file_name_i in file_names_pn:

        # Open FITS and read data/header
        hdul = fits.open(file_name_i)
        data_cube = hdul[0].data
        header = hdul[0].header

        # Build WCS and compute Galactic coverage
        data_wcs = WCS.WCS(header)
        data_ranges_lb = Cal_Data_Range_LB(data_wcs, data_cube)

        data_ranges_lb_record.append(data_ranges_lb)

    return data_ranges_lb_record, file_names_pn


def Read_Spitzer_Files_All_BG(file_names_folder, end_str='I1.fits',sort_logic=False):
    """
    Scan a folder and collect WCS coverage for all background FITS files.

    Parameters
    ----------
    file_names_folder : str
        Folder path that contains *.fits files.
    end_str : str
        String indicating file type suffix.

    Returns
    -------
    data_ranges_lb_record : list
        List of [[l_max,l_min],[b_min,b_max]] for each file.
    file_names : list
        Full paths to the FITS files in the same order.
    """
    data_ranges_lb_record = []
    file_names = [file_names_folder + f for f in os.listdir(file_names_folder) if f.endswith(end_str)]

    # Sort files
    if sort_logic:
        file_ids = Sort_Files(file_names, str_1='_', str_2='+')
        file_names = np.array(file_names)[np.argsort(file_ids)]

    for file_name_i in file_names:
        hdul = fits.open(file_name_i)
        data_cube = hdul[0].data
        header = hdul[0].header

        # Build WCS and compute Galactic coverage
        data_wcs = WCS.WCS(header)
        data_ranges_lb = Cal_Data_Range_LB(data_wcs, data_cube)

        data_ranges_lb_record.append(data_ranges_lb)

    return data_ranges_lb_record, file_names


def Sort_Files(file_names, str_1='G', str_2='p'):
    """
    Extract numeric IDs from filenames and return as list for sorting.
    """
    file_ids = []
    for file_name in file_names:
        g_pos = file_name.find(str_1)
        plus_pos = file_name.find(str_2, g_pos)
        file_id = float(file_name[g_pos + 1: plus_pos])
        file_ids.append(file_id)
    return file_ids


def Read_Spitzer_Files_I(com_wcs, data_ranges_lb_record, file_names, reduce_range=0.1):
    """
    Find the FITS file containing a given Galactic coordinate point.

    Parameters
    ----------
    com_wcs : array-like
        [GLon, GLat] in degrees (bubble center).
    data_ranges_lb_record : list
        Precomputed WCS coverage for all files.
    file_names : list
        FITS filenames corresponding to coverage list.
    reduce_range : float
        Margin to avoid edge effects (in degrees).

    Returns
    -------
    file_name : str or None
        Selected FITS filename, if found.
    data_wcs : astropy.wcs.WCS or None
        WCS for the selected file.
    data_cube : ndarray or None
        FITS data cube loaded from file.
    """
    file_name, data_wcs, data_cube = None, None, None

    # Loop through coverage records to find first matching file
    for i in range(len(data_ranges_lb_record) + 1):
        if i < len(data_ranges_lb_record):
            data_ranges_lb = data_ranges_lb_record[i]

            # Check if coordinate lies within file coverage with margin
            if (com_wcs[0] > data_ranges_lb[0][1] + reduce_range and com_wcs[0] < data_ranges_lb[0][0] - reduce_range and
                com_wcs[1] > data_ranges_lb[1][0] + reduce_range and com_wcs[1] < data_ranges_lb[1][1] - reduce_range):
                break

    # Load FITS file if a valid index was found
    if i < len(data_ranges_lb_record):
        file_name = file_names[i]
        data_header = fits.getheader(file_name)
        data_cube = fits.getdata(file_name)
        data_wcs = WCS.WCS(data_header)

    return file_name, data_wcs, data_cube


def Get_RGB_Image_Infor(blue_file, green_file, red_file, center_wcs, region_size=(0.8*u.deg, 0.8*u.deg), intensity_pers=[2,99]):
    """
    Generate an RGB image from three FITS channels (blue, green, red) centered at a given Galactic coordinate.

    Workflow:
    1) Load the blue, green, and red FITS files.
    2) Reproject blue and green channels onto the red reference WCS.
    3) Apply clipping and NaN handling.
    4) Cut out the specified region using Cutout2D.
    5) Normalize and stretch each channel using percentile scaling.
    6) Generate an RGB image using Lupton RGB combination.
    7) Return RGB image and associated WCS.

    Parameters
    ----------
    blue_file, green_file, red_file : str
        Paths to FITS files for each channel.
    center_wcs : array-like
        Center coordinates [l, b] in degrees (Galactic).
    region_size : tuple of astropy.units.Quantity
        Size of the cutout region (longitude, latitude) in degrees.
    intensity_per : float
        Percentile for intensity normalization.

    Returns
    -------
    rgb_image : ndarray
        RGB image array (uint8, 0-255).
    ref_wcs : astropy.wcs.WCS
        WCS corresponding to the RGB image.
    gal_wcs : astropy.wcs.WCS
        Galactic WCS for plotting.
    pix_scale_arcmin : float
        Pixel scale in arcminutes.
    """
    # Load FITS HDUs and extract 2D data
    blue_hdu = fits.open(blue_file)[0]
    blue_data = np.squeeze(blue_hdu.data)
    blue_wcs = WCS.WCS(blue_hdu.header).celestial

    green_hdu = fits.open(green_file)[0]
    green_data = np.squeeze(green_hdu.data)
    green_wcs = WCS.WCS(green_hdu.header).celestial

    red_hdu = fits.open(red_file)[0]
    red_data = np.squeeze(red_hdu.data)
    red_wcs = WCS.WCS(red_hdu.header).celestial

    # Ensure all data are 2D
    if any(d.ndim != 2 for d in [blue_data, green_data, red_data]):
        raise ValueError("Data is not 2D; please check FITS files.")

    # Center coordinate in Galactic
    center = SkyCoord(l=center_wcs[0]*u.deg, b=center_wcs[1]*u.deg, frame='galactic')

    # Reference WCS is the red channel
    ref_wcs = red_wcs
    
    # Cutout each channel to the specified region
    try:
        cutout_blue = Cutout2D(blue_data, position=center, size=region_size, wcs=blue_wcs)
        cutout_green = Cutout2D(green_data, position=center, size=region_size, wcs=green_wcs)
        cutout_red = Cutout2D(red_data, position=center, size=region_size, wcs=red_wcs)

        blue_data = cutout_blue.data
        green_data = cutout_green.data
        red_data = cutout_red.data
        blue_wcs = cutout_blue.wcs
        green_wcs = cutout_green.wcs
        red_wcs = cutout_red.wcs
        
        ref_wcs = red_wcs
    except ValueError as e:
        print(f"Cutout error: {e}; using full image.")

    # Reference shape is the red channel
    ref_shape = red_data.shape

    # Reproject blue and green channels onto red reference WCS
    blue_reproj, _ = reproject_interp((blue_data, blue_wcs), ref_wcs.to_header(), shape_out=ref_shape)
    green_reproj, _ = reproject_interp((green_data, green_wcs), ref_wcs.to_header(), shape_out=ref_shape)
    red_data = red_data  # red is reference

    # Convert NaNs to 0 and ensure float32
    blue_reproj = np.nan_to_num(blue_reproj.astype(np.float32), nan=0.0)
    green_reproj = np.nan_to_num(green_reproj.astype(np.float32), nan=0.0)
    red_data = np.nan_to_num(red_data.astype(np.float32), nan=0.0)

    # Stretching and normalization functions
    def stretch(image, median_div=1.0, clip_min=0, power=0.5):
        image = np.clip(image, clip_min, None)
        median = np.nanmedian(image[image > 0]) if np.any(image > 0) else 1.0
        norm = image / (median / median_div)
        return np.arcsinh(norm ** power)

    def percentile_normalize(data, pmin=5, pmax=95):
        lo, hi = np.nanpercentile(data, [pmin, pmax])
        data = np.clip(data, lo, hi)
        return (data - lo) / (hi - lo + 1e-8)

    # Normalize channels
    blue_p = percentile_normalize(blue_reproj, intensity_pers[0], intensity_pers[1])
    green_p = percentile_normalize(green_reproj, intensity_pers[0], intensity_pers[1])
    red_p = percentile_normalize(red_data, intensity_pers[0], intensity_pers[1])

    # Apply stretching
    blue_st = stretch(blue_p, median_div=1, power=1)
    green_st = stretch(green_p, median_div=1, power=1)
    red_st = stretch(red_p, median_div=1, power=1)

    # Combine RGB using Lupton method
    rgb = make_lupton_rgb(red_st, green_st, blue_st, minimum=0, stretch=12, Q=4)

    # Normalize to uint8
    rgb_image = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255, 0, 255).astype(np.uint8)

    # Pixel scale in arcminutes
    pix_scale_arcmin = ref_wcs.proj_plane_pixel_scales()[0].value * 60

    # Create Galactic WCS for the image
    cdelt_ra, cdelt_dec = ref_wcs.wcs.cdelt
    ny, nx, _ = rgb_image.shape
    cx = nx / 2
    cy = ny / 2
    center_icrs = ref_wcs.pixel_to_world(cx, cy)
    center_gal = center_icrs.galactic
    gal_wcs = WCS.WCS(naxis=2)
    gal_wcs.wcs.ctype = ['GLON-TAN', 'GLAT-TAN']
    gal_wcs.wcs.crval = [center_gal.l.deg, center_gal.b.deg]
    gal_wcs.wcs.crpix = [cx, cy]
    gal_wcs.wcs.cdelt = [cdelt_ra, cdelt_dec]
    gal_wcs.wcs.cunit = ['deg','deg']

    return rgb_image, ref_wcs, gal_wcs, pix_scale_arcmin, red_st, green_st, blue_st


def Add_Bubble_Infor_To_Spitzer(bubbleObj, data_wcs_Sp):
    """
    Transform bubble coordinates and skeleton to Spitzer WCS pixel frame.

    Steps:
    1) Translate bubble center from its original WCS to Spitzer WCS pixels.
    2) Transform skeleton ellipse coordinates from original WCS to Spitzer WCS pixels.
    3) Compute skeleton center in Spitzer pixel frame.

    Parameters
    ----------
    bubbleObj : object
        Object containing bubble center and skeleton coordinates.
    data_wcs_Sp : astropy.wcs.WCS
        WCS of the Spitzer cutout for the bubble.

    Returns
    -------
    bubble_com_Sp : ndarray
        Bubble center in Spitzer pixel coordinates.
    skeleton_coords_ellipse_Sp : ndarray
        Skeleton ellipse coordinates in Spitzer pixel coordinates.
    skeleton_com_Sp : ndarray
        Skeleton center in Spitzer pixel coordinates.
    """
    # Translate bubble center to Spitzer pixel coordinates
    bubble_com_Sp = BFTools.Translate_Coords_LBV(
        np.array([bubbleObj.bubble_com_item_wcs]), data_wcs_Sp,
        pix2world=False, world2pix=True
    )

    # Flip skeleton ellipse coordinates and append a zero velocity axis
    skt_coords_ellipse = np.flip(bubbleObj.skeleton_coords_ellipse, axis=1)
    skt_coords_ellipse = np.c_[skt_coords_ellipse, np.zeros(len(skt_coords_ellipse))]

    # Transform skeleton to original WCS world coordinates
    skt_coords_LBV_wcs = BFTools.Translate_Coords_LBV(
        skt_coords_ellipse, bubbleObj.clumpsObj.data_wcs, pix2world=True, world2pix=False
    )

    # Transform skeleton to Spitzer WCS pixel coordinates
    skeleton_coords_ellipse_Sp = BFTools.Translate_Coords_LBV(
        skt_coords_LBV_wcs, data_wcs_Sp, pix2world=False, world2pix=True
    )

    # Compute skeleton center as mean of ellipse points
    skeleton_com_Sp = np.mean(skeleton_coords_ellipse_Sp, axis=0)

    return bubble_com_Sp, skeleton_coords_ellipse_Sp, skeleton_com_Sp


def Cal_Spitzer_Infor(bubbleObj, data_ranges_lb_record, file_names, 
                      bubble_com_item_wcs=None, bubble_item=None, data_wcs_item=None,
                      Cut_Sp=1, reduce_range=0.05, intensity_pers=[10,99.9]):
    """
    Extract Spitzer cutouts for a bubble and compute associated metadata.

    Steps:
    1) Compute the bubble's Galactic extent and apply padding.
    2) Identify red, green, and blue FITS files containing the bubble.
    3) Generate an RGB cutout image centered on the bubble.
    4) Compute pixel scale, bounding boxes, and transform skeleton/bubble coordinates to the cutout frame.
    5) Store all results in bubbleObj.

    Parameters
    ----------
    bubbleObj : object
        Object containing bubble WCS and skeleton info.
    data_ranges_lb_record : list
        Precomputed coverage of Spitzer files.
    file_names : list
        FITS file paths corresponding to coverage records.
    Cut_Sp : float
        Fractional padding for the bubble extent (default 1 = 100%).
    reduce_range : float
        Margin to avoid selecting files too close to edges (in degrees).

    Returns
    -------
    has_files : bool
        True if all three RGB files are available and processed.
    """

    # Get bubble WCS and cube data
    if bubble_com_item_wcs is None:
        bubble_com_item_wcs = bubbleObj.bubble_com_item_wcs
    if bubble_item is None:
        bubble_item = bubbleObj.bubble_item
    if data_wcs_item is None:
        data_wcs_item = bubbleObj.data_wcs_item

    # Compute bubble extent with fractional padding
    data_ranges_lb = Cal_Data_Range_LB(data_wcs_item, bubble_item)
    delta_l = np.abs(data_ranges_lb[0][0] - data_ranges_lb[0][1]) * Cut_Sp
    delta_b = np.abs(data_ranges_lb[1][0] - data_ranges_lb[1][1]) * Cut_Sp
    img_center = np.around([np.min(data_ranges_lb[0]) + np.abs(data_ranges_lb[0][0] - data_ranges_lb[0][1])/2,\
                            np.min(data_ranges_lb[1]) + np.abs(data_ranges_lb[1][0] - data_ranges_lb[1][1])/2],3)
    
    # Bubble center coordinates
    com_wcs = bubble_com_item_wcs[:2]

    # Find corresponding Spitzer RGB files
    data_ranges_lb_record_red,data_ranges_lb_record_blue,data_ranges_lb_record_green = data_ranges_lb_record
    file_names_red,file_names_blue,file_names_green = file_names
    
    file_name_red, data_wcs_red, data_cube_red = Read_Spitzer_Files_I(
        com_wcs, data_ranges_lb_record_red, file_names_red, reduce_range=reduce_range
    )
    file_name_blue, data_wcs_blue, data_cube_blue = Read_Spitzer_Files_I(
        com_wcs, data_ranges_lb_record_blue, file_names_blue, reduce_range=reduce_range
    )
    file_name_green, data_wcs_green, data_cube_green = Read_Spitzer_Files_I(
        com_wcs, data_ranges_lb_record_green, file_names_green, reduce_range=reduce_range
    )

    # Process RGB image if all files exist
    if file_name_red is not None and file_name_blue is not None and file_name_green is not None:
        rgb_image, ref_wcs, data_wcs_Sp, pix_scale_arcmin, red_st, green_st, blue_st = Get_RGB_Image_Infor(
            file_name_blue, file_name_green, file_name_red, img_center,
            region_size=(delta_l*u.deg, delta_b*u.deg), intensity_pers=[10,99.9]
        )

        lb_item_start, lb_item_end, velocity_range, pixel_scale_Sp = BFTools.Cal_Item_WCS_Range(
            rgb_image.T, data_wcs_Sp
        )

        # Transform bubble and skeleton coordinates to Spitzer pixel frame
        bubble_com_Sp, skeleton_coords_ellipse_Sp, skeleton_com_Sp = Add_Bubble_Infor_To_Spitzer(bubbleObj, data_wcs_Sp)

        # Store results in bubbleObj
        bubbleObj.rgb_image_Sp = rgb_image
        bubbleObj.data_wcs_Sp = data_wcs_Sp
        bubbleObj.pix_scale_arcmin_Sp = pix_scale_arcmin
        bubbleObj.bubble_com_Sp = bubble_com_Sp
        bubbleObj.skeleton_coords_ellipse_Sp = skeleton_coords_ellipse_Sp
        bubbleObj.skeleton_com_Sp = skeleton_com_Sp
        has_files = True
    else:
        has_files = False

    return has_files


def Plot_Spitzer_Infor(bubbleObj, ax0=None, plot_bub=True, tick_logic=True,
                       grid_logic=True, spacing=None, overlay_logic=False, figsize=(8,6), fontsize=12):
    """
    Plot a Spitzer RGB cutout with optional bubble and skeleton overlays.

    Parameters
    ----------
    bubbleObj : object
        Contains RGB image, WCS, bubble and skeleton coordinates.
    ax0 : matplotlib.axes.Axes or None
        Axis to plot on. If None, a new figure is created.
    plot_bub : bool
        Whether to overlay bubble center and skeleton geometry.
    tick_logic : bool
        Whether to display WCS tick labels.
    grid_logic : bool
        Whether to display coordinate grid.
    spacing : astropy.units.Quantity or None
        Optional tick spacing.
    overlay_logic : bool
        If True, overlay FK5 (J2000) grid.
    figsize : tuple
        Figure size.
    fontsize : int
        Font size for labels.

    Returns
    -------
    ax0 : matplotlib.axes.Axes
        Axis containing the plot.
    """
    rgb_image_Sp = bubbleObj.rgb_image_Sp
    data_wcs_Sp = bubbleObj.data_wcs_Sp
    pix_scale_arcmin_Sp = bubbleObj.pix_scale_arcmin_Sp
    bubble_com_Sp = bubbleObj.bubble_com_Sp
    skeleton_coords_ellipse_Sp = bubbleObj.skeleton_coords_ellipse_Sp
    skeleton_com_Sp = bubbleObj.skeleton_com_Sp

    # Create figure and axis if not provided
    if ax0 is None and tick_logic:
        fig = plt.figure(figsize=figsize)
        ax0 = fig.add_subplot(111, projection=data_wcs_Sp.celestial)
    elif ax0 is None and not tick_logic:
        fig, ax0 = plt.subplots(1,1,figsize=figsize)
        ax0.set_xticks([]), ax0.set_yticks([])

    # Setup WCS ticks and labels
    if tick_logic:
        ax0.set_xlabel("Galactic Longitude", fontsize=fontsize)
        ax0.set_ylabel("Galactic Latitude", fontsize=fontsize)
        ax0.coords[0].set_ticklabel(fontproperties={'family': 'DejaVu Sans'})
        ax0.coords[1].set_ticklabel(fontproperties={'family': 'DejaVu Sans'})

        lon = ax0.coords[0]
        lat = ax0.coords[1]
        lon.set_major_formatter("d.d")
        lat.set_major_formatter("d.d")

        if spacing:
            lon.set_ticks(spacing=spacing)
            lat.set_ticks(spacing=spacing)

        ax0.tick_params(axis='both', which='major', labelsize=fontsize)
        if grid_logic:
            ax0.coords.grid(alpha=0.5)

    # Display the RGB image
    ax0.imshow(rgb_image_Sp, origin='lower')

    # Overlay bubble and skeleton if requested
    if plot_bub:
        ax0.scatter(bubble_com_Sp[0][0], bubble_com_Sp[0][1], color='green', marker='o', s=40, label="Cavity Com")
        ax0.scatter(skeleton_com_Sp[0], skeleton_com_Sp[1], color='lime', marker='*', s=40, label="Fitted Intensity Center")
        ax0.plot(skeleton_coords_ellipse_Sp[:,0], skeleton_coords_ellipse_Sp[:,1],
                 linewidth=2, color='lime', linestyle='-.', label="Fitted Intensity Skeleton")

    # Overlay FK5 grid if requested
    if overlay_logic:
        overlay = ax0.get_coords_overlay('fk5')
        overlay.grid(color='white', ls='dotted', lw=2)
        overlay[0].set_axislabel('Right Ascension (J2000)', fontsize=16)
        overlay[1].set_axislabel('Declination (J2000)', fontsize=16)
        overlay[0].set_ticklabel(fontproperties={'family': 'DejaVu Sans'})
        overlay[1].set_ticklabel(fontproperties={'family': 'DejaVu Sans'})

    if plot_bub:
        ax0.legend(fontsize=fontsize, loc='upper right')

    return ax0


def Plot_RGB_Img(rgb_image,gal_wcs,tick_logic=True,grid_logic=True,spacing=None,overlay_logic=False,figsize=(8,6),fontsize=12):
    fig = plt.figure(figsize=figsize)
    if tick_logic:
        ax0 = fig.add_subplot(111,projection=gal_wcs.celestial)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.color'] = 'green'
        plt.rcParams['ytick.color'] = 'green'
        plt.xlabel("Galactic Longitude",fontsize=fontsize)
        plt.ylabel("Galactic Latitude",fontsize=fontsize)
        ax0.coords[0].set_ticklabel(fontproperties={'family': 'DejaVu Sans'})
        ax0.coords[1].set_ticklabel(fontproperties={'family': 'DejaVu Sans'})
        lon = ax0.coords[0]
        lat = ax0.coords[1]
        lon.set_major_formatter("d.d")
        lat.set_major_formatter("d.d")
        ax0.tick_params(axis='both', which='major', labelsize=fontsize)
        if spacing != None:
            lon.set_ticks(spacing=spacing)
            lat.set_ticks(spacing=spacing)
        if grid_logic:
            ax0.coords.grid(alpha=0.5)
    else:
        ax0 = fig.add_subplot(111)
        ax0.set_xticks([]), ax0.set_yticks([])

    ax0.imshow(rgb_image, origin='lower')
    
    if overlay_logic:
        overlay = ax0.get_coords_overlay('fk5')
        overlay.grid(color='white', ls='dotted', lw=2)
        overlay[0].set_axislabel('Right Ascension (J2000)', fontsize=16)
        overlay[1].set_axislabel('Declination (J2000)', fontsize=16)
        overlay[0].set_ticklabel(fontproperties={'family': 'DejaVu Sans'})
        overlay[1].set_ticklabel(fontproperties={'family': 'DejaVu Sans'})

    return ax0



