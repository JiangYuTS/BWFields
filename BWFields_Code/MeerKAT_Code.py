import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.wcs as WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
import warnings
import os
warnings.filterwarnings('ignore')


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
    - WCS object may have naxis=3 or naxis=4 depending on cube definition.

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
        (Kept in this order to match later selection logic.)
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
        [nx - 1, ny - 1]])

    # Convert pixel corners to world coordinates.
    # For 3D WCS: (x, y, v)
    # For 4D WCS: (x, y, v, stokes) or similar
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
    
    # Extract l/b bounds (deg)
    l_min = galactic_coords.l.deg.min()
    l_max = galactic_coords.l.deg.max()
    b_min = galactic_coords.b.deg.min()
    b_max = galactic_coords.b.deg.max()

    # Output format consistent with later selection tests
    data_ranges_lb = [[l_max, l_min], [b_min, b_max]]
    return data_ranges_lb


def Read_MeerKAT_Files_All(file_names_folder,layer=0):
    """
    Scan a folder and collect WCS coverage for all MeerKAT FITS files.

    Parameters
    ----------
    file_names_folder : str
        Folder path that contains *.fits files.

    Returns
    -------
    data_ranges_lb_record : list
        A list of [[l_max,l_min],[b_min,b_max]] for each file.
    file_names : list
        Full paths to the FITS files (same order as data_ranges_lb_record).
    """
    data_ranges_lb_record = []
    file_names = []

    file_list = os.listdir(file_names_folder)
    for file_name_i in file_list:
        if file_name_i.endswith('.fits'):
            # Open FITS and read data/header
            hdul = fits.open(file_names_folder + file_name_i)

            # NOTE: This assumes data structure like data[0].data[0] -> (nv,ny,nx)
            # Depending on FITS, you may need to verify dimension ordering.
            if layer>=0:
                data_cube = hdul[0].data[layer]
            else:
                data_cube = hdul[0].data
            header = hdul[0].header

            # Build WCS and compute (l,b) coverage
            data_wcs = WCS.WCS(header)
            data_ranges_lb = Cal_Data_Range_LB(data_wcs, data_cube)

            data_ranges_lb_record.append(data_ranges_lb)
            file_names.append(file_names_folder + file_name_i)

    return data_ranges_lb_record, file_names


def Read_MeerKAT_Files_I(com_wcs, data_ranges_lb_record, file_names, reduce_range=0.1):
    """
    Find the MeerKAT FITS file that contains a given Galactic coordinate point.

    Parameters
    ----------
    com_wcs : array-like
        [GLon, GLat] in degrees (bubble center).
    data_ranges_lb_record : list
        Precomputed coverage list from Read_MeerKAT_Files_All.
    file_names : list
        FITS filenames aligned with data_ranges_lb_record.
    reduce_range : float
        Margin in degrees to avoid selecting a file when the point is too close to edges.

    Returns
    -------
    file_name : str or None
        The selected FITS filename, if found.
    data_wcs : astropy.wcs.WCS or None
        WCS for the selected file.
    data_cube : ndarray or None
        FITS data cube loaded from file.
    """
    file_name, data_wcs, data_cube = None, None, None

    # Iterate over all coverage records and stop at first match
    for i in range(len(data_ranges_lb_record) + 1):
        if i < len(data_ranges_lb_record):
            data_ranges_lb = data_ranges_lb_record[i]

            # Check if com_wcs is inside the file coverage with a safety margin
            if (com_wcs[0] > data_ranges_lb[0][1] + reduce_range and com_wcs[0] < data_ranges_lb[0][0] - reduce_range and
                com_wcs[1] > data_ranges_lb[1][0] + reduce_range and com_wcs[1] < data_ranges_lb[1][1] - reduce_range):
                break

    # If a valid i was found, load that file
    if i < len(data_ranges_lb_record):
        file_name = file_names[i]
        data_header = fits.getheader(file_name)
        data_cube = fits.getdata(file_name)  # NOTE: may return 3D/4D depending on FITS structure
        data_wcs = WCS.WCS(data_header)

    return file_name, data_wcs, data_cube


class MeerKATDataProcessor:
    """
    Class for processing SARAO MeerKAT 1.3 GHz Galactic Plane survey FITS data.

    Typical data format (example):
      - Original FITS data may be 4D: (1, 16, ny, nx)
      - After load_data(), stored as (16, ny, nx)
        where:
          layer 0: brightness map at ~1359.7 MHz (Jy/beam)
          layer 1: spectral index alpha
          layer 2..: subband images
    """

    def __init__(self, fits_file_path):
        """
        Initialize the processor.

        Parameters
        ----------
        fits_file_path : str
            Path to the MeerKAT FITS file.
        """
        self.fits_file_path = fits_file_path
        self.hdul = None
        self.data = None
        self.header = None
        self.wcs = None
        self.pixel_scale = None

    def load_data(self):
        """
        Load FITS data and construct a 2D WCS for spatial coordinate conversion.

        Steps:
        - Open FITS.
        - If data is 4D (1, nlayer, ny, nx), drop the first axis.
        - Build WCS using only celestial axes (naxis=2).
        - Estimate pixel scale from CDELT keywords (arcsec/pixel).
        """
        try:
            self.hdul = fits.open(self.fits_file_path)
            self.data = self.hdul[0].data
            self.header = self.hdul[0].header

            # Handle 4D cube: (1, 16, ny, nx) -> (16, ny, nx)
            if len(self.data.shape) == 4:
                self.data = self.data[0]

            # Build 2D celestial WCS (spatial axes only)
            self.wcs = WCS.WCS(self.header, naxis=2)

        except Exception as e:
            print(f"加载数据时出错: {e}")

        # Estimate pixel scale (arcsec/pixel); fall back to typical MeerKAT scale if missing
        try:
            cdelt1 = abs(self.header.get('CDELT1', 1)) * 3600
            cdelt2 = abs(self.header.get('CDELT2', 1)) * 3600
            pixel_scale = (cdelt1 + cdelt2) / 2
        except Exception:
            pixel_scale = 8.0

        self.pixel_scale = pixel_scale

    def get_layer_info(self):
        """
        Optionally print layer descriptions (brightness, spectral index, subbands).

        Note: Printing is currently commented out.
        """
        if self.data is None:
            print("请先加载数据")
            return

        layer_descriptions = {
            0: "1359.7 MHz亮度 (Jy/beam)",
            1: "谱指数 alpha (无单位)",
        }

        # Add subband descriptions for remaining layers
        for i in range(2, min(16, self.data.shape[0])):
            if i == 8 or i == 9:
                layer_descriptions[i] = f"子频段 {i-1} (RFI屏蔽)"
            else:
                layer_descriptions[i] = f"子频段 {i-1} (Jy/beam)"

    def save_cropped_data(self, cropped_result, output_prefix='meerkat_cropped', save_logic=False):
        """
        Save cropped data to new FITS files.

        Updates header:
        - CRPIX1/CRPIX2 adjusted by crop offsets
        - NAXIS1/NAXIS2 updated to new image size

        Output:
        - brightness layer as separate file
        - spectral index layer as separate file
        - full cube as separate file

        Parameters
        ----------
        cropped_result : dict
            Output of extract_longitude_range(), containing data and pixel bounds.
        output_prefix : str
            Prefix for output filenames.
        save_logic : bool
            If True, write FITS files; if False, do nothing (kept for controlled I/O).
        """
        if cropped_result is None:
            print("没有数据可保存")
            return

        data = cropped_result['data']

        # Copy header and update reference pixel positions for the cropped subimage
        new_header = self.header.copy()
        bounds = cropped_result['pixel_bounds']

        if 'CRPIX1' in new_header:
            new_header['CRPIX1'] -= bounds['col_min']
        if 'CRPIX2' in new_header:
            new_header['CRPIX2'] -= bounds['row_min']

        # Update image size keywords
        new_header['NAXIS1'] = data.shape[2]
        new_header['NAXIS2'] = data.shape[1]

        # Build filename suffix based on cropped coordinate range
        coord_range = cropped_result.get('coord_range', {})
        if coord_range:
            lon_min = coord_range['lon_min']
            lon_max = coord_range['lon_max']
            lat_min = coord_range.get('lat_min', None)
            lat_max = coord_range.get('lat_max', None)

            if lat_min is not None and lat_max is not None:
                suffix = f"L{lon_min:.1f}-{lon_max:.1f}_B{lat_min:.1f}-{lat_max:.1f}"
            else:
                suffix = f"L{lon_min:.1f}-{lon_max:.1f}"
        else:
            suffix = "cropped"

        # Optional actual saving
        if save_logic:
            if data.shape[0] > 0:
                fits.writeto(f'{output_prefix}_brightness_{suffix}.fits',
                             data[0], new_header, overwrite=True)

            if data.shape[0] > 1:
                fits.writeto(f'{output_prefix}_spectral_index_{suffix}.fits',
                             data[1], new_header, overwrite=True)

            fits.writeto(f'{output_prefix}_cube_{suffix}.fits',
                         data, new_header, overwrite=True)

            print(f"截取的数据已保存为 {output_prefix}_*_{suffix}.fits")

    def extract_sources(self, brightness_threshold=0.001, min_area=5):
        """
        Very simple source extraction on brightness layer using thresholding + connected components.

        Parameters
        ----------
        brightness_threshold : float
            Threshold in Jy/beam.
        min_area : int
            Minimum number of pixels for a detection to be accepted.

        Returns
        -------
        sources : list of dict or None
            Each dict includes center (x,y), area, peak brightness, total flux.
        """
        # NOTE: extract_brightness_map() is referenced but not defined in this snippet.
        # You likely have it in other modules.
        brightness = self.extract_brightness_map()
        if brightness is None:
            return None

        from scipy import ndimage

        source_mask = brightness > brightness_threshold
        labeled_array, num_features = ndimage.label(source_mask)

        sources = []
        for i in range(1, num_features + 1):
            source_pixels = labeled_array == i
            area = np.sum(source_pixels)

            if area >= min_area:
                y_coords, x_coords = np.where(source_pixels)
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                peak_brightness = np.max(brightness[source_pixels])
                total_flux = np.sum(brightness[source_pixels])

                sources.append({
                    'id': i,
                    'center_x': center_x,
                    'center_y': center_y,
                    'area_pixels': area,
                    'peak_brightness': peak_brightness,
                    'total_flux': total_flux
                })

        print(f"检测到 {len(sources)} 个源 (阈值: {brightness_threshold} Jy/beam)")
        return sources

    
    def extract_longitude_range(self, lon_min=11.5, lon_max=12.5, lat_min=None, lat_max=None):
        """
        Crop data by Galactic longitude (and optionally latitude) in WCS space, 
        directly converting the (lon, lat) range to pixel coordinates using all_world2pix.
    
        Optimized Workflow:
        1) Convert lon_min, lon_max, lat_min, lat_max directly to pixel coordinates using all_world2pix.
        2) Crop the data based on these pixel coordinates.
    
        Parameters
        ----------
        lon_min, lon_max : float
            Galactic longitude range in degrees.
            If lon_min > lon_max, selection wraps around 0 degrees.
        lat_min, lat_max : float or None
            Optional Galactic latitude range.
    
        Returns
        -------
        dict or None
            Contains cropped cube, mask, lon/lat grids, pixel bounds, and coord range.
        """
        if self.data is None:
            print("请先加载数据")
            return None
    
        # Create an array for the world coordinates to convert
        world_coords = np.array([[lon_min, lat_min] if lat_min is not None else [lon_min, None], 
                                 [lon_max, lat_max] if lat_max is not None else [lon_max, None]])
    
        try:
            # Convert world coordinates (lon, lat) to pixel coordinates using all_world2pix
            pixel_coords = self.wcs.all_world2pix(world_coords, 0)  # 0 indicates we're working with the first plane (2D)
    
            # Extract pixel bounds from the result
            lon_min_pixel, lat_min_pixel = pixel_coords[0]
            lon_max_pixel, lat_max_pixel = pixel_coords[1]
    
            # Calculate pixel bounds
            row_min = np.max([0,int(np.floor(lat_min_pixel))])
            row_max = np.min([int(np.ceil(lat_max_pixel)),self.data.shape[1]])
            col_min = np.min([int(np.floor(lon_min_pixel)),self.data.shape[2]])
            col_max = np.max([0,int(np.ceil(lon_max_pixel))])
    
            # Crop the data based on the pixel coordinates
            cropped_data = self.data[:, row_min:row_max, col_max:col_min]
            cropped_mask = (self.data[0, row_min:row_max, col_max:col_min] != 0)
            
            # Optional: update header to match the cropped region
            new_header = self.header.copy()
            if 'CRPIX1' in new_header:
                new_header['CRPIX1'] -= col_max
            if 'CRPIX2' in new_header:
                new_header['CRPIX2'] -= row_min
            new_header['NAXIS1'] = cropped_data.shape[2]
            new_header['NAXIS2'] = cropped_data.shape[1]
            self.new_header = new_header
    
            # Return cropped data and associated information
            return {
                'data': cropped_data,
                'mask': cropped_mask,
                'pixel_bounds': {
                    'row_min': row_min, 'row_max': row_max,
                    'col_min': col_min, 'col_max': col_max
                },
                'coord_range': {
                    'lon_min': lon_min, 'lon_max': lon_max,
                    'lat_min': lat_min, 'lat_max': lat_max
                }
            }
    
        except Exception as e:
            print(f"错误: {e}")
            return None
            

    def _extract_by_pixel_estimation(self, lon_min, lon_max, lat_min=None, lat_max=None):
        """
        Fallback cropping method based on linear approximation from FITS header keywords.

        This is less accurate than full WCS conversion but can work when WCS fails.
        """
        try:
            crval1 = self.header.get('CRVAL1', 0)
            crval2 = self.header.get('CRVAL2', 0)
            crpix1 = self.header.get('CRPIX1', 1)
            crpix2 = self.header.get('CRPIX2', 1)
            cdelt1 = self.header.get('CDELT1', 1)
            cdelt2 = self.header.get('CDELT2', 1)

            # Estimate pixel columns from longitude
            col_min = int((lon_min - crval1) / cdelt1 + crpix1 - 1)
            col_max = int((lon_max - crval1) / cdelt1 + crpix1 - 1)

            # Estimate pixel rows from latitude if provided
            if lat_min is not None and lat_max is not None:
                row_min = int((lat_min - crval2) / cdelt2 + crpix2 - 1)
                row_max = int((lat_max - crval2) / cdelt2 + crpix2 - 1)
            else:
                row_min, row_max = 0, self.data.shape[1] - 1

            # Clamp to valid pixel range
            col_min = max(0, min(col_min, self.data.shape[2] - 1))
            col_max = max(0, min(col_max, self.data.shape[2] - 1))
            row_min = max(0, min(row_min, self.data.shape[1] - 1))
            row_max = max(0, min(row_max, self.data.shape[1] - 1))

            cropped_data = self.data[:, row_min:row_max + 1, col_min:col_max + 1]

            return {
                'data': cropped_data,
                'pixel_bounds': {
                    'row_min': row_min, 'row_max': row_max,
                    'col_min': col_min, 'col_max': col_max
                }
            }

        except Exception as e:
            print(f"备用方法也失败: {e}")
            return None

    def close(self):
        """Close the FITS file handle."""
        if self.hdul:
            self.hdul.close()


def Extract_Slice(fits_file, lon_min=11.5, lon_max=12.5,
                  lat_min=None, lat_max=None, save_output=True):
    """
    Convenience wrapper: load a FITS file, crop by lon/lat, and optionally save outputs.

    Returns
    -------
    processor : MeerKATDataProcessor
        Processor instance containing loaded header/WCS and (optionally) new_header.
    cropped_result : dict or None
        Output from processor.extract_longitude_range().
    """
    processor = MeerKATDataProcessor(fits_file)

    try:
        processor.load_data()

        cropped_result = processor.extract_longitude_range(
            lon_min=lon_min, lon_max=lon_max,
            lat_min=lat_min, lat_max=lat_max
        )

        if cropped_result is not None and save_output:
            processor.save_cropped_data(
                cropped_result,
                f'meerkat_L{lon_min:.1f}-{lon_max:.1f}'
            )

        return processor, cropped_result

    finally:
        processor.close()


def Add_Bubble_Infor_To_MK(processor, bubbleObj):
    """
    Project bubble information (from bubbleObj WCS / pixel space) into the cropped MeerKAT pixel frame.

    What it does:
    - Uses processor.new_header (cropped header) to build a new WCS.
    - Converts bubble center (world) -> MeerKAT cropped pixel coordinates.
    - Converts skeleton ellipse coordinates from bubbleObj pixel->world, then world->MeerKAT pixel.
    - Computes skeleton center in MeerKAT pixel frame.
    - Stores these arrays on the processor for plotting.

    Parameters
    ----------
    processor : MeerKATDataProcessor
        Must already have processor.new_header from extract_longitude_range().
    bubbleObj : object
        Must provide:
          - bubble_com_item_wcs (world coords for bubble center)
          - clumpsObj.data_wcs (WCS of the bubble dataset)
          - skeleton_coords_ellipse (pixel coords in bubble dataset)
    """
    # Bubble center in world coordinates (lon, lat)
    com_wcs = bubbleObj.bubble_com_item_wcs[:2]

    # Build WCS corresponding to the CROPPED MeerKAT map (important!)
    data_wcs_MK_new = WCS.WCS(processor.new_header)

    # Convert bubble world coord -> MeerKAT cropped pixel
    # all_world2pix signature depends on WCS dimension; here uses 5 args (lon,lat,0,0,0)
    if data_wcs_MK_new.naxis == 2:
        bubble_com_MK_T = data_wcs_MK_new.all_world2pix(com_wcs[0], com_wcs[1])
    elif data_wcs_MK_new.naxis == 3:
        bubble_com_MK_T = data_wcs_MK_new.all_world2pix(com_wcs[0], com_wcs[1], 0, 0)
    elif data_wcs_MK_new.naxis == 4:
        bubble_com_MK_T = data_wcs_MK_new.all_world2pix(com_wcs[0], com_wcs[1], 0, 0, 0)

    # Reorder into (row, col) style used later in plotting
    bubble_com_MK = np.array([bubble_com_MK_T[1], bubble_com_MK_T[0]])

    # Convert skeleton ellipse coordinates from bubble pixel -> bubble world coordinates
    skeleton_coords_ellipse_wsc_T = bubbleObj.clumpsObj.data_wcs.all_pix2world(
        bubbleObj.skeleton_coords_ellipse[:, 1],
        bubbleObj.skeleton_coords_ellipse[:, 0],
        np.array([0] * len(bubbleObj.skeleton_coords_ellipse)),
        0
    )
    skeleton_coords_ellipse_wsc = np.c_[skeleton_coords_ellipse_wsc_T[0], skeleton_coords_ellipse_wsc_T[1]]

    # Convert skeleton world coordinates -> MeerKAT cropped pixel coordinates
    if data_wcs_MK_new.naxis == 2:
        skeleton_coords_ellipse_MK_T = data_wcs_MK_new.all_world2pix(
            skeleton_coords_ellipse_wsc[:, 0],
            skeleton_coords_ellipse_wsc[:, 1]
        )
    elif data_wcs_MK_new.naxis == 3:
        skeleton_coords_ellipse_MK_T = data_wcs_MK_new.all_world2pix(
            skeleton_coords_ellipse_wsc[:, 0],
            skeleton_coords_ellipse_wsc[:, 1],
            np.array([0] * len(skeleton_coords_ellipse_wsc)),
            0
        )
    elif data_wcs_MK_new.naxis == 4:
        skeleton_coords_ellipse_MK_T = data_wcs_MK_new.all_world2pix(
            skeleton_coords_ellipse_wsc[:, 0],
            skeleton_coords_ellipse_wsc[:, 1],
            np.array([0] * len(skeleton_coords_ellipse_wsc)),
            np.array([0] * len(skeleton_coords_ellipse_wsc)),
            0
        )
        
    skeleton_coords_ellipse_MK = np.c_[skeleton_coords_ellipse_MK_T[1], skeleton_coords_ellipse_MK_T[0]]

    # Skeleton center in MeerKAT pixel frame
    skeleton_com_MK = np.mean(skeleton_coords_ellipse_MK, axis=0)

    # Attach for downstream plotting
    processor.data_wcs_MK_new = data_wcs_MK_new
    processor.bubble_com_MK = bubble_com_MK
    processor.skeleton_coords_ellipse_MK = skeleton_coords_ellipse_MK
    processor.skeleton_com_MK = skeleton_com_MK


def Plot_Origin_Data(file_name, layer_index=0):
    """
    Plot the original (uncropped) FITS data in WCS celestial projection.

    NOTE:
    - This function uses `fontsize` but it is not defined inside the function.
      You may want to add fontsize as an argument or define it globally.

    Parameters
    ----------
    file_name : str
        FITS file to display.
    layer_index : int
        Layer index to display (currently hardcoded to data[0][0] in your code).

    Returns
    -------
    ax0 : matplotlib axis
        WCS axis handle for further overlays.
    """
    hdul = fits.open(file_name)
    data = hdul[0].data
    header = hdul[0].header
    data_wcs = WCS.WCS(header)

    fig = plt.figure(figsize=(8, 6))
    ax0 = fig.add_subplot(111, projection=data_wcs.celestial)

    # NOTE: currently plots data[0][0]; adjust if you want layer_index supported
    gci = ax0.imshow(
        data[0][0],
        origin='lower',
        cmap='hot',
        interpolation='none'
    )

    cbar = plt.colorbar(gci, pad=0)
    cbar.ax.tick_params(labelsize=fontsize)  # fontsize must exist
    cbar.set_label(label='K km s$^{-1}$', fontsize=fontsize)

    # Axis styling
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.color'] = 'green'
    plt.rcParams['ytick.color'] = 'green'
    plt.xlabel("Galactic Longitude", fontsize=fontsize)
    plt.ylabel("Galactic Latitude", fontsize=fontsize)

    ax0.coords[0].set_ticklabel(fontproperties={'family': 'DejaVu Sans'})
    ax0.coords[1].set_ticklabel(fontproperties={'family': 'DejaVu Sans'})
    ax0.tick_params(axis='both', which='major', labelsize=fontsize)

    lon = ax0.coords[0]
    lat = ax0.coords[1]
    lon.set_major_formatter("d.d")
    lat.set_major_formatter("d.d")

    return ax0


def Cal_MeerKAT_Infor(bubbleObj, data_ranges_lb_record, file_names, 
                      bubble_com_item_wcs=None, bubble_item=None, data_wcs_item=None,
                      Cut_MK=0.2, reduce_range=0.2):
    """
    Helper function to compute the (l,b) coverage of a bubble and locate the corresponding MeerKAT cutout.

    Steps:
    1) Calculate the (l,b) coverage of the bubble's data cube to estimate the region's size.
    2) Expand this region by a fractional padding (Cut_MK) to define the MeerKAT cutout window.
    3) Determine which MeerKAT FITS file contains the bubble center (Read_MeerKAT_Files_I).
    4) Extract the corresponding data slice from the MeerKAT cube (Extract_Slice).
    5) Project the bubble skeleton geometry into the MeerKAT cutout pixel frame.
    6) Store the results in the bubble object for later plotting.

    Parameters:
    ----------
    bubbleObj : object
        The bubble object containing information about the bubble center, data WCS, and other relevant fields.
    data_ranges_lb_record, file_names : list
        Outputs from Read_MeerKAT_Files_All, used to locate relevant MeerKAT files for the cutout.
    Cut_MK : float
        The fractional padding applied to the bubble's (l,b) extent when defining the MeerKAT cutout window.
    
    Returns:
    -------
    None
        Updates the `bubbleObj` with the computed cutout and associated processor instance.
    """
    
    # Get bubble WCS and cube data
    if bubble_com_item_wcs is None:
        bubble_com_item_wcs = bubbleObj.bubble_com_item_wcs
    if bubble_item is None:
        bubble_item = bubbleObj.bubble_item
    if data_wcs_item is None:
        data_wcs_item = bubbleObj.data_wcs_item

    # Calculate the (l,b) extent of the bubble's data and apply fractional padding
    data_ranges_lb = Cal_Data_Range_LB(data_wcs_item, bubble_item)
    delta_l = np.abs(data_ranges_lb[0][0] - data_ranges_lb[0][1])
    delta_b = np.abs(data_ranges_lb[1][0] - data_ranges_lb[1][1])

    # Define the extended cutout boundaries based on Cut_MK
    lon_max = data_ranges_lb[0][0] - delta_l * Cut_MK
    lon_min = data_ranges_lb[0][1] + delta_l * Cut_MK
    lat_min = data_ranges_lb[1][0] + delta_b * Cut_MK
    lat_max = data_ranges_lb[1][1] - delta_b * Cut_MK

    # Bubble center coordinates in (lon, lat)
    com_wcs = bubble_com_item_wcs[:2]

    # Locate the MeerKAT file that contains the bubble center and the required region
    file_name, data_wcs, data_cube = Read_MeerKAT_Files_I(
        com_wcs, data_ranges_lb_record, file_names, reduce_range=reduce_range
    )

    processor_MK = None
    cutted_result = None
    
    # If a valid file is found, extract the corresponding slice from the MeerKAT cube
    if file_name is not None:
        processor_MK, cutted_result = Extract_Slice(
            file_name, lon_min=lon_min, lon_max=lon_max,
            lat_min=lat_min, lat_max=lat_max, save_output=False
        )

        # Add the bubble geometry to the extracted data for visualization
        Add_Bubble_Infor_To_MK(processor_MK, bubbleObj)
        has_file = True
    else:
        has_file = False

    # Store results in the bubble object for further processing
    bubbleObj.processor_MK = processor_MK
    bubbleObj.cutted_result = cutted_result
    
    return has_file


def Plot_MeerKAT_Infor(bubbleObj, layer_index=0, ax0=None, plot_bub=True,
                         show_galactic_coords=True, colormap='hot', show_beam=False,
                         tick_logic=True, grid_logic=True, spacing=None,
                         fontsize=12, figsize=(8, 6), linewidth=2):
    """
    Plot a cropped MeerKAT slice (one layer) with optional bubble overlays.

    Features:
    - Auto-scales data units based on dynamic range (Jy/beam -> mJy/beam -> μJy/beam).
    - WCS plotting with Galactic Longitude/Latitude ticks and labels (if tick_logic=True).
    - Optionally overlays bubble center, skeleton center, and skeleton ellipse curve if plot_bub=True.
    - Optional beam size indicator (approximated based on pixel scale).

    Parameters:
    ----------
    bubbleObj : object
        The object containing the bubble center WCS, bubble cube, and other relevant fields.
    cropped_result : dict
        Result from `extract_longitude_range()` containing the cropped data slice.
    layer_index : int
        The index of the layer to plot (0 for brightness, 1 for spectral index, 2+ for subbands).
    plot_bub : bool
        If True, overlays bubble and skeleton geometry.
    show_beam : bool
        If True, draws an approximate beam circle.
    tick_logic, grid_logic : bool
        Control whether WCS ticks and coordinate grid are displayed.
    spacing : astropy.units.quantity or None
        Optional tick spacing for WCS axes.
    fontsize : int
        Font size for axis labels, ticks, and colorbar labels.
    figsize : tuple
        Size of the figure for plotting.
    linewidth : float
        Line width for plotting the skeleton curve.

    Returns:
    -------
    ax0 : matplotlib.axes.Axes
        The axis on which the plot was created.
    """
    
    # Extract relevant data from bubble object
    processor_MK = bubbleObj.processor_MK
    cutted_result = bubbleObj.cutted_result
    
    if cutted_result is None:
        return ax0

    data = cutted_result['data']
    if layer_index >= data.shape[0]:
        print(f"层索引 {layer_index} 超出范围")
        return

    layer_data = data[layer_index]

    # Create the figure and axis if not provided
    if ax0 is None and tick_logic:
        fig = plt.figure(figsize=figsize)
        ax0 = fig.add_subplot(111,projection=processor_MK.data_wcs_MK_new.celestial)
    elif ax0 is None and not tick_logic:
        fig, (ax0) = plt.subplots(1,1,figsize=figsize)
        ax0.set_xticks([]), ax0.set_yticks([])
        
    # WCS axis setup for ticks and grid logic
    if tick_logic:
        ax0.set_xlabel("Galactic Longitude", fontsize=fontsize)
        ax0.set_ylabel("Galactic Latitude", fontsize=fontsize)
        ax0.coords[0].set_ticklabel(fontproperties={'family': 'DejaVu Sans'})
        ax0.coords[1].set_ticklabel(fontproperties={'family': 'DejaVu Sans'})

        lon = ax0.coords[0]
        lat = ax0.coords[1]
        lon.set_major_formatter("d.d")
        lat.set_major_formatter("d.d")

        # Optional custom tick spacing
        if spacing:
            lon.set_ticks(spacing=spacing)
            lat.set_ticks(spacing=spacing)

        ax0.tick_params(axis='both', which='major', labelsize=fontsize)
        if grid_logic:
            ax0.coords.grid(alpha=0.5)

    # Scaling and unit selection based on data values
    data_abs_max = max(abs(np.nanmin(layer_data)), abs(np.nanmax(layer_data)))
    if layer_index == 0:  # Brightness layer
        if data_abs_max >= 1.0:
            data_display = layer_data
            unit_label = 'Jy/beam'
        elif data_abs_max >= 0.001:
            data_display = layer_data * 1000
            unit_label = 'mJy/beam'
        else:
            data_display = layer_data * 1e6
            unit_label = r'$\mu$Jy/beam'

    elif layer_index == 1:  # Spectral index layer
        data_display = layer_data
        unit_label = 'Spectral Index'

    else:  # Subband layers
        data_display = layer_data * (1000 if data_abs_max >= 0.001 else 1e6)
        unit_label = 'mJy/beam' if data_abs_max >= 0.001 else r'$\mu$Jy/beam'

    # Contrast scaling
    vmin, vmax = np.nanpercentile(data_display, [1, 99])
    gci = ax0.imshow(data_display, origin='lower', cmap=colormap, vmin=vmin, vmax=vmax)

    # Bubble and skeleton overlay
    if plot_bub:
        ax0.scatter(processor_MK.bubble_com_MK[1], processor_MK.bubble_com_MK[0], 
                    color='green', marker='o', s=40, label="Cavity Com")
        ax0.scatter(processor_MK.skeleton_com_MK[1], processor_MK.skeleton_com_MK[0], 
                    color='lime', marker='*', s=40, label="Fitted Intensity Center")
        ax0.plot(processor_MK.skeleton_coords_ellipse_MK[:, 1], processor_MK.skeleton_coords_ellipse_MK[:, 0], linewidth=linewidth, 
                 color='lime', linestyle='-.', label="Fitted Intensity Skeleton")

    # Colorbar and unit label
    cbar = plt.colorbar(gci, pad=0)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label(unit_label, fontsize=fontsize)

    # Beam size indicator (optional)
    if show_beam:
        beam_radius_pixels = max(2, int(20 / self.pixel_scale))
        circle = plt.Circle((beam_radius_pixels * 1.5, beam_radius_pixels * 1.5), beam_radius_pixels / 2, 
                            color='lime', fill=False, linewidth=2)
        ax0.add_patch(circle)

    # plt.tight_layout()

    # Add legend if bubbles are plotted
    if plot_bub:
        ax0.legend(fontsize=fontsize, loc='upper right')

    return ax0








