import numpy as np
from astropy.io import fits
import astropy.wcs as WCS
from astropy.wcs.utils import pixel_to_skycoord
from astropy.coordinates import SkyCoord
import astropy.units as u

from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
from astropy.nddata import Cutout2D
from astropy.visualization import make_lupton_rgb

from . import Spitzer_Code as SpCode  
from . import Bubble_Funs_Tools as BFTools 

def get_data_wcs(fname):
    """
    使用 fits.getdata / getheader
    自动跳过 PrimaryHDU
    """
    data = fits.getdata(fname)
    hdr  = fits.getheader(fname)

    if data.ndim == 3:
        data = data[0]

    if not np.isfinite(data).any():
        raise ValueError(f"No finite pixels in {fname}")

    wcs = WCS(hdr)
    return data, wcs


def unwrap_longitude(l_deg):
    l = np.asarray(l_deg) % 360.0
    ref = np.nanmedian(l)
    l[l < ref - 180] += 360
    l[l > ref + 180] -= 360
    return l


def galactic_range_effective_hspirepsw(fname, step=20):
    data, wcs = get_data_wcs(fname)

    # print(data.shape)
    yy, xx = np.where(np.isfinite(data))

    # 抽样加速
    if step > 1:
        idx = np.arange(xx.size)[::step]
        xx, yy = xx[idx], yy[idx]

    # print(xx, yy)
    sky = wcs.pixel_to_world(xx, yy)
    print('sky:',sky)
    gal = sky.galactic

    l = unwrap_longitude(gal.l.deg)
    b = gal.b.deg

    return l.min(), l.max(), b.min(), b.max()





def find_celestial_hdu(fname):
    """
    返回：(ext, data2d, wcs_celestial)
    自动找到包含天球WCS的图像HDU
    """
    with fits.open(fname, memmap=False) as hdul:
        candidates = []
        for ext, hdu in enumerate(hdul):
            if hdu.data is None:
                continue
            try:
                w = WCS(hdu.header)
            except Exception:
                continue

            if not w.has_celestial:
                continue

            data = hdu.data
            if data.ndim == 3:
                data = data[0]

            if data.ndim != 2:
                continue

            if not np.isfinite(data).any():
                continue

            # 只用天球子WCS，避免其他轴干扰
            wc = w.celestial
            candidates.append((ext, data, wc))

        if not candidates:
            raise ValueError(
                "在这个 FITS 里没有找到包含天球WCS的图像HDU（has_celestial=False）。"
            )

        # 如果有多个候选：选像素数最大的（通常就是主sky map）
        candidates.sort(key=lambda x: x[1].size, reverse=True)
        return candidates[0]  # (ext, data, wcs_celestial)

def unwrap_longitude(l_deg):
    l = np.asarray(l_deg) % 360.0
    ref = np.nanmedian(l)
    l[l < ref - 180] += 360
    l[l > ref + 180] -= 360
    return l

def galactic_range_effective_hpacs(fname, step=20):
    ext, data, wcs = find_celestial_hdu(fname)

    yy, xx = np.where(np.isfinite(data))
    if step > 1:
        idx = np.arange(xx.size)[::step]
        xx, yy = xx[idx], yy[idx]

    sky = pixel_to_skycoord(xx, yy, wcs, origin=0)
    gal = sky.galactic

    l = unwrap_longitude(gal.l.deg)
    b = gal.b.deg

    return ext, (l.min(), l.max(), b.min(), b.max())


# ====== 2) 自动找“包含天球WCS”的 ext，并读 data+wcs ======
def load_celestial_image(fname):
    with fits.open(fname, memmap=False) as hdul:
        best = None
        for ext, hdu in enumerate(hdul):
            if hdu.data is None:
                continue
            try:
                w = WCS(hdu.header)
            except Exception:
                continue
            if not w.has_celestial:
                continue

            data = hdu.data
            if data.ndim == 3:
                data = data[0]
            if data.ndim != 2:
                continue
            if not np.isfinite(data).any():
                continue

            # 用 celestial 子WCS
            wc = w.celestial
            # 选最大图层作为主图
            score = data.size
            if (best is None) or (score > best[0]):
                best = (score, ext, data.astype(float), wc)

        if best is None:
            raise ValueError(f"{fname} 中没有找到 has_celestial=True 且有图像 data 的 HDU。")
        _, ext, data, wc = best
        return ext, data, wc


def robust_norm_asinh(img, pmin=1.0, pmax=99.7, asinh_a=3.0):
    """
    不用 PercentileInterval，直接用 np.nanpercentile
    输出 [0,1]
    """
    x = img.astype(np.float64)
    x[~np.isfinite(x)] = np.nan
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)

    vmin = np.nanpercentile(x[finite], pmin)
    vmax = np.nanpercentile(x[finite], pmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(x)

    y = np.clip(x, vmin, vmax)
    y = (y - vmin) / (vmax - vmin)

    # asinh stretch
    y = np.arcsinh(asinh_a * y) / np.arcsinh(asinh_a)
    y[~np.isfinite(y)] = 0.0
    return np.clip(y, 0.0, 1.0)

def crop_to_common_valid(b, g, r):
    mask = np.isfinite(b) & np.isfinite(g) & np.isfinite(r)
    if not mask.any():
        raise ValueError("三波段没有共同有效像素（可能根本不重叠，或重投影/文件选错）。")
    yy, xx = np.where(mask)
    y0, y1 = yy.min(), yy.max() + 1
    x0, x1 = xx.min(), xx.max() + 1
    return (b[y0:y1, x0:x1], g[y0:y1, x0:x1], r[y0:y1, x0:x1], (x0, x1, y0, y1), mask)

def galactic_range_for_image(wcs_icrs, shape, mask=None, sample_step=25):
    """
    给定 ICRS WCS 和图像 shape，估算银经银纬范围
    - corners: 四角范围
    - effective: 若提供 mask，则用有效像素抽样估算范围
    """
    ny, nx = shape

    # 四角
    corners_x = np.array([0, nx-1, 0, nx-1])
    corners_y = np.array([0, 0, ny-1, ny-1])
    sky_c = pixel_to_skycoord(corners_x, corners_y, wcs_icrs, origin=0)
    gal_c = sky_c.galactic
    l_c = gal_c.l.deg
    b_c = gal_c.b.deg

    # unwrap l 便于 min/max
    def unwrap_l(l):
        l = np.asarray(l) % 360.0
        ref = np.nanmedian(l)
        l[l < ref - 180] += 360
        l[l > ref + 180] -= 360
        return l

    out = {}
    lcu = unwrap_l(l_c)
    out["corners"] = (lcu.min()%360, lcu.max()%360, np.nanmin(b_c), np.nanmax(b_c))

    if mask is not None and mask.any():
        yy, xx = np.where(mask)
        if sample_step > 1:
            idx = np.arange(xx.size)[::sample_step]
            xx, yy = xx[idx], yy[idx]
        sky = pixel_to_skycoord(xx, yy, wcs_icrs, origin=0)
        gal = sky.galactic
        lu = unwrap_l(gal.l.deg)
        bu = gal.b.deg
        out["effective"] = (lu.min()%360, lu.max()%360, np.nanmin(bu), np.nanmax(bu))
    else:
        out["effective"] = None

    return out




def Get_RGB_Image_Infor(blue_file, green_file, red_file, center_wcs, region_size=(0.8*u.deg, 0.8*u.deg), intensity_pers=[5,99], gamma=1.2):
    blue_hdu = fits.open(blue_file)[1]
    blue_data = np.squeeze(blue_hdu.data)
    blue_wcs = WCS.WCS(blue_hdu.header).celestial
    
    green_hdu = fits.open(green_file)[1]
    green_data = np.squeeze(green_hdu.data)
    green_wcs = WCS.WCS(green_hdu.header).celestial
    
    red_hdu = fits.open(red_file)[1]
    red_data = np.squeeze(red_hdu.data)
    red_wcs = WCS.WCS(red_hdu.header).celestial
    
    # Ensure all data are 2D
    if any(d.ndim != 2 for d in [blue_data, green_data, red_data]):
        raise ValueError("Data is not 2D; please check FITS files.")
    
    # Center coordinate in Galactic
    center = SkyCoord(l=center_wcs[0]*u.deg, b=center_wcs[1]*u.deg, frame='galactic')
    
    
    cutout_blue = Cutout2D(blue_data, position=center, size=region_size, wcs=blue_wcs)
    cutout_green = Cutout2D(green_data, position=center, size=region_size, wcs=green_wcs)
    cutout_red = Cutout2D(red_data, position=center, size=region_size, wcs=red_wcs)
    
    blue_data = np.nan_to_num(cutout_blue.data)
    green_data = np.nan_to_num(cutout_green.data)
    red_data = np.nan_to_num(cutout_red.data)
    blue_wcs = cutout_blue.wcs
    green_wcs = cutout_green.wcs
    red_wcs = cutout_red.wcs
    
    (target_wcs, target_shape) = find_optimal_celestial_wcs(
        [(blue_data,  blue_wcs), (green_data, green_wcs), (red_data, red_wcs)],
        frame="galactic", resolution=None)
    
    blue_data,  _ = reproject_interp((blue_data,  blue_wcs),  target_wcs, shape_out=target_shape)
    green_data, _ = reproject_interp((green_data, green_wcs), target_wcs, shape_out=target_shape)
    red_data, _ = reproject_interp((red_data, red_wcs), target_wcs, shape_out=target_shape)

    # Stretching and normalization functions
    def stretch(image, median_div=1.0, clip_min=0, power=0.5):
        image = np.clip(image, clip_min, None)
        median = np.nanmedian(image[image != 0]) if np.any(image != 0) else 1.0
        norm = image / (median / median_div)
        return np.arcsinh(norm ** power)
    
    def percentile_normalize(data, pmin=5, pmax=99):
        data = np.nan_to_num(data)
        lo, hi = np.nanpercentile(data[data!=0], [pmin, pmax])
        data = np.clip(data, lo, hi)
        return (data - lo) / (hi - lo + 1e-8)

    blue_p = percentile_normalize(blue_data, intensity_pers[0], intensity_pers[1])
    green_p = percentile_normalize(green_data, intensity_pers[0], intensity_pers[1])
    red_p = percentile_normalize(red_data, intensity_pers[0], intensity_pers[1])

    # Apply stretching
    blue_st = stretch(blue_p, median_div=1, power=1)
    green_st = stretch(green_p, median_div=1, power=1)
    red_st = stretch(red_p, median_div=1, power=1)

    rgb = make_lupton_rgb(red_st, green_st, blue_st, minimum=0, stretch=12, Q=4)


    # Normalize to uint8
    rgb_image = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255 * gamma, 0, 255).astype(np.uint8)
    mask = ~(np.isfinite(blue_data) & np.isfinite(green_data) & np.isfinite(red_data))
    rgb_image[:,:,0][mask] = 255
    rgb_image[:,:,1][mask] = 255
    rgb_image[:,:,2][mask] = 255
    
    # Pixel scale in arcminutes
    pix_scale_arcmin = target_wcs.proj_plane_pixel_scales()[0].value * 60
    
    # Create Galactic WCS for the image
    cdelt_ra, cdelt_dec = target_wcs.wcs.cdelt
    ny, nx, _ = rgb_image.shape
    cx = nx / 2
    cy = ny / 2
    center_icrs = target_wcs.pixel_to_world(cx, cy)
    center_gal = center_icrs.galactic
    gal_wcs = WCS.WCS(naxis=2)
    gal_wcs.wcs.ctype = ['GLON-TAN', 'GLAT-TAN']
    gal_wcs.wcs.crval = [center_gal.l.deg, center_gal.b.deg]
    gal_wcs.wcs.crpix = [cx, cy]
    gal_wcs.wcs.cdelt = [cdelt_ra, cdelt_dec]
    gal_wcs.wcs.cunit = ['deg','deg']
    return rgb_image, target_wcs, gal_wcs, pix_scale_arcmin, red_st, green_st, blue_st


def Cal_Heschel_Infor(bubbleObj, file_names, bubble_com_item_wcs=None, bubble_item=None, data_wcs_item=None, 
                      Cut_Sp=1, reduce_range=0.05, intensity_pers=[10,99.9]):
    """
    Extract Heschel cutouts for a bubble and compute associated metadata.

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
    data_ranges_lb = SpCode.Cal_Data_Range_LB(data_wcs_item, bubble_item)
    delta_l = np.abs(data_ranges_lb[0][0] - data_ranges_lb[0][1]) * Cut_Sp
    delta_b = np.abs(data_ranges_lb[1][0] - data_ranges_lb[1][1]) * Cut_Sp

    img_center = np.around([np.min(data_ranges_lb[0]) + np.abs(data_ranges_lb[0][0] - data_ranges_lb[0][1])/2,\
                            np.min(data_ranges_lb[1]) + np.abs(data_ranges_lb[1][0] - data_ranges_lb[1][1])/2],3)
    
    # Bubble center coordinates
    com_wcs = bubble_com_item_wcs[:2]

    # Find corresponding Spitzer RGB files
    # data_ranges_lb_record_red,data_ranges_lb_record_blue,data_ranges_lb_record_green = data_ranges_lb_record
    file_name_red,file_name_blue,file_name_green = file_names
    
    # file_name_red, data_wcs_red, data_cube_red = Read_Spitzer_Files_I(
    #     com_wcs, data_ranges_lb_record_red, file_names_red, reduce_range=reduce_range
    # )
    # file_name_blue, data_wcs_blue, data_cube_blue = Read_Spitzer_Files_I(
    #     com_wcs, data_ranges_lb_record_blue, file_names_blue, reduce_range=reduce_range
    # )
    # file_name_green, data_wcs_green, data_cube_green = Read_Spitzer_Files_I(
    #     com_wcs, data_ranges_lb_record_green, file_names_green, reduce_range=reduce_range
    # )

    # Process RGB image if all files exist
    if file_name_red is not None and file_name_blue is not None and file_name_green is not None:
        rgb_image, ref_wcs, data_wcs_Sp, pix_scale_arcmin, red_st, green_st, blue_st = Get_RGB_Image_Infor(
            file_name_blue, file_name_green, file_name_red, img_center,
            region_size=(delta_l*u.deg, delta_b*u.deg), intensity_pers=intensity_pers
        )

        lb_item_start, lb_item_end, velocity_range, pixel_scale_Sp = BFTools.Cal_Item_WCS_Range(
            rgb_image.T, data_wcs_Sp
        )

        # Transform bubble and skeleton coordinates to Spitzer pixel frame
        bubble_com_Sp, skeleton_coords_ellipse_Sp, skeleton_com_Sp = SpCode.Add_Bubble_Infor_To_Spitzer(bubbleObj, data_wcs_Sp)

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


