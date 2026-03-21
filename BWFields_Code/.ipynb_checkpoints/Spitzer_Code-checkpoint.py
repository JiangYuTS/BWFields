from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import make_lupton_rgb
from reproject import reproject_interp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle
import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import WCSAxes


def Get_RGB_Image_Infor(blue_file,green_file,red_file,center_wcs,region_size =(0.8*u.deg,0.8*u.deg),intensity_per=1):
    blue_hdu = fits.open(blue_file)[0]
    blue_data = np.squeeze(blue_hdu.data)
    blue_wcs = WCS(blue_hdu.header).celestial
    
    green_hdu = fits.open(green_file)[0]
    green_data = np.squeeze(green_hdu.data)
    green_wcs = WCS(green_hdu.header).celestial
    
    red_hdu = fits.open(red_file)[0]
    red_data = np.squeeze(red_hdu.data)
    red_wcs = WCS(red_hdu.header).celestial
    
    # 使用红色通道作为参考
    ref_wcs = red_wcs
    ref_shape = red_data.shape
    
    if any(d.ndim != 2 for d in [blue_data, green_data, red_data]):
        raise ValueError("数据维度不为2D；请检查FITS文件。")
    
    # 重新投影蓝色和绿色通道到参考WCS
    blue_reproj, _ = reproject_interp((blue_data, blue_wcs), ref_wcs.to_header(), shape_out=ref_shape)
    green_reproj, _ = reproject_interp((green_data, green_wcs), ref_wcs.to_header(), shape_out=ref_shape)
    red_data = red_data
    
    # 处理 NaN 值并转换为 float
    blue_reproj = np.nan_to_num(blue_reproj.astype(np.float32), nan=0.0)
    green_reproj = np.nan_to_num(green_reproj.astype(np.float32), nan=0.0)
    red_data = np.nan_to_num(red_data.astype(np.float32), nan=0.0)
    
    # 打印诊断信息（包含最大值，用于调试强度）
    # print(f"Blue max: {blue_reproj.max()}, shape: {blue_data.shape}, WCS naxis: {blue_wcs.naxis}")
    # print(f"Green max: {green_reproj.max()}, shape: {green_data.shape}, WCS naxis: {green_wcs.naxis}")
    # print(f"Red max: {red_data.max()}, shape: {red_data.shape}, WCS naxis: {ref_wcs.naxis}")
    
    # 定义绘图范围（基于 G38.9-0.4 区域；调整大小以匹配示例图像视场，如0°40'标记）
    center = SkyCoord(l=center_wcs[0]*u.deg, b=center_wcs[1]*u.deg, frame='galactic')  # 中心坐标
      # 视场大小，覆盖0°40'等标记
    
    # 切片到指定范围
    try:
        cutout_blue = Cutout2D(blue_reproj, position=center, size=region_size, wcs=ref_wcs)
        cutout_green = Cutout2D(green_reproj, position=center, size=region_size, wcs=ref_wcs)
        cutout_red = Cutout2D(red_data, position=center, size=region_size, wcs=ref_wcs)
    
        blue_reproj = cutout_blue.data
        green_reproj = cutout_green.data
        red_data = cutout_red.data
        ref_wcs = cutout_red.wcs
    except ValueError as e:
        print(f"切片错误：{e}；使用全图。")
    
    def stretch(image, median_div=1.0, clip_min=0, power=0.5):
        image = np.clip(image, clip_min, None)
        median = np.nanmedian(image[image > 0]) if np.any(image > 0) else 1.0
        norm = image / (median / median_div)
        return np.arcsinh(norm ** power)
    
    def percentile_normalize(data, pmin=5, pmax=95):
        lo, hi = np.nanpercentile(data, [pmin, pmax])
        data = np.clip(data, lo, hi)
        return (data - lo) / (hi - lo + 1e-8)
    
    # 对三通道使用 5–95% 的强度范围
    blue_p = percentile_normalize(blue_reproj, intensity_per, 100-intensity_per)
    green_p = percentile_normalize(green_reproj, intensity_per, 100-intensity_per)
    red_p = percentile_normalize(red_data, intensity_per, 100-intensity_per)
    
    # 应用拉伸（针对强度：降低red的median_div以提升其亮度；power<1压缩动态范围）
    blue_st = stretch(blue_p, median_div=1, power=1) 
    green_st = stretch(green_p, median_div=1, power=1)
    red_st = stretch(red_p, median_div=1, power=1)     
    
    # 生成 RGB（注意：lupton 会自动做亮度压缩）
    rgb = make_lupton_rgb(red_st, green_st, blue_st, minimum=0, stretch=12, Q=4)
    
    # 全局归一化（优化：确保基于强度分布的显示效果一致）
    rgb_image = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255, 0, 255).astype(np.uint8)

    pix_scale_arcmin = ref_wcs.proj_plane_pixel_scales()[0].value * 60

    
    cdelt_ra, cdelt_dec = ref_wcs.wcs.cdelt   
    ny, nx, _ = rgb_image.shape
    cx = nx/2
    cy = ny/2
    center_icrs = ref_wcs.pixel_to_world(cx, cy)
    center_gal = center_icrs.galactic
    gal_wcs = WCS(naxis=2)
    gal_wcs.wcs.ctype = ['GLON-TAN', 'GLAT-TAN']
    gal_wcs.wcs.crval = [center_gal.l.deg, center_gal.b.deg]
    gal_wcs.wcs.crpix = [cx, cy]
    gal_wcs.wcs.cdelt = [cdelt_ra, cdelt_dec]
    gal_wcs.wcs.cunit = ['deg','deg']
    
    return rgb_image,ref_wcs,gal_wcs,pix_scale_arcmin


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





