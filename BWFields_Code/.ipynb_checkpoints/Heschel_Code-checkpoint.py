import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from astropy.coordinates import SkyCoord
import astropy.units as u

from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs



def get_data_wcs(fname):
    """
    使用 fits.getdata / getheader
    自动跳过 PrimaryHDU（data=None 的坑）
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
                "在这个 FITS 里没有找到包含天球WCS的图像HDU（has_celestial=False）。\n"
                "请用下面的“诊断代码”把每个 ext 的 CTYPE 打印出来，我再帮你定位。"
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
    不用 PercentileInterval，直接用 np.nanpercentile（避免你遇到的类型问题）
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


