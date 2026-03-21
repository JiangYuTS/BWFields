import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def resample_vectors_to_target_shape(I_map, Q_map, U_map, target_shape, pitch=2):
    """
    将矢量数据重采样到目标形状，并生成对应的矢量场
    
    Parameters:
    -----------
    I_map, Q_map, U_map : array
        原始偏振数据
    target_shape : tuple
        目标图像的形状 (ny, nx)
    pitch : int
        矢量采样间距
    s
    Returns:
    --------
    x, y, ux, uy : arrays
        重采样后的矢量坐标和方向
    """
    
    # 计算缩放因子
    scale_y = target_shape[0] / I_map.shape[0]
    scale_x = target_shape[1] / I_map.shape[1]
    
    print(f"Original shape: {I_map.shape}")
    print(f"Target shape: {target_shape}")
    print(f"Scale factors: y={scale_y:.3f}, x={scale_x:.3f}")
    
    # 重采样偏振数据到目标形状
    I_resampled = zoom(I_map, (scale_y, scale_x), order=1)
    Q_resampled = zoom(Q_map, (scale_y, scale_x), order=1)
    U_resampled = zoom(U_map, (scale_y, scale_x), order=1)
    
    # 在目标形状上生成矢量
    x, y, ux, uy = vectors_for_target_shape(I_resampled, Q_resampled, U_resampled, 
                                           target_shape, pitch=pitch)
    
    return x, y, ux, uy

def vectors_for_target_shape(I_map, Q_map, U_map, target_shape, pitch=10, normalize=True):
    """
    为指定目标形状生成矢量场
    """
    # 计算矢量幅度
    uu = np.sqrt(Q_map**2 + U_map**2)
    
    # 处理零矢量
    ii = (uu == 0.).nonzero()
    if np.size(ii) > 0:
        uu[ii] = 1.0
    
    if normalize:
        ux = Q_map / uu
        uy = U_map / uu
    else:
        ux = Q_map / np.max(uu)
        uy = U_map / np.max(uu)
    
    ux[ii] = 0.
    uy[ii] = 0.
    
    # 在目标形状范围内创建矢量网格
    ny, nx = target_shape
    X, Y = np.meshgrid(
        np.arange(pitch//2, nx, pitch),  # 从pitch/2开始，确保覆盖整个范围
        np.arange(pitch//2, ny, pitch)
    )
    
    # 确保索引在有效范围内
    Y_safe = np.clip(Y, 0, I_map.shape[0]-1).astype(int)
    X_safe = np.clip(X, 0, I_map.shape[1]-1).astype(int)
    
    ux0 = ux[Y_safe, X_safe]
    uy0 = uy[Y_safe, X_safe]
    
    return X, Y, ux0, uy0

def plot_with_shape_correction(clumps_data, I_map, Q_map, U_map, data_wcs, pitch=2):
    """
    修正形状不匹配问题的绘图函数
    """
    
    # 获取显示数据
    if len(clumps_data.shape) == 3:
        plot_data = clumps_data.sum(0)
    else:
        plot_data = clumps_data
    
    target_shape = plot_data.shape
    
    print(f"Plot data shape: {target_shape}")
    print(f"Polarization data shape: {I_map.shape}")
    
    # 方法1：重采样矢量数据到目标形状
    x, y, ux, uy = resample_vectors_to_target_shape(I_map, Q_map, U_map, 
                                                   target_shape, pitch=pitch)
    
    # 创建图形
    fig = plt.figure(figsize=(12, 5))
    ax0 = fig.add_subplot(111, projection=data_wcs.celestial)
    
    # 设置颜色范围
    vmin = np.min(plot_data[np.where(plot_data != 0)])
    vmax = np.nanpercentile(plot_data[np.where(plot_data != 0)], 99.)
    
    # 绘制背景图像
    gci = ax0.imshow(plot_data,
                     origin='lower',
                     cmap='gray',
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='none')
    
    # 添加白色轮廓
    ax0.contourf(plot_data, levels=[0., 0.01], colors='w')
    
    # 绘制矢量
    arrows = ax0.quiver(x, y, ux, uy,
                       units='width',
                       color='red',
                       pivot='middle',
                       scale=60.,
                       headlength=0,
                       headwidth=1,
                       linewidth=0.3,
                       alpha=0.8)
    
    # 设置格式
    fontsize = 14
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
    
    cbar = plt.colorbar(gci, pad=0)
    cbar.set_label('K km s$^{-1}$')
    plt.tight_layout()
    
    return fig, ax0

def plot_with_wcs_mapping(clumps_data, I_map, Q_map, U_map, data_wcs, 
                         pol_wcs=None, pitch=2):
    """
    使用WCS坐标映射的版本（如果有独立的偏振数据WCS）
    """
    
    if len(clumps_data.shape) == 3:
        plot_data = clumps_data.sum(0)
    else:
        plot_data = clumps_data
    
    # 如果没有独立的偏振WCS，假设它们共享相同的WCS范围
    if pol_wcs is None:
        pol_wcs = data_wcs
    
    # 计算矢量
    x_pol, y_pol, ux, uy = vectors(I_map, Q_map, U_map, pitch=pitch)
    
    # 将偏振数据的像素坐标转换为世界坐标，再转换为目标图像坐标
    if I_map.shape != plot_data.shape:
        # 创建映射关系
        scale_y = plot_data.shape[0] / I_map.shape[0]
        scale_x = plot_data.shape[1] / I_map.shape[1]
        
        # 缩放坐标
        x_mapped = x_pol * scale_x
        y_mapped = y_pol * scale_y
    else:
        x_mapped = x_pol
        y_mapped = y_pol
    
    # 创建图形并绘制
    fig = plt.figure(figsize=(12, 5))
    ax0 = fig.add_subplot(111, projection=data_wcs.celestial)
    
    vmin = np.min(plot_data[np.where(plot_data != 0)])
    vmax = np.nanpercentile(plot_data[np.where(plot_data != 0)], 98.)
    
    gci = ax0.imshow(plot_data,
                     origin='lower',
                     cmap='jet',
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='none')
    
    ax0.contourf(plot_data, levels=[0., 0.01], colors='w')
    
    # 使用映射后的坐标
    arrows = ax0.quiver(x_mapped, y_mapped, ux, uy,
                       units='width',
                       color='red',
                       pivot='middle',
                       scale=40.,
                       headlength=0,
                       headwidth=1,
                       linewidth=0.5,
                       alpha=1)
    
    # 格式设置
    fontsize = 14
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
    
    cbar = plt.colorbar(gci, pad=0)
    cbar.set_label('K km s$^{-1}$')
    plt.tight_layout()
    
    return fig, ax0

# 调试版本：显示坐标信息
def debug_coordinates(clumps_data, I_map, Q_map, U_map, pitch=2):
    """
    调试坐标映射的函数
    """
    if len(clumps_data.shape) == 3:
        plot_data = clumps_data.sum(0)
    else:
        plot_data = clumps_data
    
    print("=== Debug Information ===")
    print(f"Clumps data shape: {clumps_data.shape}")
    print(f"Plot data shape: {plot_data.shape}")
    print(f"Polarization data shape: {I_map.shape}")
    
    # 原始方法的坐标
    x_orig, y_orig, ux_orig, uy_orig = vectors(I_map, Q_map, U_map, pitch=pitch)
    print(f"Original vector coordinates:")
    print(f"  x range: {x_orig.min()} - {x_orig.max()}")
    print(f"  y range: {y_orig.min()} - {y_orig.max()}")
    print(f"  Shape: {x_orig.shape}")
    
    # 修正后的坐标
    x_new, y_new, ux_new, uy_new = resample_vectors_to_target_shape(
        I_map, Q_map, U_map, plot_data.shape, pitch=pitch)
    print(f"Resampled vector coordinates:")
    print(f"  x range: {x_new.min()} - {x_new.max()}")
    print(f"  y range: {y_new.min()} - {y_new.max()}")
    print(f"  Shape: {x_new.shape}")
    
    return x_orig, y_orig, x_new, y_new



def vectors(image, vx, vy, pitch=10, normalize=True):
    """
    修正的vectors函数，确保坐标正确对应
    """
    sz = np.shape(image)
    nx, ny = sz[0], sz[1]
    
    # 计算矢量幅度
    uu = np.sqrt(vx**2 + vy**2)
    
    # 处理零矢量
    ii = (uu == 0.).nonzero()
    if np.size(ii) > 0:
        uu[ii] = 1.0
    
    if normalize:
        ux = vx / uu
        uy = vy / uu
    else:
        ux = vx / np.max(uu)
        uy = vy / np.max(uu)
    
    ux[ii] = 0.
    uy[ii] = 0.
    
    # 关键修正点3：创建正确的坐标网格
    # 使用像素中心坐标
    X, Y = np.meshgrid(
        np.arange(pitch//2, ny-1, pitch),  # 从pitch/2开始，确保在有效范围内
        np.arange(pitch//2, nx-1, pitch)
    )
    
    # 确保索引在有效范围内
    Y_safe = np.clip(Y, 0, nx-1)
    X_safe = np.clip(X, 0, ny-1)
    
    ux0 = ux[Y_safe, X_safe]
    uy0 = uy[Y_safe, X_safe]
    
    return X, Y, ux0, uy0