import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.colors as colors
import astropy.io.fits as fits
import astropy.wcs as WCS
from astropy import units as u
from astropy.table import Table
from skimage import filters,measure,morphology
from scipy.stats import multivariate_normal
from scipy.optimize import least_squares
from scipy import optimize,linalg
import scipy.ndimage as ndimage
import networkx as nx
from collections import defaultdict

from scipy.interpolate import splprep, splev, RegularGridInterpolator
from scipy.special import ellipe 

from tqdm import tqdm
import warnings

import FacetClumps
import DPConCFil
from DPConCFil import Filament_Class_Funs_Analysis as FCFA 


def GNC_FacetClumps(core_data,cores_coordinate):
    xres,yres = core_data.shape
    x_center = cores_coordinate[0]+1
    y_center = cores_coordinate[1]+1
    x_arange = np.arange(max(0,x_center-1),min(xres,x_center+2))
    y_arange = np.arange(max(0,y_center-1),min(yres,y_center+2))
    [x, y] = np.meshgrid(x_arange, y_arange);
    xy = np.column_stack([x.flat, y.flat])
    gradients = core_data[xy[:,0],xy[:,1]]\
                - core_data[x_center,y_center]
    g_step = np.where(gradients == gradients.max())[0][0]
    new_center = list(xy[g_step]-1)
    return gradients,new_center

def Build_MPR_Dict(origin_data,regions):
    k = 1
    reg = -1
    peak_dict = {}
    peak_dict[k] = []
    mountain_dict = {}
    mountain_dict[k] = []
    region_mp_dict = {}
    origin_data = origin_data #+ np.random.random(origin_data.shape) / 100000
    mountain_array = np.zeros_like(origin_data)
    temp_origin_data = np.zeros(tuple(np.array(origin_data.shape)+2))
    temp_origin_data[1:temp_origin_data.shape[0]-1,1:temp_origin_data.shape[1]-1]=origin_data
    for i in range(len(regions)):
        region_mp_dict[i] = []
    for region in regions:
        reg += 1
        coordinates = region.coords
        for i in range(coordinates.shape[0]):
            temp_coords = []
            if mountain_array[coordinates[i][0],coordinates[i][1]] == 0:
                temp_coords.append(coordinates[i].tolist())
                mountain_array[coordinates[i][0],coordinates[i][1]] = k
                gradients,new_center = GNC_FacetClumps(temp_origin_data,coordinates[i])
                if gradients.max() > 0 and mountain_array[new_center[0],new_center[1]] == 0:
                    temp_coords.append(new_center)
                while gradients.max() > 0 and mountain_array[new_center[0],new_center[1]] == 0:
                    mountain_array[new_center[0],new_center[1]] = k
                    gradients,new_center = GNC_FacetClumps(temp_origin_data,new_center)
                    if gradients.max() > 0 and mountain_array[new_center[0],new_center[1]] == 0:
                        temp_coords.append(new_center)
                mountain_array[np.stack(temp_coords)[:,0],np.stack(temp_coords)[:,1]]=\
                    mountain_array[new_center[0],new_center[1]]
                mountain_dict[mountain_array[new_center[0],new_center[1]]] += temp_coords
                if gradients.max() <= 0:
                    peak_dict[k].append(new_center)
                    region_mp_dict[reg].append(k)
                    k += 1
                    mountain_dict[k] = []
                    peak_dict[k] = []
    del(mountain_dict[k])
    del(peak_dict[k])
    return mountain_array,mountain_dict,peak_dict,region_mp_dict
    

def Solve_Ellipse(x, y, center=None):
    """
    Ellipse fitting function
    
    Parameters:
        x, y: Coordinates of points on the ellipse
        center: tuple (x0, y0) Given ellipse center, if None, automatically calculated
    
    Returns:
        paras: Ellipse parameters [a, b, c, d, e, f]
        center_used: Actually used center coordinates
    """
    # Use given center or calculate center
    if center is None:
        x0, y0 = x.mean(), y.mean()
    else:
        x0, y0 = center
    
    # Translate coordinates to center
    D1 = np.array([(x-x0)**2, (x-x0)*(y-y0), (y-y0)**2]).T
    D2 = np.array([x-x0, y-y0, np.ones(y.shape)]).T
    
    S1 = np.dot(D1.T, D1)
    S2 = np.dot(D1.T, D2)
    S3 = np.dot(D2.T, D2)
    
    T = -1 * np.dot(np.linalg.inv(S3), S2.T)
    M = S1 + np.dot(S2, T)
    M = np.array([M[2]/2, -M[1], M[0]/2])
    
    lam, eigen = np.linalg.eig(M)
    cond = 4*eigen[0]*eigen[2] - eigen[1]**2
    A1 = eigen[:, cond > 0]
    
    paras = np.vstack([A1, np.dot(T, A1)]).flatten()
    
    # Convert parameters back to original coordinate system
    A3 = paras[3] - 2*paras[0]*x0 - paras[1]*y0
    A4 = paras[4] - 2*paras[2]*y0 - paras[1]*x0
    A5 = paras[5] + paras[0]*x0**2 + paras[2]*y0**2 + paras[1]*x0*y0 - paras[3]*x0 - paras[4]*y0
    
    paras[3] = A3
    paras[4] = A4
    paras[5] = A5
    
    return paras


def Solve_Ellipse_Infor(paras, fixed_center=None):
    """
    Extract ellipse information from parameters (improved version with numerical stability checks)
    
    Parameters:
        paras: Ellipse parameters [a, b, c, d, e, f]
        fixed_center: tuple (x0, y0) If provided, use this fixed center; otherwise calculate from parameters
    
    Returns:
        ellipse_infor: [x0, y0, angle, a, b]
        coords_fit: [x_coords, y_coords] Ellipse contour coordinates
        ellipse_perimeter: Ellipse perimeter
    """
    paras = paras / paras[5]
    A, B, C, D, E = paras[:5]
    
    # Calculate actual center (for verification)
    denominator = 4*A*C - B**2
    if abs(denominator) < 1e-10:
        x0_calc = 0
        y0_calc = 0
    else:
        x0_calc = (B*E - 2*C*D) / denominator
        y0_calc = (B*D - 2*A*E) / denominator
    
    # Use fixed center or calculated center
    if fixed_center is not None:
        x0, y0 = fixed_center
        # Check deviation between fixed center and calculated center
        dist = np.sqrt((x0 - x0_calc)**2 + (y0 - y0_calc)**2)
        if dist > 10:  # Warning if deviation is too large
            warnings.warn(f"Large deviation between fixed center and fitted center: {dist:.2f} pixels")
    else:
        x0, y0 = x0_calc, y0_calc
    
    # Calculate rotation angle
    alpha_res = 0.5 * np.arctan2(B, A - C)
    
    # Calculate semi-major and semi-minor axes - add numerical stability handling
    # Calculate discriminant
    discriminant = np.sqrt((A-C)**2 + B**2)
    
    # Calculate numerator
    numerator = 2*A*(x0**2) + 2*C*(y0**2) + 2*B*x0*y0 - 2
    
    # Calculate denominator
    denom_a = A + C + discriminant
    denom_b = A + C - discriminant
    
    # Check numerical validity
    if numerator <= 0:
        # Use absolute value
        numerator = abs(numerator)
    
    if denom_a <= 0 or denom_b <= 0:
        denom_a = abs(denom_a) + 1e-10
        denom_b = abs(denom_b) + 1e-10
    
    # Calculate semi-major and semi-minor axes
    a_res = np.sqrt(abs(numerator / denom_a))
    b_res = np.sqrt(abs(numerator / denom_b))
    
    # Ensure semi-major axis is larger than semi-minor axis
    if a_res < b_res:
        a_res, b_res = b_res, a_res
    
    # Calculate ellipse perimeter
    try:
        e_squared = 1 - (b_res**2 / a_res**2)
        e_squared = np.clip(e_squared, 0, 0.99)  # Limit eccentricity range
        E = ellipe(e_squared)
        ellipse_perimeter = 4 * a_res * E
    except:
        ellipse_perimeter = 2 * np.pi * np.sqrt((a_res**2 + b_res**2) / 2)
    
    # Generate ellipse contour points
    num_points = int(ellipse_perimeter)
    theta_res = np.linspace(0.0, 2*np.pi, num_points)

    a_res = np.sqrt(abs(numerator / denom_a))
    b_res = np.sqrt(abs(numerator / denom_b))
    
    x_res = a_res * np.cos(theta_res) * np.cos(alpha_res) \
            - b_res * np.sin(theta_res) * np.sin(alpha_res)
    y_res = b_res * np.sin(theta_res) * np.cos(alpha_res) \
            + a_res * np.cos(theta_res) * np.sin(alpha_res)
    
    coords_fit = [x_res + x0, y_res + y0]
    
    ellipse_angle = -np.around(np.rad2deg(alpha_res), 2)
    ellipse_infor = [x0, y0, ellipse_angle, a_res, b_res]
    
    return ellipse_infor, coords_fit, ellipse_perimeter


# def Get_Ellipse_Coords(bubble_com_bl, bubble_region, bubble_regions_data):
#     """
#     Get ellipse fitting results (improved version)
    
#     Parameters:
#         bubble_com: tuple (x0, y0) Given bubble center coordinates
#         bubble_region: Bubble region object
#         bubble_regions_data: Region data
#         tolerance: Maximum allowed deviation (pixels) if fixed center fitting fails
    
#     Returns:
#         ellipse_infor: Ellipse information [x0, y0, angle, a, b]
#         ellipse_coords: Ellipse contour coordinates
#     """
#     # Get contour coordinates
#     coords_i = bubble_region.coords
#     contour_data = np.zeros((bubble_regions_data.shape[1], bubble_regions_data.shape[2]), 
#                             dtype='uint16')
#     contour_data[coords_i[:, 1], coords_i[:, 2]] = 1
#     contour_data = morphology.dilation(contour_data, morphology.disk(1))
#     contour_data = morphology.erosion(contour_data, morphology.disk(1))
    
#     contour = measure.find_contours(contour_data, 0.5)
#     contour_temp = []
#     for i in range(len(contour)):
#         contour_temp += list(contour[i])
#     contour = [np.array(contour_temp)]
    
#     # First try fitting with fixed center
#     try:
#         paras = Solve_Ellipse(contour[0][:, 0], contour[0][:, 1], center=bubble_com_bl)
#         ellipse_infor, coords_fit, ellipse_perimeter = Solve_Ellipse_Infor(paras, fixed_center=bubble_com_bl)
        
#         # Check if result is valid
#         if np.isnan(ellipse_infor[3]) or np.isnan(ellipse_infor[4]):
#             raise ValueError("Fitting result contains NaN")
            
#     except Exception as e:
#         warnings.warn(f"Fitting with fixed center failed: {str(e)}, trying automatic center calculation")
        
#         # Fallback: don't fix center
#         paras = Solve_Ellipse(contour[0][:, 0], contour[0][:, 1], center=None)
#         ellipse_infor, coords_fit, ellipse_perimeter = Solve_Ellipse_Infor(paras, fixed_center=None)
            
#     ellipse_coords = np.c_[coords_fit[0], coords_fit[1]]
#     delta_sample = 1
#     ellipse_coords_sample = ellipse_coords[::delta_sample]
    
#     return ellipse_infor, ellipse_coords


def Fit_Ellipse_Algebraic(points, center=None):
    """
    Algebraic ellipse fitting with fixed center
    
    Parameters:
        points: Nx2 array of points
        center: tuple (xc, yc) fixed center
    
    Returns:
        a, b, theta, center: semi-major axis, semi-minor axis, rotation angle (radians), center
    """

    if center is None:
        center = np.mean(points,axis=0)
        
    xc, yc = center
    x = points[:, 0] - xc
    y = points[:, 1] - yc
    
    D = np.column_stack([x**2, x*y, y**2])
    params = np.linalg.lstsq(D, np.ones(len(points)), rcond=None)[0]
    
    A, B, C = params
    theta = 0.5 * np.arctan2(B, A - C)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    A_rot = A * cos_t**2 + B * cos_t * sin_t + C * sin_t**2
    C_rot = A * sin_t**2 - B * cos_t * sin_t + C * cos_t**2
    
    a = 1.0 / np.sqrt(A_rot)
    b = 1.0 / np.sqrt(C_rot)
    
    if a < b:
        a, b = b, a
        theta += np.pi/2
    coords_fit,ellipse_infor = Generate_Ellipse_Points(a,b,theta,center)
    return coords_fit,ellipse_infor


def Fit_Ellipse_Geometric(points, center, initial_guess=None):
    """
    Geometric ellipse fitting with fixed center (minimizes geometric distance)
    
    Parameters:
        points: Nx2 array of points
        center: tuple (xc, yc) fixed center
        initial_guess: [a, b, theta] or None
    
    Returns:
        a, b, theta, center: semi-major axis, semi-minor axis, rotation angle (radians), center
    """
    
    if center is None:
        center = np.mean(points,axis=0)
        
    if initial_guess is None:
        x = points[:, 0] - center[0]
        y = points[:, 1] - center[1]
        a = np.sqrt(np.mean(x**2 + y**2)) * 1.2
        initial_guess = [a, a * 0.7, 0.0]
    
    def residuals(params):
        a, b, theta = params
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x = points[:, 0] - center[0]
        y = points[:, 1] - center[1]
        x_rot = cos_t * x + sin_t * y
        y_rot = -sin_t * x + cos_t * y
        return (x_rot/a)**2 + (y_rot/b)**2 - 1
    
    result = least_squares(residuals, initial_guess, method='lm')
    a, b, theta = result.x

    coords_fit,ellipse_infor = Generate_Ellipse_Points(a,b,theta,center)
    return ellipse_infor, coords_fit


def Generate_Ellipse_Points(a_res,b_res,alpha_res,center=None):
    ellipse_perimeter = 2 * np.pi * np.sqrt((a_res**2 + b_res**2) / 2)
    
    # Generate ellipse contour points
    num_points = int(ellipse_perimeter)
    theta_res = np.linspace(0.0, 2*np.pi, num_points)
    
    x_res = a_res * np.cos(theta_res) * np.cos(alpha_res) \
            - b_res * np.sin(theta_res) * np.sin(alpha_res)
    y_res = b_res * np.sin(theta_res) * np.cos(alpha_res) \
            + a_res * np.cos(theta_res) * np.sin(alpha_res)
    
    coords_fit = [x_res + center[0], y_res + center[1]]
    
    ellipse_angle = np.around(np.rad2deg(alpha_res), 2)
    ellipse_infor = [center[0], center[1], ellipse_angle, a_res, b_res]
    return coords_fit,ellipse_infor
    

def Get_Ellipse_Coords(bubble_com_bl, bubble_region, bubble_weight_data):
    """
    Get ellipse fitting results (improved version)
    
    Parameters:
        bubble_com: tuple (x0, y0) Given bubble center coordinates
        bubble_region: Bubble region object
        bubble_regions_data: Region data
        tolerance: Maximum allowed deviation (pixels) if fixed center fitting fails
    
    Returns:
        ellipse_infor: Ellipse information [x0, y0, angle, a, b]
        ellipse_coords: Ellipse contour coordinates
    """
    # Get contour coordinates
    coords_i = bubble_region.coords
    contour_data = np.zeros((bubble_weight_data.shape[1], bubble_weight_data.shape[2]), 
                            dtype='uint16')
    contour_data[coords_i[:, 1], coords_i[:, 2]] = 1
    contour_data = morphology.dilation(contour_data, morphology.disk(1))
    contour_data = morphology.erosion(contour_data, morphology.disk(1))
    
    contour = measure.find_contours(contour_data, 0.5)
    contour_temp = []
    for i in range(len(contour)):
        contour_temp += list(contour[i])
    contour = np.array(contour_temp)

    # _, _, contour, _ = Cal_2D_Region_From_3D_Coords(coords_i,cal_contours=True)

    # First try fitting with fixed center
    try:
        ellipse_infor, coords_fit = Fit_Ellipse_Geometric(contour, center=bubble_com_bl, initial_guess=None)
        
        # Check if result is valid
        if np.isnan(ellipse_infor[3]) or np.isnan(ellipse_infor[4]):
            raise ValueError("Fitting result contains NaN")
            
    except Exception as e:
        warnings.warn(f"Fitting with fixed center failed: {str(e)}, trying automatic center calculation")
        
        # Fallback: don't fix center
        bubble_com_bl = np.mean(contour[0],axis=0)
        ellipse_infor, coords_fit = Fit_Ellipse_Geometric(contour, center=bubble_com_bl, initial_guess=None)
            
    ellipse_coords = np.c_[coords_fit[0], coords_fit[1]]
    delta_sample = 1
    ellipse_coords_sample = ellipse_coords[::delta_sample]
    
    return ellipse_infor, ellipse_coords


def Generate_Circle_Points(center_x, center_y, radius):
    circumference = 2 * np.pi * radius
    num_points = int(round(circumference)) 
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=True)
    x = center_x + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)
    points = np.column_stack((x, y))
    return points, circumference
    

def Get_Bubble_Ellipse_Infor_By_Provide(bubbleObj,bubble_infor_provided):
    bubble_clump_ids = bubble_infor_provided[0]
    bubble_com_pi = bubble_infor_provided[1]
    bubble_com_wcs_pi = bubble_infor_provided[2]
    bubble_radius_p = bubble_infor_provided[3]

    if bubble_com_pi == None:
        clump_centers_p = bubbleObj.clumpsObj.centers[bubble_clump_ids]
        bubble_com_pi = clump_centers_p.mean(axis=0)
    if bubble_com_wcs_pi == None:
        bubble_com_wcs_pi = bubbleObj.clumpsObj.centers_wcs[bubble_clump_ids].mean(axis=0)
    if bubble_radius_p == None:
        bubble_radius_p = np.sqrt(((clump_centers_p - bubble_com_pi)**2)[:,1:].mean(axis=0).sum())
    
    bubble_contour_p, circumference = Generate_Circle_Points(bubble_com_pi[1], bubble_com_pi[2], bubble_radius_p)
    bubble_contour_p += np.random.random(bubble_contour_p.shape)/10**5
    ellipse_infor = [bubble_com_pi[1], bubble_com_pi[2], 0, bubble_radius_p, bubble_radius_p]
    return bubble_com_pi,bubble_com_wcs_pi,bubble_radius_p,bubble_contour_p,ellipse_infor


def Get_Singals(origin_data,Sigma=None,threshold=1):
    kernal_radius=1
    open_data = morphology.opening(origin_data > threshold,morphology.ball(kernal_radius)) 
    dilation_data = morphology.dilation(open_data,morphology.ball(kernal_radius))
    # dilation_data = open_data
    if Sigma is not None:
        filtered = filters.gaussian(origin_data, Sigma)
        filtered[origin_data < threshold] = 0
        dilation_data &= (filtered > threshold) 
    # dilation_label = measure.label(dilation_data,connectivity=2)
    # dilation_regions = measure.regionprops(dilation_label)
    dilation_data[dilation_data>0] = 1
    return dilation_data


def Cal_2D_Region_From_3D_Coords(coords,cal_contours=False):
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()
    v_delta = x_max - x_min + 1
    coords_range = np.array([x_min, x_max, y_min, y_max, z_min, z_max])
    box_data = np.zeros([y_max - y_min + 5, z_max - z_min + 5])
    box_data[coords[:, 1] - y_min + 2, coords[:, 2] - z_min + 2] = 1
    box_label = measure.label(box_data,connectivity=2)
    box_regions = measure.regionprops(box_label)
    box_region_max = box_regions[0]
    if len(box_regions) != 1:
        region_sizes = []
        for region in box_regions:
            region_sizes.append(len(region.coords))
        box_region_max = box_regions[np.argmax(region_sizes)]
    contours_max = np.array([0,0])
    boundary_coords = np.array([0,0])
    if cal_contours:
        contours = measure.find_contours(box_label, level=0.5, fully_connected='high')
        if len(contours)!=0:
            for i in range(0,len(contours)):
                if len(contours_max) < len(contours[i]):
                    contours_max = contours[i]
            if len(contours_max) != 0 :
                contours_max = np.array(contours_max)+np.array([y_min-2,z_min-2])

        bub_mask_erosion_boundary = morphology.binary_erosion(box_data, morphology.disk(1))
        # bub_mask_dilation_boundary = morphology.binary_dilation(box_data, morphology.disk(1))
        contour_data = box_data*~bub_mask_erosion_boundary
        regions = measure.regionprops(morphology.label(contour_data))
        boundary_coords = []
        for region in regions: 
            boundary_coords_i = region.coords 
            boundary_coords += boundary_coords_i.tolist()
        boundary_coords = np.array(boundary_coords)+np.array([y_min-2,z_min-2])
    return coords_range, box_region_max, contours_max, boundary_coords


def Get_Bubble_2D_Regions(label_sum_i):
    label_sum_2 = measure.label(label_sum_i,connectivity=2)  
    regions_lable = measure.regionprops(label_sum_2)
    bubble_regions = []
    bubble_regions_i = []
    euler_data = np.zeros_like(label_sum_i)
    for i in range(len(regions_lable)):
        euler_number = regions_lable[i].euler_number 
        if euler_number != 1:
            euler_coords = regions_lable[i].coords   
            euler_data[euler_coords[:,0],euler_coords[:,1]] = 1
            euler_data_1 = ndimage.binary_fill_holes(euler_data)
            euler_data_2 = euler_data_1 - euler_data
            euler_data_3 = measure.label(euler_data_2,connectivity=2)
            bubble_regions = measure.regionprops(euler_data_3)
            euler_data[euler_coords[:,0],euler_coords[:,1]] = 0
        bubble_regions_i += bubble_regions
    return bubble_regions_i


def Get_Bubble_2D(label_sum):
    bubble_regions_record = []
    label_sum_1 = np.zeros_like(label_sum)
    label_sum_1[label_sum>0] = 1
    label_sum_1_erosion = morphology.erosion(label_sum_1,morphology.disk(1)) 
    label_sum_1_dilation = morphology.dilation(label_sum_1,morphology.disk(1)) 
    for label_sum_i in [label_sum_1,label_sum_1_dilation]:#label_sum_1_erosion
        bubble_regions_i = Get_Bubble_2D_Regions(label_sum_i)
        bubble_regions_record += bubble_regions_i
    return bubble_regions_record
    

def Get_Bubble_Weighted_Data(RMS,Threshold,Sigma,SlicedV,sr_data_i):
    SNR_min = np.int64(Threshold/RMS)
    SNR_max = np.int64(sr_data_i.max()/RMS)
    bubble_repeat_num = np.zeros_like(sr_data_i)
    bubble_weight_data = np.zeros_like(sr_data_i)
    len_v = sr_data_i.shape[0]
    end_id = len_v + 1
    sigma = 9
    for SNR_i in range(SNR_min,SNR_max):
        Threshold_i = SNR_i * RMS 
        singal_mask = Get_Singals(sr_data_i,Sigma,Threshold_i)
        SlicedVS = np.arange(1,SlicedV)
        for delta_v in SlicedVS:
            start_id = np.int64((len_v%delta_v)/2)
            sliced_ids = np.arange(start_id,end_id,delta_v)
            for i in range(len(sliced_ids)-1):
                singal_mask_sum = singal_mask[sliced_ids[i]:sliced_ids[i+1],:,:].sum(0)
                bubble_regions = Get_Bubble_2D(singal_mask_sum)
                for bubble_region in bubble_regions:
                    rb_coords = bubble_region.coords
                    bubble_mor_center = ((sliced_ids[i] + sliced_ids[i+1]) // 2, \
                                         rb_coords[:,0].mean().astype(np.int64), \
                                         rb_coords[:,1].mean().astype(np.int64))
                    z_coords = np.arange(sliced_ids[i], sliced_ids[i+1], dtype=np.int64)
                    r_sq = (z_coords.reshape(len(z_coords), 1) - bubble_mor_center[0])**2 + \
                           (rb_coords[:, 0].reshape(1, len(rb_coords)) - bubble_mor_center[1])** 2 + \
                           (rb_coords[:, 1].reshape(1, len(rb_coords)) - bubble_mor_center[2])**2
                    bubble_weight_data_gauss_local = np.exp(-r_sq / (2 * sigma ** 2)) / 1000
                    bubble_repeat_num[sliced_ids[i]:sliced_ids[i+1],rb_coords[:,0],rb_coords[:,1]] += 1
                    bubble_weight_data[sliced_ids[i]:sliced_ids[i+1],rb_coords[:,0],rb_coords[:,1]] += SNR_i/delta_v
                    bubble_weight_data[sliced_ids[i]:sliced_ids[i+1],rb_coords[:,0], rb_coords[:,1]] += bubble_weight_data_gauss_local
    return bubble_repeat_num,bubble_weight_data


# def Update_Bubble_Weight_Data_Sum_i(bubble_weight_data_i,region_coords,BubbleSize=0):
#     bubble_weight_data_sum = bubble_weight_data_i.sum(0)
#     bubble_weight_sum_mask = np.zeros_like(bubble_weight_data_sum)
#     bubble_weight_sum_mask[bubble_weight_data_sum>0] = 1
#     bubble_weight_data_sum = bubble_weight_data_sum/bubble_weight_data_sum.max()
#     sigma_i = bubble_weight_data_sum[bubble_weight_data_sum>0].std()
#     bubble_weight_data_sum = filters.gaussian(bubble_weight_data_sum,3*sigma_i)
    
#     bubble_weight_sum_mask_label = measure.label(bubble_weight_sum_mask,connectivity=2)
#     bubble_weight_sum_mask_regions = measure.regionprops(bubble_weight_sum_mask_label)
    
#     bubble_weight_data_sum = bubble_weight_data_sum #+ np.random.random(bubble_weight_data_sum.shape) / 1000000
#     _,mountain_dict,_,_ = Build_MPR_Dict(bubble_weight_data_sum,bubble_weight_sum_mask_regions)
#     used_region_coords_record = []
#     for key in mountain_dict.keys():
#         mountain_coords = np.array(mountain_dict[key])
#         if len(mountain_coords) > BubbleSize:
#             matches = (region_coords[:, 1:3, np.newaxis] == mountain_coords.T).all(axis=1).any(axis=1)
#             used_region_coords = region_coords[matches]
#             box_region_max,used_region_coords = Cal_Max_Sub_Region_Coords(used_region_coords)
#             _,box_region,_,_ = Cal_2D_Region_From_3D_Coords(used_region_coords)
#             if box_region.area > BubbleSize :
#                 used_region_coords_record.append(used_region_coords)
#     return used_region_coords_record


# def Update_Bubble_Weight_Data(bubble_repeat_num,bubble_weight_data,BubbleSize,Weight):
#     bubble_weight_data_temp_0 = np.zeros_like(bubble_weight_data)
#     bubble_weight_data_temp_1 = np.zeros_like(bubble_weight_data)
#     bubble_weight_mask = np.zeros_like(bubble_weight_data,dtype='int32')
#     bubble_weight_mask_temp = np.zeros_like(bubble_weight_data,dtype='int32')
#     bubble_weight_mask[bubble_weight_data>0] = 1
#     # bubble_weight_mask = morphology.dilation(bubble_weight_mask,morphology.ball(1)) 
#     bubble_weight_mask_label = measure.label(bubble_weight_mask,connectivity=2)
#     bubble_weight_mask_regions = measure.regionprops(bubble_weight_mask_label)
#     bubbles_coords = []
#     for region in bubble_weight_mask_regions:
#         region_coords = (region.coords[:,0],region.coords[:,1],region.coords[:,2])
#         bubble_weight_data_temp_0[region_coords] = bubble_weight_data[region_coords]
#         bubble_weight_mask_temp[bubble_weight_data_temp_0>Weight] = 1
#         bubble_weight_mask_temp_dilation = morphology.dilation(bubble_weight_mask_temp,morphology.ball(1)) 
#         bubble_weight_mask_temp_opening = morphology.opening(bubble_weight_mask_temp_dilation,morphology.ball(1))
#         bubble_weight_mask_temp = bubble_weight_mask_temp*bubble_weight_mask_temp_opening
#         bubble_weight_data_temp_0[bubble_weight_mask_temp==0] = 0
        
#         if len(bubble_weight_mask_temp[bubble_weight_mask_temp>0]) != 0:
#             bubble_weight_mask_temp_label = measure.label(bubble_weight_mask_temp,connectivity=2)
#             bubble_weight_mask_regions_1 = measure.regionprops(bubble_weight_mask_temp_label)
    
#             region_1_coords = bubble_weight_mask_regions_1[0].coords
#             for i in range(1,len(bubble_weight_mask_regions_1)):
#                 region_1_coords = np.r_[region_1_coords,bubble_weight_mask_regions_1[i].coords]
#             used_region_coords_record = Update_Bubble_Weight_Data_Sum_i(bubble_weight_data_temp_0,region_1_coords,BubbleSize)
#             bubbles_coords += used_region_coords_record
#             for used_coords in used_region_coords_record:
#                 used_coords = (used_coords[:,0],used_coords[:,1],used_coords[:,2])
#                 bubble_weight_data_temp_1[used_coords] = bubble_weight_data[used_coords]
#         bubble_weight_data_temp_0[region_coords] = 0
#         bubble_weight_mask_temp[region_coords] = 0
#     bubble_repeat_num[bubble_weight_data_temp_1==0] = 0
#     return bubble_repeat_num,bubble_weight_data_temp_1,bubbles_coords


def Update_Bubble_Weight_Data(bubble_repeat_num,bubble_weight_data,BubbleSize,Weight):
    bubble_weight_data_temp_0 = np.zeros_like(bubble_weight_data)
    bubble_weight_data_temp_1 = np.zeros_like(bubble_weight_data)
    bubble_weight_mask = np.zeros_like(bubble_weight_data,dtype='int32')
    bubble_weight_mask_temp = np.zeros_like(bubble_weight_data,dtype='int32')
    bubble_weight_mask[bubble_weight_data>0] = 1
    bubble_weight_mask_label = measure.label(bubble_weight_mask,connectivity=2)
    bubble_weight_mask_regions = measure.regionprops(bubble_weight_mask_label)
    bubbles_coords = []
    for region in bubble_weight_mask_regions:
        region_coords = (region.coords[:,0],region.coords[:,1],region.coords[:,2])
        bubble_weight_data_temp_0[region_coords] = bubble_weight_data[region_coords]
        bubble_weight_mask_temp[bubble_weight_data_temp_0>Weight] = 1
        bubble_weight_mask_temp_opening = morphology.opening(bubble_weight_mask_temp,morphology.ball(1))
        bubble_weight_mask_temp_opening = morphology.dilation(bubble_weight_mask_temp_opening,morphology.ball(1)) 
        bubble_weight_mask_temp = bubble_weight_mask_temp*bubble_weight_mask_temp_opening
        bubble_weight_data_temp_0[bubble_weight_mask_temp==0] = 0
        
        if len(bubble_weight_mask_temp[bubble_weight_mask_temp>0]) != 0:
            bubble_weight_mask_temp_label = measure.label(bubble_weight_mask_temp,connectivity=2)
            bubble_weight_mask_regions_1 = measure.regionprops(bubble_weight_mask_temp_label)
    
            region_1_coords = bubble_weight_mask_regions_1[0].coords
            for i in range(0,len(bubble_weight_mask_regions_1)):
                # region_1_coords = np.r_[region_1_coords,bubble_weight_mask_regions_1[i].coords]
            # used_region_coords_record = Update_Bubble_Weight_Data_Sum_i(bubble_weight_data_temp_0,region_1_coords,BubbleSize)
            # bubbles_coords += used_region_coords_record
            # for used_coords in used_region_coords_record:
                used_coords = bubble_weight_mask_regions_1[i].coords
                coords_range,box_region,contours,boundary_coords = Cal_2D_Region_From_3D_Coords(used_coords)
                if box_region.area > BubbleSize :
                    bubbles_coords.append(used_coords)
                    used_coords = (used_coords[:,0],used_coords[:,1],used_coords[:,2])
                    bubble_weight_data_temp_1[used_coords] = bubble_weight_data[used_coords]
        bubble_weight_data_temp_0[region_coords] = 0
        bubble_weight_mask_temp[region_coords] = 0
    bubble_repeat_num[bubble_weight_data_temp_1==0] = 0
    return bubble_repeat_num,bubble_weight_data_temp_1,bubbles_coords


def Bubble_Weight_Data_Detect_By_SR(srs_list,origin_data,parameters):
    RMS = parameters[0]
    Threshold = parameters[1]
    Sigma = parameters[2]
    SlicedV = parameters[3]
    BubbleSize = parameters[4]
    Weight = parameters[5]
    delta_times = []
    bubbles_coords_record = []
    bubble_repeat_num_array = np.zeros_like(origin_data)
    bubble_weight_data_array = np.zeros_like(origin_data)
    regions_array_shape = origin_data.shape
    for index in tqdm(range(len(srs_list))):
        start_1_i = time.time()
        coords_item = srs_list[index].coords
        l_max = np.min([coords_item[:,2].max()+3,regions_array_shape[2]])
        l_min = np.max([0, coords_item[:,2].min()-3])
        b_max = np.min([coords_item[:,1].max()+3,regions_array_shape[1]])
        b_min = np.max([0, coords_item[:,1].min()-3])
        v_max = np.min([coords_item[:,0].max()+3,regions_array_shape[0]])
        v_min = np.max([0, coords_item[:,0].min()-3])
        sr_data_i = np.zeros([v_max-v_min,b_max-b_min,l_max-l_min])
        sr_data_i[coords_item[:,0]-v_min,coords_item[:,1]-b_min,coords_item[:,2]-l_min] = \
                                origin_data[coords_item[:,0],coords_item[:,1],coords_item[:,2]]
        
        bubble_repeat_num,bubble_weight_data = Get_Bubble_Weighted_Data(RMS,Threshold,Sigma,SlicedV,sr_data_i)
        bubble_repeat_num,bubble_weight_data,bubbles_coords = Update_Bubble_Weight_Data(bubble_repeat_num,bubble_weight_data,BubbleSize,Weight)
        
        if len(bubbles_coords) != 0:
            for bubble_coords in bubbles_coords:
                bubble_coords = (bubble_coords[:,0],bubble_coords[:,1],bubble_coords[:,2])
                bubble_repeat_num_array[bubble_coords[0]+v_min,bubble_coords[1]+b_min,bubble_coords[2]+l_min] = \
                                                                                bubble_repeat_num[bubble_coords]
                bubble_weight_data_array[bubble_coords[0]+v_min,bubble_coords[1]+b_min,bubble_coords[2]+l_min] = \
                                                                                bubble_weight_data[bubble_coords]
                bubble_coords_array = np.c_[bubble_coords[0]+v_min,bubble_coords[1]+b_min,bubble_coords[2]+l_min]
                bubbles_coords_record.append(bubble_coords_array)
        end_1_i = time.time()
        delta_time = np.around(end_1_i - start_1_i, 2)
        delta_times.append(delta_time)
    return bubble_repeat_num_array,bubble_weight_data_array,bubbles_coords_record,delta_times


def Get_Bubble_Regions_By_FacetClumps(srs_array,bubble_weight_data,par_FacetClumps_bub):
    SWindow = 3 
    KBins = 35
    FwhmBeam = 2
    VeloRes = 2
    RMS_bub = par_FacetClumps_bub[1]
    Threshold_bub = par_FacetClumps_bub[1]
    SRecursionLBV_bub = par_FacetClumps_bub[2]
    MergeIou = par_FacetClumps_bub[3]

    bubble_weight_data_dilation = morphology.dilation(bubble_weight_data > 0, morphology.ball(1))
    srs_array_bub = measure.label(bubble_weight_data_dilation, connectivity=3)
    srs_list_bub = measure.regionprops(srs_array_bub)

    numbers = []
    bubble_peaks = []
    clump_mask_array = np.zeros_like(bubble_weight_data,dtype='uint32')
    regions_array_shape = bubble_weight_data.shape
    # bubble_weight_data += np.random.random(bubble_weight_data.shape)/10**5
    for index in range(len(srs_list_bub)):
        coords_item = srs_list_bub[index].coords
        l_max = np.min([coords_item[:,2].max()+3,regions_array_shape[2]])
        l_min = np.max([0, coords_item[:,2].min()-3])
        b_max = np.min([coords_item[:,1].max()+3,regions_array_shape[1]])
        b_min = np.max([0, coords_item[:,1].min()-3])
        v_max = np.min([coords_item[:,0].max()+3,regions_array_shape[0]])
        v_min = np.max([0, coords_item[:,0].min()-3])
        signal_region_data_i = np.zeros([v_max-v_min,b_max-b_min,l_max-l_min])
        signal_region_data_i[coords_item[:,0]-v_min,coords_item[:,1]-b_min,coords_item[:,2]-l_min] = \
                            bubble_weight_data[coords_item[:,0],coords_item[:,1],coords_item[:,2]] 
        detect_infor_dict = FacetClumps.FacetClumps_3D_Funs.Detect_FacetClumps(\
                        RMS_bub,Threshold_bub,SWindow, KBins,FwhmBeam,VeloRes,SRecursionLBV_bub,signal_region_data_i)
        if len(detect_infor_dict['peak_location']) != 0:
            number = len(detect_infor_dict['peak_location'])
            regions_data = detect_infor_dict['regions_data']
            regions_list = measure.regionprops(regions_data)
            if len(numbers) == 0:
                for i in range(len(regions_list)):
                    coords = regions_list[i].coords
                    clump_mask_array[coords[:,0]+v_min,coords[:,1]+b_min,coords[:,2]+l_min] = i + 1 
            else:
                for i in range(len(regions_list)):
                    coords = regions_list[i].coords
                    clump_mask_array[coords[:,0]+v_min,coords[:,1]+b_min,coords[:,2]+l_min] = i + 1 + np.sum(numbers)
            bubble_peaks += (np.array(detect_infor_dict['peak_location']) + np.array([v_min,b_min,l_min])).tolist()
            numbers.append(number)
    
    # detect_infor_dict = FacetClumps.FacetClumps_3D_Funs.Detect_FacetClumps(\
    #                 RMS_bub,Threshold_bub,SWindow, KBins,FwhmBeam,VeloRes,SRecursionLBV_bub,bubble_weight_data)
    # bubble_peaks = detect_infor_dict['peak_location']
    # bubble_regions_data = detect_infor_dict['regions_data']
    bubble_regions_data,bubbles_coords_record = Update_Bubble_Regions_Data(bubble_weight_data,srs_array,clump_mask_array,\
                                                                           srs_array_bub,srs_list_bub,bubble_peaks,MergeIou)

    return bubble_regions_data,bubbles_coords_record
    

def Update_Bubble_Regions_Data(bubble_weight_data,srs_array,bubble_regions_data,srs_array_bub,srs_list_bub,bubble_peaks,MergeIou=0.2):
    rc_dict_bub = DPConCFil.Clump_Class_Funs.Build_RC_Dict_Simplified(bubble_peaks, srs_array_bub, srs_list_bub)
    clump_coords_dict_bub = {}
    clumps_list_bub = measure.regionprops(bubble_regions_data)
    for i in range(len(clumps_list_bub)):
        clump_coords_dict_bub[i] = clumps_list_bub[i].coords

    for key in rc_dict_bub.keys():
        if len(rc_dict_bub[key]) > 1:
            for key_i in rc_dict_bub[key]:
                for key_j in rc_dict_bub[key]:
                    if key_i < key_j and key_i in clump_coords_dict_bub.keys() and key_j in clump_coords_dict_bub.keys():
                        coords_bl_i = np.unique(clump_coords_dict_bub[key_i][:,1:], axis=0)
                        coords_bl_j = np.unique(clump_coords_dict_bub[key_j][:,1:], axis=0)
                        match = np.all(coords_bl_i[:, np.newaxis] ==coords_bl_j, axis=2)
                        a_indices = np.where(np.any(match, axis=1))[0]
                        common_rows = np.unique(coords_bl_i[a_indices], axis=0)
                        # region_IOU = len(common_rows)/(len(coords_bl_i) + \
                        #                                len(coords_bl_j) - \
                        #                                len(common_rows) * 2)
                        region_IOU = len(common_rows)/np.min([len(coords_bl_i),len(coords_bl_j)])
                        if region_IOU > MergeIou:
                            bubble_regions_data[clump_coords_dict_bub[key_j][:,0],\
                                                clump_coords_dict_bub[key_j][:,1],\
                                                clump_coords_dict_bub[key_j][:,2]] = key_i + 1
                            
                            clump_coords_dict_bub[key_i] = np.r_[clump_coords_dict_bub[key_i],clump_coords_dict_bub[key_j]]
                            del clump_coords_dict_bub[key_j]
                      
    for key in rc_dict_bub.keys():
        if len(rc_dict_bub[key]) > 1:
            for key_i in [rc_dict_bub[key][0]]:
                for key_j in rc_dict_bub[key]:
                    if key_i in clump_coords_dict_bub.keys() and key_j in clump_coords_dict_bub.keys():
                        bubble_inner_item,start_coords_inner = Get_Bubble_Inner_Item(srs_array,clump_coords_dict_bub[key_j])
                        coords_range, _, contours_max,boundary_coords = Cal_2D_Region_From_3D_Coords(\
                                                                                clump_coords_dict_bub[key_j],cal_contours=True)
    
                        bubble_contour_i = boundary_coords - start_coords_inner[1:]
                        bubble_boundry_values = bubble_inner_item.sum(0)[bubble_contour_i[:,0],bubble_contour_i[:,1]]
    
                        if len(np.where(bubble_boundry_values==0)[0]) > len(bubble_boundry_values)*MergeIou:
                            bubble_regions_data[clump_coords_dict_bub[key_j][:,0],\
                                                clump_coords_dict_bub[key_j][:,1],\
                                                clump_coords_dict_bub[key_j][:,2]] = key_i + 1
    
    bubbles_coords_record = []
    clumps_list_bub = measure.regionprops(bubble_regions_data)

    for i in range(len(clumps_list_bub)):
        clump_coords_bub = clumps_list_bub[i].coords

        # bubble_weight_values = bubble_weight_data[clump_coords_bub[:,0],clump_coords_bub[:,1],clump_coords_bub[:,2]]
        # threshold = filters.threshold_otsu(bubble_weight_values)
        # fliter_logic = np.where(bubble_weight_values < threshold)
        # used_logic = np.where(bubble_weight_values >= 0)

        _,_,max_sub_coords,_ = Cal_Max_Sub_Region_Coords(clump_coords_bub,extend_len=1)
        clump_coords_bub_used = max_sub_coords

        # bubble_regions_data[clump_coords_bub[fliter_logic][:,0],\
        #                     clump_coords_bub[fliter_logic][:,1],\
        #                     clump_coords_bub[fliter_logic][:,2]] = 0
        
        bubble_regions_data[clump_coords_bub_used[:,0],\
                            clump_coords_bub_used[:,1],\
                            clump_coords_bub_used[:,2]] = i + 1
        bubbles_coords_record.append(clump_coords_bub_used)
        
    return bubble_regions_data,bubbles_coords_record


def Bubble_Infor_Morphology(bubble_weight_data,bubbles_coords_record):
    bubble_coms = []
    radius_lb = []
    areas_lb = []
    eccentricities_lb = []
    ranges_v = []
    volume = []
    confidences = []
    contours = []
    ellipses_infor = []
    bubble_infor = {}
    bubble_regions_data = np.zeros_like(bubble_weight_data,dtype='uint32')
    
    coords_l_min = []
    for coords_i in bubbles_coords_record:
        coords_l_min.append(coords_i[:,2].min())
    coords_l_min_argsort = np.argsort(coords_l_min)[::-1]

    k = 1
    for l_min_i in tqdm(coords_l_min_argsort):
        coords_i = bubbles_coords_record[l_min_i]
        coords_range,box_region,contour,boundary_coords = Cal_2D_Region_From_3D_Coords(coords_i,cal_contours=True)
        od_mass = bubble_weight_data[coords_i[:,0],coords_i[:,1],coords_i[:,2]]
        mass_array = np.c_[od_mass,od_mass,od_mass]
        bubble_com = np.around((mass_array*np.c_[coords_i[:,0],coords_i[:,1],coords_i[:,2]]).sum(0)/od_mass.sum(),3).tolist()
        bubble_coms.append(bubble_com)
        radius_lb.append(box_region.equivalent_diameter/2)
        areas_lb.append(box_region.area)
        eccentricities_lb.append(np.around(box_region.eccentricity,2))
        contours.append(contour)
        # delta_v_len = np.int64((coords_i[:,0].max() - coords_i[:,0].min())/2)
        range_v = [coords_i[:,0].min(),coords_i[:,0].max()]
        ranges_v.append(range_v)
        volume.append(len(coords_i))
        confidences.append(np.around(np.mean(bubble_weight_data[coords_i[:,0],coords_i[:,1],coords_i[:,2]]),3))
        bubble_regions_data[coords_i[:,0],coords_i[:,1],coords_i[:,2]] = k
        k += 1
    bubble_regions = measure.regionprops(bubble_regions_data)
    bubble_infor['bubble_coms'] = bubble_coms
    bubble_infor['radius_lb'] = radius_lb
    bubble_infor['areas_lb'] = areas_lb
    bubble_infor['eccentricities_lb'] = eccentricities_lb
    bubble_infor['contours'] = contours
    bubble_infor['ranges_v'] = ranges_v
    bubble_infor['volume'] = volume
    bubble_infor['confidences'] = confidences
    return bubble_infor,bubble_regions_data,bubble_regions


def Bubble_Infor_Morphology_WCS(data_wcs,bubble_infor):
    bubble_vs_wcs = []
    bubble_coms = np.array(bubble_infor['bubble_coms'])
    ranges_v = np.array(bubble_infor['ranges_v'])
    bubble_infor['bubble_coms_wcs'] = []
    bubble_infor['ranges_v_wcs'] = []
    if len(bubble_coms)!=0:
        bubble_coms_wcs = data_wcs.all_pix2world(bubble_coms[:,2],bubble_coms[:,1],bubble_coms[:,0],0)
        bubble_coms_wcs = np.around(np.c_[bubble_coms_wcs[0],bubble_coms_wcs[1],bubble_coms_wcs[2]/1000],3)
        for index in range(len(bubble_coms)):
            bubble_v0_wcs = data_wcs.all_pix2world(bubble_coms[index][2],bubble_coms[index][1],ranges_v[index][0],0)
            bubble_v1_wcs = data_wcs.all_pix2world(bubble_coms[index][2],bubble_coms[index][1],ranges_v[index][1],0)
            bubble_v_wcs = list(np.around([bubble_v0_wcs[2]/1000,bubble_v1_wcs[2]/1000],3))
            bubble_vs_wcs.append(bubble_v_wcs)
        bubble_infor['bubble_coms_wcs'] = bubble_coms_wcs
        bubble_infor['ranges_v_wcs'] = bubble_vs_wcs
    return bubble_infor


def Cal_Max_Sub_Region_Coords(coords,extend_len=1,dilation_r=None):
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()
    coords_range = np.array([x_min, x_max, y_min, y_max, z_min, z_max])
    box_data = np.zeros([x_max - x_min + extend_len*2 + 1, y_max - y_min + extend_len*2 + 1, z_max - z_min + extend_len*2 + 1])
    box_data[coords[:, 0] - x_min + extend_len, coords[:, 1] - y_min + extend_len, coords[:, 2] - z_min + extend_len] = 1
    box_label = measure.label(box_data,connectivity=2)
    box_regions = measure.regionprops(box_label)
    region_sizes = []
    for region in box_regions:
        region_sizes.append(len(region.coords))
    box_region_max = box_regions[np.argmax(region_sizes)]
    max_sub_coords = box_region_max.coords + np.array([x_min - extend_len,y_min - extend_len,z_min - extend_len])
    max_sub_coords_dilation = None
    if dilation_r is not None:
        box_data_dilation = morphology.binary_dilation(box_data, morphology.ball(dilation_r))
        box_label = measure.label(box_data_dilation,connectivity=2)
        box_regions = measure.regionprops(box_label)
        region_sizes = []
        for region in box_regions:
            region_sizes.append(len(region.coords))
        box_region_max = box_regions[np.argmax(region_sizes)]
        max_sub_coords_dilation = box_region_max.coords + np.array([x_min - extend_len,y_min - extend_len,z_min - extend_len])
    return coords_range,box_region_max,max_sub_coords,max_sub_coords_dilation


def Get_Bubble_Clump_Ids(bubble_region,clump_centers,regions_data,connected_ids_dict,bubble_clump_ids=None,dilation_r=1):
    coords_i = np.array([])
    v_extend_logic = False
    if bubble_clump_ids is None:
        coords_i = bubble_region.coords
        _,_,_,coords_i = Cal_Max_Sub_Region_Coords(coords_i,extend_len=10,dilation_r=dilation_r)
        bubble_clump_ids = list(set(regions_data[coords_i[:,0],coords_i[:,1],coords_i[:,2]]))
        if 0 in bubble_clump_ids:
            bubble_clump_ids.remove(0)
        bubble_clump_ids = np.array(bubble_clump_ids) - 1
    
    bubble_clump_ids_con = []
    bubble_clump_ids_con_used = []
    for clump_id in bubble_clump_ids:
        bubble_clump_ids_con += connected_ids_dict[clump_id]
        
    bubble_clump_ids = bubble_clump_ids.tolist()
    bubble_clump_ids_con = list(set(bubble_clump_ids_con))
    for clump_id_con in bubble_clump_ids_con:
        if len(coords_i)>0:
            v_extend_logic = clump_centers[clump_id_con][1] > coords_i[:,1].min() and \
                             clump_centers[clump_id_con][1] < coords_i[:,1].max() and \
                             clump_centers[clump_id_con][2] > coords_i[:,2].min() and \
                             clump_centers[clump_id_con][2] < coords_i[:,2].max()  
        if clump_id_con not in bubble_clump_ids and v_extend_logic:
            bubble_clump_ids.append(clump_id_con)
        elif clump_id_con not in bubble_clump_ids:
            bubble_clump_ids_con_used.append(clump_id_con)
            
    bubble_clump_ids = np.array(bubble_clump_ids)
    bubble_clump_ids_con = np.array(bubble_clump_ids_con_used)
    return bubble_clump_ids,bubble_clump_ids_con


def Get_Bubble_Gas_Infor(bubbleObj,add_con,systemic_v_type):
    origin_data = bubbleObj.clumpsObj.origin_data
    regions_data = bubbleObj.clumpsObj.regions_data
    data_wcs = bubbleObj.clumpsObj.data_wcs
    clump_coords_dict = bubbleObj.clumpsObj.clump_coords_dict
    bubble_clump_ids = bubbleObj.bubble_clump_ids
    bubble_clump_ids_con = bubbleObj.bubble_clump_ids_con
    bubble_gas_com_1 = np.around(bubbleObj.clumpsObj.centers[bubble_clump_ids].mean(axis=0),3)
    bubble_gas_com_12 = np.around(bubbleObj.clumpsObj.centers[bubble_clump_ids_con].mean(axis=0),3)
    bubble_gas_com_2 = np.around((bubble_gas_com_1 + bubble_gas_com_12)/2,3)
    bubble_gas_com_wcs_1 = np.around(bubbleObj.clumpsObj.centers_wcs[bubble_clump_ids].mean(axis=0),3)
    bubble_gas_com_wcs_12 = np.around(bubbleObj.clumpsObj.centers_wcs[bubble_clump_ids_con].mean(axis=0),3)
    bubble_gas_com_wcs_2 = np.around((bubble_gas_com_wcs_1 + bubble_gas_com_wcs_12)/2,3)
    
    if systemic_v_type == 1:
        systemic_v = bubble_gas_com_wcs_2[2]
    elif systemic_v_type == 2:
        systemic_v = bubble_coms_wcs[index][2]

    clump_coords_dict = bubbleObj.clumpsObj.clump_coords_dict
    bubble_gas_coords_1 = clump_coords_dict[bubble_clump_ids[0]]
    bubble_gas_coords_2 = clump_coords_dict[bubble_clump_ids[0]]


    for bubble_clump_id in np.r_[bubble_clump_ids[1:]]:
        bubble_gas_coords_1 = np.r_[bubble_gas_coords_2,clump_coords_dict[bubble_clump_id]]

    for bubble_clump_id in np.r_[bubble_clump_ids[1:],bubble_clump_ids_con]:
        bubble_gas_coords_2 = np.r_[bubble_gas_coords_1,clump_coords_dict[bubble_clump_id]]

    bubble_gas_ranges_lbv_mins = []
    bubble_gas_ranges_lbv_maxs = []
    for bubble_gas_coords in [bubble_gas_coords_1,bubble_gas_coords_2]:
        if data_wcs.naxis == 3:
            bubble_ranges_lbv_min = data_wcs.all_pix2world(bubble_gas_coords[:,2].min(),\
                                                           bubble_gas_coords[:,1].min(),\
                                                           bubble_gas_coords[:,0].min(),0)
            bubble_ranges_lbv_max = data_wcs.all_pix2world(bubble_gas_coords[:,2].max(),\
                                                           bubble_gas_coords[:,1].max(),\
                                                           bubble_gas_coords[:,0].max(),0)
        elif data_wcs.naxis == 4:
            bubble_ranges_lbv_min = data_wcs.all_pix2world(bubble_gas_coords[:,2].min(),\
                                                           bubble_gas_coords[:,1].min(),\
                                                           bubble_gas_coords[:,0].min(),0,0)
            bubble_ranges_lbv_max = data_wcs.all_pix2world(bubble_gas_coords[:,2].max(),\
                                                           bubble_gas_coords[:,1].max(),\
                                                           bubble_gas_coords[:,0].max(),0,0)
        
        bubble_gas_ranges_lbv_mins.append(np.around([bubble_ranges_lbv_min[0],bubble_ranges_lbv_min[1],bubble_ranges_lbv_min[2]/1000],3))
        bubble_gas_ranges_lbv_maxs.append(np.around([bubble_ranges_lbv_max[0],bubble_ranges_lbv_max[1],bubble_ranges_lbv_max[2]/1000],3))

    clump_ids = bubble_clump_ids
    bubble_gas_coords, bubble_gas_item, data_wcs_gas_item, regions_data_T, start_coords, clumps_item_mask_2D, lb_area = \
                                        FCFA.Filament_Coords(origin_data, regions_data, data_wcs, clump_coords_dict, clump_ids)
    bubbleObj.bubble_gas_item_1 = bubble_gas_item
    bubbleObj.data_wcs_gas_item_1 = data_wcs_gas_item
    bubbleObj.start_coords_gas_item_1 = start_coords
    
    clump_ids = np.r_[bubble_clump_ids,bubble_clump_ids_con]
    bubble_gas_coords, bubble_gas_item, data_wcs_gas_item, regions_data_T, start_coords, clumps_item_mask_2D, lb_area = \
                                        FCFA.Filament_Coords(origin_data, regions_data, data_wcs, clump_coords_dict, clump_ids)
    bubbleObj.bubble_gas_item_2 = bubble_gas_item
    bubbleObj.data_wcs_gas_item_2 = data_wcs_gas_item
    bubbleObj.start_coords_gas_item_2 = start_coords
    
    bubbleObj.bubble_gas_coords = bubble_gas_coords
    bubbleObj.bubble_gas_com_1 = bubble_gas_com_1
    bubbleObj.bubble_gas_com_12 = bubble_gas_com_12
    bubbleObj.bubble_gas_com_2 = bubble_gas_com_2
    bubbleObj.bubble_gas_com_wcs_1 = bubble_gas_com_wcs_1
    bubbleObj.bubble_gas_com_wcs_12 = bubble_gas_com_wcs_12
    bubbleObj.bubble_gas_com_wcs_2 = bubble_gas_com_wcs_2
    bubbleObj.systemic_v = systemic_v
    
    bubbleObj.bubble_gas_ranges_lbv_min_1 = bubble_gas_ranges_lbv_mins[0]
    bubbleObj.bubble_gas_ranges_lbv_max_1 = bubble_gas_ranges_lbv_maxs[0]
    bubbleObj.bubble_gas_ranges_lbv_min_2 = bubble_gas_ranges_lbv_mins[1]
    bubbleObj.bubble_gas_ranges_lbv_max_2 = bubble_gas_ranges_lbv_maxs[1]


def Cal_Item_WCS_Range(data_item,data_wcs_item):
    data_item_shape = data_item.shape
        
    # Define the bounds in pixel coordinates
    l_min, l_max = 0, data_item_shape[2] - 1  # Longitude range
    b_min, b_max = 0, data_item_shape[1] - 1  # Latitude range
    v_min, v_max = 0, data_item_shape[0] - 1  # Velocity range
    
    # Convert pixel coordinates to world coordinates
    if data_wcs_item.naxis == 4:
        # For 4D WCS (includes a time or stokes dimension)
        lbv_start = data_wcs_item.all_pix2world(l_min, b_min, v_min, 0, 0)
        lbv_end = data_wcs_item.all_pix2world(l_max, b_max, v_max, 0, 0)
    elif data_wcs_item.naxis == 3:
        # For 3D WCS (just l, b, v)
        lbv_start = data_wcs_item.all_pix2world(l_min, b_min, v_min, 0)
        lbv_end = data_wcs_item.all_pix2world(l_max, b_max, v_max, 0)
    
    # Format coordinates for readability (convert v from m/s to km/s)
    lbv_item_start = [np.around(lbv_start[0], 2), np.around(lbv_start[1], 2), np.around(lbv_start[2] / 1000, 2)]
    lbv_item_end = [np.around(lbv_end[0], 2), np.around(lbv_end[1], 2), np.around(lbv_end[2] / 1000, 2)]
    
    # Create a linear scale for the velocity dimension
    velocity_range = np.linspace(lbv_item_start[2], lbv_item_end[2], data_item_shape[0])

    if data_wcs_item.has_celestial:
        pixel_scale = np.abs(data_wcs_item.wcs.cdelt[0])
    else:
        pixel_scale = 0.0083333333333333
    return lbv_item_start,lbv_item_end,velocity_range,pixel_scale
    

def Generate_Neighbor_coords(center, range_x=3, range_y=3, step=1):
    half_x = (range_x - 1) // 2
    half_y = (range_y - 1) // 2
    
    x_offsets = [dx * step for dx in range(-half_x, half_x + 1)]
    y_offsets = [dy * step for dy in range(-half_y, half_y + 1)]
    
    neighbor_coords = [
        (center[0] + dx, center[1] + dy)
        for dx in x_offsets
        for dy in y_offsets]
    return np.array(neighbor_coords)
    

def Resort_Ellipse_Coords(bubbleObj):
    bubble_gas_item_1 = bubbleObj.bubble_gas_item_1 
    start_coords_gas_item_1 = bubbleObj.start_coords_gas_item_1
    ellipse_coords_gas_1 = np.int64(np.around(bubbleObj.ellipse_coords - start_coords_gas_item_1[1:]))

    ellipse_values = []
    for ellipse_coord_gas_1 in ellipse_coords_gas_1:
        neighbor_coords = Generate_Neighbor_coords(np.int64(np.around(ellipse_coord_gas_1)))
        ellipse_value = bubble_gas_item_1.sum(0)[neighbor_coords[:,0],neighbor_coords[:,1]].sum()
        ellipse_values.append(ellipse_value)
        
    ellipse_coords_order = np.argsort(ellipse_values)
    ellipse_coords_updated = np.r_[bubbleObj.ellipse_coords[ellipse_coords_order[-1]:],\
                                   bubbleObj.ellipse_coords[:ellipse_coords_order[-1]+1]]
    bubbleObj.ellipse_coords = ellipse_coords_updated 


def Get_Bubble_Inner_Item(bubble_regions_data,coords):
    # coords = region.coords
    x_min = coords[:,0].min()
    x_max = coords[:,0].max()
    y_min = coords[:,1].min()
    y_max = coords[:,1].max()
    z_min = coords[:,2].min()
    z_max = coords[:,2].max()
    length = np.max([x_max-x_min,y_max-y_min,z_max-z_min])+5
    bubble_inner_item =  np.zeros([length,length,length])
    start_x = np.int64((length - (x_max-x_min))/2)
    start_y = np.int64((length - (y_max-y_min))/2)
    start_z = np.int64((length - (z_max-z_min))/2)
    bubble_inner_item[coords[:,0]-x_min+start_x,coords[:,1]-y_min+start_y,coords[:,2]-z_min+start_z] = \
                        bubble_regions_data[coords[:,0],coords[:,1],coords[:,2]]
    start_coords_inner = [x_min-start_x,y_min-start_y,z_min-start_z]
    return bubble_inner_item,start_coords_inner


def Cal_Contours_IOU(contour, ellipse_coords):
    """
    Calculate the contour overlap rate between original contour and fitted ellipse
    using pure NumPy and Matplotlib Path
    
    Parameters:
    -----------
    contour : numpy.ndarray
        Original contour coordinates, shape (N, 2)
    ellipse_coords : numpy.ndarray
        Ellipse coordinates, shape (M, 2)
    
    Returns:
    --------
    overlap_rate : float
        Overlap rate between 0 and 1
    visualization : numpy.ndarray
        Visualization of contours and overlap
    """
    # Determine the plot boundaries
    all_coords = np.vstack([contour, ellipse_coords])
    x_min, y_min = np.floor(all_coords.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_coords.max(axis=0)).astype(int)
    
    # Create a grid of points
    x = np.linspace(x_min, x_max, x_max - x_min + 1)
    y = np.linspace(y_min, y_max, y_max - y_min + 1)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Create paths for contour and ellipse
    contour_path = Path(contour)
    ellipse_path = Path(ellipse_coords)
    
    # Find points inside each path
    contour_mask = contour_path.contains_points(grid_points)
    ellipse_mask = ellipse_path.contains_points(grid_points)
    
    # Calculate areas and overlap
    contour_area = np.sum(contour_mask)
    ellipse_area = np.sum(ellipse_mask)
    intersection_area = np.sum(contour_mask & ellipse_mask)
    
    # Calculate overlap rate
    contour_ellipse_IOU = intersection_area / contour_area if contour_area > 0 else 0
    contour_ellipse_IOU = np.around(contour_ellipse_IOU,2)
    # Visualization
    visualization = np.zeros((len(y), len(x), 3), dtype=np.uint8)
    
    # Color the regions
    contour_grid = contour_mask.reshape(xx.shape)
    ellipse_grid = ellipse_mask.reshape(xx.shape)
    intersection_grid = (contour_mask & ellipse_mask).reshape(xx.shape)
    
    visualization[:,:,0] = (contour_grid * 255).astype(np.uint8)  # Red for contour
    visualization[:,:,1] = (ellipse_grid * 255).astype(np.uint8)  # Green for ellipse
    visualization[:,:,2] = (intersection_grid * 255).astype(np.uint8)  # Blue for intersection
    
    return contour_ellipse_IOU, visualization,contour_grid,ellipse_grid





