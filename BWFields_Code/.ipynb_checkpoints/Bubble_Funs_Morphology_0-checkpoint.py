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
from scipy import optimize,linalg
import scipy.ndimage as ndimage
import networkx as nx
from collections import defaultdict

from scipy.interpolate import splprep, splev, RegularGridInterpolator
from scipy.special import ellipe 

from tqdm import tqdm


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
    origin_data = origin_data + np.random.random(origin_data.shape) / 100000
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

def Solve_Ellipse(x,y):
    #a*x**2 + b*x*y + c*y**2 + d*x + e*y + f
    x0,y0 = x.mean(),y.mean()      
    D1=np.array([(x-x0)**2,(x-x0)*(y-y0),(y-y0)**2]).T
    D2=np.array([x-x0,y-y0,np.ones(y.shape)]).T
    S1=np.dot(D1.T,D1)
    S2=np.dot(D1.T,D2)
    S3=np.dot(D2.T,D2)    
    T=-1*np.dot(np.linalg.inv(S3),S2.T)
    M=S1+np.dot(S2,T)
    M=np.array([M[2]/2,-M[1],M[0]/2])
    lam,eigen=np.linalg.eig(M)
    cond=4*eigen[0]*eigen[2]-eigen[1]**2
    A1=eigen[:,cond>0] 
    paras=np.vstack([A1,np.dot(T,A1)]).flatten()
    A3=paras[3]-2*paras[0]*x0-paras[1]*y0
    A4=paras[4]-2*paras[2]*y0-paras[1]*x0
    A5=paras[5]+paras[0]*x0**2+paras[2]*y0**2+paras[1]*x0*y0-paras[3]*x0-paras[4]*y0
    paras[3]=A3;paras[4]=A4;paras[5]=A5
    return paras


def Solve_Ellipse_Infor(paras):
    paras=paras/paras[5]
    A,B,C,D,E=paras[:5]
    x0=(B*E-2*C*D)/(4*A*C-B**2)
    y0=(B*D-2*A*E)/(4*A*C-B**2)
    alpha_res = 0.5 * np.arctan2(B, A - C) 
    a_res= np.sqrt((2*A*(x0**2)+2*C*(y0**2)+2*B*x0*y0-2)/(A+C+np.sqrt(((A-C)**2+B**2))))
    b_res= np.sqrt((2*A*(x0**2)+2*C*(y0**2)+2*B*x0*y0-2)/(A+C-np.sqrt(((A-C)**2+B**2))))
    
    if a_res < b_res:  
        a_res, b_res = b_res, a_res
    e_squared = 1 - (b_res**2 / a_res**2) 
    E = ellipe(e_squared) 
    ellipse_perimeter = 4 * a_res * E
    theta_res = np.linspace(0.0, 6.28, np.int32(np.around(ellipse_perimeter)))

    a_res= np.sqrt((2*A*(x0**2)+2*C*(y0**2)+2*B*x0*y0-2)/(A+C+np.sqrt(((A-C)**2+B**2))))
    b_res= np.sqrt((2*A*(x0**2)+2*C*(y0**2)+2*B*x0*y0-2)/(A+C-np.sqrt(((A-C)**2+B**2))))
    x_res = a_res * np.cos(theta_res) * np.cos(alpha_res) \
            - b_res * np.sin(theta_res) * np.sin(alpha_res)
    y_res = b_res * np.sin(theta_res) * np.cos(alpha_res) \
            + a_res * np.cos(theta_res) * np.sin(alpha_res)
    coords_fit = [x_res+x0,y_res+y0]
    ellipse_angle = -np.around(np.rad2deg(alpha_res),2)
    ellipse_infor = [x0,y0,ellipse_angle,a_res,b_res]
    return ellipse_infor,coords_fit,ellipse_perimeter
    

def Get_Ellipse_Coords(bubble_region,bubble_regions_data):
    coords_i = bubble_region.coords
    contour_data =  np.zeros((bubble_regions_data.shape[1],bubble_regions_data.shape[2]),dtype='uint16')
    contour_data[coords_i[:,1],coords_i[:,2]] = 1   
    contour_data = morphology.dilation(contour_data,morphology.disk(1))
    contour_data = morphology.erosion(contour_data,morphology.disk(1))
    contour = measure.find_contours(contour_data,0.5)
    contour_temp = []
    for i in range(0,len(contour)):
        contour_temp += list(contour[i])
    contour = [np.array(contour_temp)]
    paras = Solve_Ellipse(contour[0][:,0],contour[0][:,1])
    ellipse_infor,coords_fit,ellipse_perimeter = Solve_Ellipse_Infor(paras)
    ellipse_coords = np.c_[coords_fit[0],coords_fit[1]]
    ellipse_x0,ellipse_y0,ellipse_angle,a_res,b_res = ellipse_infor
    
    delta_sample = 1 # np.int32(len(ellipse_coords)/ellipse_perimeter)
    ellipse_coords_sample = ellipse_coords[::delta_sample]
    # ellipse_coords_sample_num = np.int32(np.around(len(ellipse_coords_sample)/2))
    # ellipse_coords_sample = ellipse_coords_sample[:ellipse_coords_sample_num+1]

    return ellipse_infor,ellipse_coords,ellipse_coords_sample





def Get_Singals(origin_data,Sigma=None,threshold='otsu'):
    kernal_radius=1
    open_data = morphology.opening(origin_data > threshold,morphology.ball(kernal_radius)) 
    dilation_data = morphology.dilation(open_data,morphology.ball(kernal_radius))
    # dilation_data = open_data
    if Sigma!=None:
        filter_data = origin_data.copy()
        filter_data[origin_data < threshold] = 0
        filter_data = filters.gaussian(filter_data,Sigma)
        dilation_data = dilation_data*(filter_data > threshold)
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
    box_data = np.zeros([y_max - y_min + 3, z_max - z_min + 3])
    box_data[coords[:, 1] - y_min + 1, coords[:, 2] - z_min + 1] = 1
    box_label = measure.label(box_data,connectivity=2)
    box_regions = measure.regionprops(box_label)
    box_region_max = box_regions[0]
    if len(box_regions) != 1:
        region_sizes = []
        for region in box_regions:
            region_sizes.append(len(region.coords))
        box_region_max = box_regions[np.argmax(region_sizes)]
    contours = None
    if cal_contours:
        contours = measure.find_contours(box_label, level=0.5, fully_connected='high')
        if len(contours)!=0:
            contours_temp = []
            for i in range(0,len(contours)):
                contours_temp += list(contours[i])
            if len(contours_temp) != 0 :
                contours = np.array(contours_temp)+np.array([y_min-1,z_min-1])
    return coords_range, box_region_max, contours


def Cal_Max_Sub_Region_Coords(coords):
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()
    coords_range = np.array([x_min, x_max, y_min, y_max, z_min, z_max])
    box_data = np.zeros([x_max - x_min + 3, y_max - y_min + 3, z_max - z_min + 3])
    box_data[coords[:, 0] - x_min + 1, coords[:, 1] - y_min + 1, coords[:, 2] - z_min + 1] = 1
    box_label = measure.label(box_data,connectivity=2)
    box_regions = measure.regionprops(box_label)
    region_sizes = []
    for region in box_regions:
        region_sizes.append(len(region.coords))
    box_region_max = box_regions[np.argmax(region_sizes)]
    max_sub_coords = box_region_max.coords + np.array([x_min - 1,y_min - 1,z_min - 1])
    return box_region_max,max_sub_coords


def Get_Bubble_2D(label_sum):
    center_lb = []
    radius_lb = []
    areas_lb = []
    bubble_regions = []
    eccentricity = []
    bubble_data = np.zeros_like(label_sum)
    label_sum_1 = np.zeros_like(label_sum)
    label_sum_1[np.where(label_sum!=0)] = 1
    label_sum_2 = measure.label(label_sum_1,connectivity=2)  
    regions_lable = measure.regionprops(label_sum_2)
    for i in range(len(regions_lable)):
        euler_number = regions_lable[i].euler_number 
        if euler_number != 1:
            euler_coords = regions_lable[i].coords   
            euler_data = np.zeros_like(label_sum)
            euler_data[euler_coords[:,0],euler_coords[:,1]] = 1
            euler_data_1 = ndimage.binary_fill_holes(euler_data)
            euler_data_2 = euler_data_1 - euler_data
            euler_data_3 = measure.label(euler_data_2,connectivity=2)
            bubble_regions = measure.regionprops(euler_data_3)
    return bubble_regions
    

def Get_Bubble_Weighted_Data(RMS,Threshold,Sigma,SlicedV,BubbleSize,sr_data_i):
    SNR_min = np.int64(Threshold/RMS)
    SNR_max = np.int64(sr_data_i.max()/RMS)
    bubble_repeat_num = np.zeros_like(sr_data_i)
    bubble_weight_data = np.zeros_like(sr_data_i)
    len_v = sr_data_i.shape[0]
    end_id = len_v + 1
    for SNR_i in range(SNR_min,SNR_max):
        Threshold_i = SNR_i * RMS 
        singal_mask = Get_Singals(sr_data_i,Sigma,Threshold_i)
        SlicedVS = np.arange(1,SlicedV)
        for delta_v in SlicedVS:
            start_id = np.int64((len_v%delta_v)/2)
            sliced_ids = np.arange(start_id,end_id,delta_v)
            for i in range(len(sliced_ids)-1):
                singal_mask_sum = singal_mask[sliced_ids[i]:sliced_ids[i+1],:,:].sum(0)
                # ax0 = Plot_Images_2D(singal_data_sum,title='singal_data_sum')
                # plt.show()
                bubble_regions = Get_Bubble_2D(singal_mask_sum)
                for bubble_region in bubble_regions:
                    rb_coords = bubble_region.coords
                    bubble_repeat_num[sliced_ids[i]:sliced_ids[i+1],rb_coords[:,0],rb_coords[:,1]] += 1
                    bubble_weight_data[sliced_ids[i]:sliced_ids[i+1],rb_coords[:,0],rb_coords[:,1]] += SNR_i/delta_v
    return bubble_repeat_num,bubble_weight_data


def Update_Bubble_Weight_Data_Sum_i(bubble_weight_data_i,region_coords,BubbleSize=0):
    bubble_weight_data_sum = bubble_weight_data_i.sum(0)
    bubble_weight_sum_mask = np.zeros_like(bubble_weight_data_sum)
    bubble_weight_sum_mask[bubble_weight_data_sum>0] = 1
    bubble_weight_data_sum = bubble_weight_data_sum/bubble_weight_data_sum.max()
    sigma_i = bubble_weight_data_sum[bubble_weight_data_sum>0].std()
    bubble_weight_data_sum = filters.gaussian(bubble_weight_data_sum,3*sigma_i)
    
    bubble_weight_sum_mask_label = measure.label(bubble_weight_sum_mask,connectivity=2)
    bubble_weight_sum_mask_regions = measure.regionprops(bubble_weight_sum_mask_label)
    
    bubble_weight_data_sum = bubble_weight_data_sum + np.random.random(bubble_weight_data_sum.shape) / 1000000
    mountain_array,mountain_dict,peak_dict,region_mp_dict = Build_MPR_Dict(bubble_weight_data_sum,bubble_weight_sum_mask_regions)
    used_region_coords_record = []
    for key in mountain_dict.keys():
        mountain_coords = np.array(mountain_dict[key])
        if len(mountain_coords) > BubbleSize:
            matches = (region_coords[:, 1:3, np.newaxis] == mountain_coords.T).all(axis=1).any(axis=1)
            used_region_coords = region_coords[matches]
            box_region_max,used_region_coords = Cal_Max_Sub_Region_Coords(used_region_coords)
            coords_range,box_region,contours = Cal_2D_Region_From_3D_Coords(used_region_coords)
            if box_region.area > BubbleSize :
                used_region_coords_record.append(used_region_coords)
    return used_region_coords_record


def Update_Bubble_Weight_Data(bubble_repeat_num,bubble_weight_data,BubbleSize,Weight):
    bubble_weight_data_temp_0 = np.zeros_like(bubble_weight_data)
    bubble_weight_data_temp_1 = np.zeros_like(bubble_weight_data)
    bubble_weight_mask = np.zeros_like(bubble_weight_data,dtype='int32')
    bubble_weight_mask_temp = np.zeros_like(bubble_weight_data,dtype='int32')
    bubble_weight_mask[bubble_weight_data>0] = 1
    # bubble_weight_mask = morphology.dilation(bubble_weight_mask,morphology.ball(1)) 
    bubble_weight_mask_label = measure.label(bubble_weight_mask,connectivity=2)
    bubble_weight_mask_regions = measure.regionprops(bubble_weight_mask_label)
    bubbles_coords = []
    for region in bubble_weight_mask_regions:
        region_coords = (region.coords[:,0],region.coords[:,1],region.coords[:,2])
        bubble_weight_data_temp_0[region_coords] = bubble_weight_data[region_coords]
        bubble_weight_mask_temp[bubble_weight_data_temp_0>Weight] = 1
        bubble_weight_mask_temp_dilation = morphology.dilation(bubble_weight_mask_temp,morphology.ball(1)) 
        bubble_weight_mask_temp_opening = morphology.opening(bubble_weight_mask_temp_dilation,morphology.ball(1))
        bubble_weight_mask_temp = bubble_weight_mask_temp*bubble_weight_mask_temp_opening
        bubble_weight_data_temp_0[bubble_weight_mask_temp==0] = 0
        
        if len(bubble_weight_mask_temp[bubble_weight_mask_temp>0]) != 0:
            bubble_weight_mask_temp_label = measure.label(bubble_weight_mask_temp,connectivity=2)
            bubble_weight_mask_regions_1 = measure.regionprops(bubble_weight_mask_temp_label)
    
            region_1_coords = bubble_weight_mask_regions_1[0].coords
            for i in range(1,len(bubble_weight_mask_regions_1)):
                region_1_coords = np.r_[region_1_coords,bubble_weight_mask_regions_1[i].coords]
            used_region_coords_record = Update_Bubble_Weight_Data_Sum_i(bubble_weight_data_temp_0,region_1_coords,BubbleSize)
            bubbles_coords += used_region_coords_record
            for used_coords in used_region_coords_record:
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
        
        bubble_repeat_num,bubble_weight_data = Get_Bubble_Weighted_Data(RMS,Threshold,Sigma,SlicedV,BubbleSize,sr_data_i)
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
    k = 1
    for coords_i in tqdm(bubbles_coords_record):
        coords_range,box_region,contour = Cal_2D_Region_From_3D_Coords(coords_i,cal_contours=True)
        od_mass = bubble_weight_data[coords_i[:,0],coords_i[:,1],coords_i[:,2]]
        mass_array = np.c_[od_mass,od_mass,od_mass]
        bubble_com = np.around((mass_array*np.c_[coords_i[:,0],coords_i[:,1],coords_i[:,2]]).sum(0)/od_mass.sum(),3).tolist()
        bubble_coms.append(bubble_com)
        radius_lb.append(box_region.equivalent_diameter/2)
        areas_lb.append(box_region.area)
        eccentricities_lb.append(np.around(box_region.eccentricity,2))
        contours.append(contour)
        delta_v_len = np.int64((coords_i[:,0].max() - coords_i[:,0].min())/2)
        range_v = [coords_i[:,0].min()-delta_v_len,coords_i[:,0].max()+delta_v_len]
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


def Get_Bubble_Clump_Ids(bubble_region,regions_data,connected_ids_dict):
    coords_i = bubble_region.coords
    bubble_clump_ids = list(set(regions_data[coords_i[:,0],coords_i[:,1],coords_i[:,2]]))
    if 0 in bubble_clump_ids:
        bubble_clump_ids.remove(0)
    bubble_clump_ids = np.array(bubble_clump_ids) - 1
    
    bubble_clump_ids_con = []
    bubble_clump_ids_con_used = []
    for clump_id in bubble_clump_ids:
        bubble_clump_ids_con += connected_ids_dict[clump_id]
    bubble_clump_ids_con = list(set(bubble_clump_ids_con))
    for clump_id_con in bubble_clump_ids_con:
        if clump_id_con not in bubble_clump_ids:
            bubble_clump_ids_con_used.append(clump_id_con)
    bubble_clump_ids_con = np.array(bubble_clump_ids_con_used)
    return bubble_clump_ids,bubble_clump_ids_con


def Get_Bubble_Inner_Item(bubble_regions_data,region):
    coords = region.coords
    x_min = coords[:,0].min()
    x_max = coords[:,0].max()
    y_min = coords[:,1].min()
    y_max = coords[:,1].max()
    z_min = coords[:,2].min()
    z_max = coords[:,2].max()
    length = np.max([x_max-x_min,y_max-y_min,z_max-z_min])+1
    bubble_inner_item =  np.zeros([length,length,length])
    start_x = np.int64((length - (x_max-x_min))/2)
    start_y = np.int64((length - (y_max-y_min))/2)
    start_z = np.int64((length - (z_max-z_min))/2)
    bubble_inner_item[coords[:,0]-x_min+start_x,coords[:,1]-y_min+start_y,coords[:,2]-z_min+start_z] = \
                        bubble_regions_data[coords[:,0],coords[:,1],coords[:,2]]
    start_coords_inner = [x_min+start_x,y_min+start_y,z_min+start_z]
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





