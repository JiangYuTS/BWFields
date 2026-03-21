import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle,Ellipse
import matplotlib.colors as mcolors
import colorsys
from astropy import units as u
import copy
import colorsys

from DPConCFil import Filament_Class_Funs_Analysis as FCFA 
from . import Bubble_Funs_Morphology as BFM
from . import Bubble_Funs_PV as BFPV      
from . import Bubble_Funs_Table as BFT  
from . import Bubble_Funs_Profile as BFPr
from . import Bubble_Funs_Skeleton as BFS
from . import Bubble_Funs_Tools as BFTools


def Distinct_Dark_Colors(n, s_range=(0.5, 0.9), v_range=(0.5, 0.9), shuffle=False):
    hues = np.linspace(0, 1, n, endpoint=False)
    if shuffle: 
        rng = np.random.default_rng()
        rng.shuffle(hues)
        
    s_low, s_high = s_range
    v_low, v_high = v_range

    out = []
    for h in hues:
        s = np.random.uniform(s_low, s_high)
        v = np.random.uniform(v_low, v_high)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append((r, g, b))
    return np.array(out)


def Plot_Images_2D(origin_data_2D,figsize=(6,6),fontsize=12,title='Integrated Image'):
    fig,(ax0)= plt.subplots(1,1, figsize=figsize)
    
    ax0.set_title('{}'.format(title),fontsize=fontsize,color='r')
    plt.xlabel("Galactic Longitude", fontsize=fontsize)
    plt.ylabel("Galactic Latitude", fontsize=fontsize)
    ax0.tick_params(axis='both', which='major', labelsize=fontsize)
    gci = ax0.imshow(origin_data_2D)
    
    ax0.invert_yaxis()
    cbar = plt.colorbar(gci,pad=0)
    return ax0


def Plot_Clumps_Infor_Limited_Range(clumpsObj,xlim,ylim,vlim,ax0=None,figsize=(12,10),line_scale=3,num_text=True):
    centers = clumpsObj.centers
    angles = clumpsObj.angles
    edges = clumpsObj.edges
    used_ids = []
    clumps_data = np.zeros_like(clumpsObj.origin_data)
    for i in range(len(clumpsObj.clump_coords_dict)):
        clump_coords = (clumpsObj.clump_coords_dict[i][:, 0], clumpsObj.clump_coords_dict[i][:, 1], \
                        clumpsObj.clump_coords_dict[i][:, 2])
        clumps_data[clump_coords] = clumpsObj.origin_data[clump_coords]
    if ax0 is None:
        fig, (ax0) = plt.subplots(1,1,figsize=figsize)
    for index in range(len(centers)):
        center_x = centers[index][1]
        center_y = centers[index][2]
        center_v = centers[index][0]
        if center_y > xlim[0] and center_y < xlim[1] and \
            center_x > ylim[0] and center_x < ylim[1] and \
            center_v > vlim[0] and center_v < vlim[1]: 
            cen_x1 = center_x + line_scale * np.sin(np.deg2rad(angles[index]))
            cen_y1 = center_y + line_scale * np.cos(np.deg2rad(angles[index]))
            cen_x2 = center_x - line_scale * np.sin(np.deg2rad(angles[index]))
            cen_y2 = center_y - line_scale * np.cos(np.deg2rad(angles[index]))
            if edges[index] == 0:
                lines = plt.plot([cen_y1, center_y, cen_y2], [cen_x1, center_x, cen_x2])
                plt.setp(lines[0], linewidth=2, color='red', marker='.', markersize=3)
            ax0.plot(center_y, center_x, 'r*', markersize=6)
            if num_text==True:
                ax0.text(center_y,center_x,"{}".format(index),color='r',fontsize=10)
            used_ids.append(index)
    ax0.imshow(clumps_data.sum(0),
               origin='lower',
               cmap='gray',
               interpolation='none')
    ax0.contourf(clumps_data.sum(0),
                 levels=[0., .1],
                 colors='w')
    
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)
    return ax0,used_ids
    

def Plot_Clumps_Infor_By_Ids(clumpsObj,clump_ids,ax0=None,figsize=(16,14),plot_clump=True,plot_contour=False,tick_logic=True,cbar_logic=True,\
                             fontsize=14,cmap='gray',colors='red',num_text=False,line_scale=3,linewidth=2):
    clump_angles = clumpsObj.angles
    clump_edges = clumpsObj.edges
    clump_centers = clumpsObj.centers
    clump_centers_wcs = clumpsObj.centers_wcs
    origin_data = clumpsObj.origin_data
    regions_data = clumpsObj.regions_data
    data_wcs = clumpsObj.data_wcs
    connected_ids_dict = clumpsObj.connected_ids_dict
    clump_coords_dict = clumpsObj.clump_coords_dict
    
    filament_coords, filament_item, data_wcs_item, regions_data_T, start_coords, \
                 filament_item_mask_2D, lb_area = FCFA.Filament_Coords(origin_data, \
                 regions_data, data_wcs, clump_coords_dict, clump_ids, CalSub=False)

    if ax0 is None and tick_logic:
        fig = plt.figure(figsize=figsize)
        ax0 = fig.add_subplot(111,projection=data_wcs_item.celestial)
    elif ax0 is None and tick_logic:
        fig, (ax0) = plt.subplots(1,1,figsize=figsize)
    elif ax0 is None and not tick_logic:
        fig, (ax0) = plt.subplots(1,1,figsize=figsize)
        ax0.set_xticks([]), ax0.set_yticks([])
    if tick_logic:
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
    if len(colors) == 1:
        colors = [colors]*len(clump_ids)
    if plot_clump:
        k = 0
        for index in clump_ids:
            center_x = clump_centers[index][1]-start_coords[1]
            center_y = clump_centers[index][2]-start_coords[2]
            cen_x1 = center_x + line_scale*np.sin(np.deg2rad(clump_angles[index]))
            cen_y1 = center_y + line_scale*np.cos(np.deg2rad(clump_angles[index]))
            cen_x2 = center_x - line_scale*np.sin(np.deg2rad(clump_angles[index]))
            cen_y2 = center_y - line_scale*np.cos(np.deg2rad(clump_angles[index]))
            if clump_edges[index] == 0:
                lines = plt.plot([cen_y1,center_y,cen_y2],[cen_x1,center_x,cen_x2])
                plt.setp(lines[0], linewidth=linewidth,color = colors[k],marker='.',markersize=3)
            ax0.plot(center_y,center_x,color=colors[k],marker='*',markersize = 6)
            if num_text==True:
                ax0.text(center_y,center_x,"{}".format(index),color=colors[k],fontsize=fontsize-2)
            k += 1
            if plot_contour:
                _, _, contour_i, _ = BFM.Cal_2D_Region_From_3D_Coords(clump_coords_dict[index],cal_contours=True)
                ax0.plot(contour_i[:,1]-start_coords[2],contour_i[:,0]-start_coords[1],linewidth=linewidth-0.5)
            
    show_data = filament_item.sum(0) * clumpsObj.delta_v
    vmin = np.min(show_data[show_data != 0])
    vmax = np.nanpercentile(show_data[np.where(show_data != 0)], 99.5)
    gci = ax0.imshow(show_data,
               origin='lower',
               interpolation='none',
               cmap=cmap,
               norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    ax0.contourf(show_data,
                 levels=[0., .0001],
                 colors='w')
    
    if cbar_logic:
        cbar=plt.colorbar(gci,pad=0)
        cbar.ax.tick_params(labelsize=fontsize) 
        cbar.set_label(label='K km s$^{-1}$',fontsize=fontsize) 
        
    return filament_item,start_coords,ax0


def Plot_Bubble_Inner_Contours(bubbleObj,bg_data=1,bubble_valid_ids=None,plot_contour=True,num_text=True,\
               cmap='gray',ax0=None,figsize=(5,4),tick_logic=True,cbar_logic=True,linewidth=2,markersize=2,fontsize=12,spacing=None):
    
    data_wcs = bubbleObj.clumpsObj.data_wcs
    contours = bubbleObj.contours
    bubble_coms = bubbleObj.bubble_coms
    origin_data = bubbleObj.clumpsObj.origin_data
    regions_data = bubbleObj.clumpsObj.regions_data
    clumps_data = np.zeros_like(origin_data)
    clumps_data[regions_data>0] = origin_data[regions_data>0]
    bubble_weight_data = bubbleObj.bubble_weight_data
    bubble_regions_data = bubbleObj.bubble_regions_data

    if ax0 is None and tick_logic:
        fig = plt.figure(figsize=figsize)
        ax0 = fig.add_subplot(111,projection=data_wcs.celestial)
    else:
        fig, (ax0) = plt.subplots(1,1,figsize=figsize)
        ax0.set_xticks([]), ax0.set_yticks([])
    if tick_logic:
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
        if spacing != None:
            lon.set_ticks(spacing=spacing)
            lat.set_ticks(spacing=spacing)
        ax0.tick_params(axis='both', which='major', labelsize=fontsize)

    if plot_contour:
        colors_T = Distinct_Dark_Colors(len(bubble_coms))
        np.random.seed(0)
        colors_T = np.random.permutation(colors_T)
        if bubble_valid_ids is None:
            for i in range(len(contours)):
                ax0.plot(contours[i][:,1],contours[i][:,0],color=colors_T[i],linewidth=linewidth)
            for i in range(len(bubble_coms)):
                ax0.plot(bubble_coms[i][2],bubble_coms[i][1],color=colors_T[i],marker='o',markersize=markersize)
                if num_text==True:
                    ax0.text(bubble_coms[i][2],bubble_coms[i][1],"{}".format(i+1),color='r',fontsize=fontsize-2)
        else:
            for i in bubble_valid_ids:
                ax0.plot(contours[i][:,1],contours[i][:,0],color=colors_T[i],linewidth=linewidth)
            for i in bubble_valid_ids:
                ax0.plot(bubble_coms[i][2],bubble_coms[i][1],color=colors_T[i],marker='o',markersize=markersize)
                if num_text==True:
                    ax0.text(bubble_coms[i][2],bubble_coms[i][1],"{}".format(i+1),color='r',fontsize=fontsize-2)
                    
    if bg_data == 1:
        show_data = clumps_data.sum(0)*bubbleObj.vel_resolution
        cbar_name = 'K km s$^{-1}$'
    elif bg_data == 2:
        show_data = bubble_weight_data*(bubble_regions_data>0)
        show_data = show_data.sum(0)
        show_data[show_data != 0] = np.sqrt(show_data[show_data != 0])
        cbar_name = ''
        ax0.text(0.85, 0.1, r'$\sqrt{W_{l,b,v}}$', 
                transform=ax0.transAxes, verticalalignment='top', fontsize=fontsize-2,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    vmin = np.min(show_data[show_data != 0])
    vmax = np.nanpercentile(show_data[np.where(show_data != 0)], 99.5)
    
    gci = ax0.imshow(show_data,
               origin='lower',
               cmap=cmap,
               interpolation='none',
               norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    ax0.contourf(show_data,
                 levels=[0., .0001],
                 colors='w')
    if cbar_logic:
        cbar=plt.colorbar(gci,pad=0)
        cbar.ax.tick_params(labelsize=fontsize) 
        cbar.set_label(label=cbar_name,fontsize=fontsize)
    
    return ax0


def Plot_Bubble_Inner_Data(bubbleObj,index=0,text_title='B',cmap='hot',fontsize=12):
    bubble_inner_data_item = bubbleObj.bubble_inner_data_item 
    bubble_com_inner_item = np.array(bubbleObj.bubble_coms[index]) - bubbleObj.start_coords_inner 
    bubble_com_item_wcs = bubbleObj.bubble_com_item_wcs
    range_v_wcs = bubbleObj.ranges_v_wcs[index]
    
    fig,(ax0,ax1,ax2)= plt.subplots(1,3, figsize=(12, 12))
    # ax0.set_title('LB Integral Data',fontsize=12,color='r')
    ax0.set_xlabel('{}'.format('Galactic Longitude'),fontsize=fontsize,color='black')
    ax0.set_ylabel('{}'.format('Galactic Latitude'),fontsize=fontsize,color='black')
    
    # ax1.set_title('LV Integral Mask',fontsize=12,color='r')
    ax1.set_xlabel('{}'.format('Galactic Longitude'),fontsize=fontsize,color='black')
    ax1.set_ylabel('{}'.format('Velocity'),fontsize=fontsize,color='black')
    
    # ax2.set_title('BV Integral Mask',fontsize=12,color='r')
    ax2.set_xlabel('{}'.format('Galactic Latitude'),fontsize=fontsize,color='black')
    ax2.set_ylabel('{}'.format('Velocity'),fontsize=fontsize,color='black')
    
    par_map = bubble_inner_data_item.sum(0)
    vmin = par_map[np.where(par_map!=0)].min()
    vmax = par_map.max()
    gci = ax0.imshow(par_map, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    ax0.contourf(par_map,levels = [-1, 0.00001],colors = 'w')

    par_map = bubble_inner_data_item.sum(1)
    vmin = par_map[np.where(par_map!=0)].min()
    vmax = par_map.max()
    gci = ax1.imshow(par_map, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    ax1.contourf(par_map,levels = [-1, 0.00001],colors = 'w')

    par_map = bubble_inner_data_item.sum(2)
    vmin = par_map[np.where(par_map!=0)].min()
    vmax = par_map.max()
    gci = ax2.imshow(par_map, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    ax2.contourf(par_map,levels = [-1, 0.00001],colors = 'w')

    ax0.scatter(bubble_com_inner_item[2],bubble_com_inner_item[1],color='green',marker='o',s=50)
    ax1.scatter(bubble_com_inner_item[2],bubble_com_inner_item[0],color='green',marker='o',s=50)
    ax2.scatter(bubble_com_inner_item[1],bubble_com_inner_item[0],color='green',marker='o',s=50)
    
    ax0.text(0.05, 0.95, '({}) Cavity Weight'.format(text_title), color='black',\
                         transform=ax0.transAxes, verticalalignment='top', fontsize=10)
    ax0.text( 0.05, 0.88, r'Com = [{}$^\circ$, {}$^\circ$, {}km s $^{{-1}}$]'.format(
                         bubble_com_item_wcs[0],bubble_com_item_wcs[1],bubble_com_item_wcs[2]),\
                         color='black',transform=ax0.transAxes, verticalalignment='top', fontsize=10)
    ax0.text(0.05, 0.81, r'V Range = ({}$\sim${}) km s$^{{-1}}$'.format(range_v_wcs[0],range_v_wcs[1]),\
                         color='black',transform=ax0.transAxes, verticalalignment='top', fontsize=10)

    ax0.set_xticks([]), ax0.set_yticks([])
    ax1.set_xticks([]), ax1.set_yticks([])
    ax2.set_xticks([]), ax2.set_yticks([])
    return ax0,ax1,ax2


def Plot_Bubble_Gas_Region(bubbleObj,add_con=False,plot_clump=False,plot_contour=False,cmap='gray',text_num=False,text_com=True,
                           tick_logic=True,cbar_logic=True,linewidth=2,ax0=None,figsize=(8,6),fontsize=12):
    bubble_clump_ids = bubbleObj.bubble_clump_ids 
    bubble_clump_ids_con = bubbleObj.bubble_clump_ids_con  
    bubble_coords = bubbleObj.bubble_coords 
    bubble_com_item = bubbleObj.bubble_com_item
    bubble_com_item_wcs = bubbleObj.bubble_com_item_wcs
    bubble_contour = bubbleObj.bubble_contour
    ellipse_infor = bubbleObj.ellipse_infor 
    ellipse_coords = bubbleObj.ellipse_coords 
    start_coords_item = bubbleObj.start_coords
    
    if not add_con:
        clump_ids = list(bubble_clump_ids) 
        colors = ['red']*len(bubble_clump_ids)
        if len(colors) == 1:
            colors = [colors]
        bubble_gas_com_wcs = bubbleObj.bubble_gas_com_wcs_1
        bubble_gas_ranges_lbv_min = bubbleObj.bubble_gas_ranges_lbv_min_1
        bubble_gas_ranges_lbv_max = bubbleObj.bubble_gas_ranges_lbv_max_1
    else :
        clump_ids = list(bubble_clump_ids) + list(bubble_clump_ids_con)
        colors = ['red']*len(bubble_clump_ids) + ['blue']*len(bubble_clump_ids_con)
        bubble_gas_com_wcs = bubbleObj.bubble_gas_com_wcs_2
        bubble_gas_ranges_lbv_min = bubbleObj.bubble_gas_ranges_lbv_min_2
        bubble_gas_ranges_lbv_max = bubbleObj.bubble_gas_ranges_lbv_max_2
        
    bubble_item,start_coords,ax0 = Plot_Clumps_Infor_By_Ids(bubbleObj.clumpsObj,clump_ids,tick_logic=tick_logic,cbar_logic=cbar_logic,\
                            cmap=cmap,ax0=ax0,figsize=figsize,fontsize=fontsize,plot_clump=plot_clump,plot_contour=plot_contour,\
                            colors=colors,num_text=text_num,line_scale=3)
    
    bubble_com_item_i = np.array([bubble_com_item[1]+start_coords_item[1]-start_coords[1],\
                                        bubble_com_item[2]+start_coords_item[2]-start_coords[2]])
    bubble_contour_i = np.c_[bubble_contour[:,0]+start_coords_item[1]-start_coords[1],\
                                bubble_contour[:,1]+start_coords_item[2]-start_coords[2]]
    ellipse_coords_i = np.c_[ellipse_coords[:,0]-start_coords[1],ellipse_coords[:,1]-start_coords[2]]
    ax0.scatter(bubble_com_item_i[1],bubble_com_item_i[0],\
             color='green',marker='o',s=30,label="Cavity Com",zorder=3)
    ax0.plot(bubble_contour_i[:,1],bubble_contour_i[:,0],\
             linewidth=linewidth,color='green',label="Cavity Contour",zorder=3)
    ax0.scatter(ellipse_infor[1]-start_coords[2],ellipse_infor[0]-start_coords[1],color='cyan',\
                marker='*',s=20,label="Fited Cavity Center",zorder=4)
    ax0.plot(ellipse_coords_i[:,1],ellipse_coords_i[:,0],linewidth=linewidth-0.5,color='cyan',label="Fited Cavity Contour",zorder=4)
    
    if text_com:
        ax0.text( 0.05, 0.88, r'Gas Com = [{}$^\circ$, {}$^\circ$, {}km s $^{{-1}}$]'.format(
                             bubble_gas_com_wcs[0],bubble_gas_com_wcs[1],bubble_gas_com_wcs[2]),\
                             color='green',transform=ax0.transAxes, verticalalignment='top', fontsize=10)
        ax0.text(0.05, 0.81, r'Gas V Range = ({}$\sim${}) km s$^{{-1}}$'.format(bubble_gas_ranges_lbv_min[2],bubble_gas_ranges_lbv_max[2]),\
                             color='green',transform=ax0.transAxes, verticalalignment='top', fontsize=10)

    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(fontsize=fontsize-2,loc='upper right')
    return ax0
    

def Plot_Velocity_Channel_Maps(bubbleObj,gas_type=2,velocity_range=None,n_slices=9,integration_width=None,figsize=(12, 11),
    cmap='viridis',vmin=None,vmax=None,v_label_position='top_right',percentile_value=90,fontsize=16,linewidth=2,add_contours=True,
    contour_levels=None,contour_colors='black',contour_linewidths=1.5,contour_alpha=0.8,show_ellipse=True,
    ellipse_color='blue',ellipse_linestyle='--',show_legend=True,cbar_logic=False):
    """
    Plot integrated velocity channel maps in Galactic coordinates
    
    Key Improvement: Instead of single velocity channels, shows integrated 
    intensity over velocity ranges
    
    Parameters:
    -----------
    bubbleObj : object
        Bubble object containing data_cube and wcs
    velocity_range : tuple, optional
        Velocity range (v_min, v_max) in km/s
    n_slices : int
        Number of integrated velocity maps to display (default: 9)
    integration_width : float, optional
        Width of velocity integration in km/s (default: auto-calculated)
    figsize : tuple
        Figure size (default: (12, 11))
    cmap : str
        Colormap name (default: 'viridis')
    vmin, vmax : float
        Color range limits
    v_label_position : str
        Position of velocity label: 'bottom_right', 'top_left', 'top_right', 'bottom_left'
    percentile_value : float
        Percentile for signal detection (default: 90)
    fontsize : int
        Font size for labels (default: 16)
    linewidth : float
        Line width for ellipse (default: 2)
    add_contours : bool
        Add contour lines (default: True)
    contour_levels : int or array, optional
        Contour levels
    contour_colors : str
        Contour color (default: 'black')
    contour_linewidths : float
        Contour line width (default: 1.5)
    contour_alpha : float
        Contour transparency (default: 0.8)
    show_ellipse : bool
        Show fitted ellipse (default: True)
    ellipse_color : str
        Ellipse color (default: 'blue')
    ellipse_linestyle : str
        Ellipse line style (default: '--')
    show_legend : bool
        Show legend (default: True)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    # Get data and WCS
    if gas_type == 1:
        data_cube = bubbleObj.bubble_gas_item_1
        data_wcs_gas_item = bubbleObj.data_wcs_gas_item_2
        start_coords_gas_item = bubbleObj.start_coords_gas_item_1
    elif gas_type == 2:
        data_cube = bubbleObj.bubble_gas_item_2
        data_wcs_gas_item = bubbleObj.data_wcs_gas_item_2
        start_coords_gas_item = bubbleObj.start_coords_gas_item_2
    else:
        print('Chose a gas type, 1 or 2.')
        
    # Detect signal range
    threshold = np.percentile(data_cube[data_cube > 0], percentile_value)
    data_coords = np.where(data_cube > threshold)
    start_idx, end_idx = data_coords[0].min(), data_coords[0].max()
    
    nv_original, ny, nx = data_cube.shape
    
    # Subset data to signal range if no velocity_range specified
    if velocity_range is None:
        data_cube = data_cube[start_idx:end_idx+1, :, :]
        velocity_offset = start_idx
    else:
        velocity_offset = 0
    
    nv, ny, nx = data_cube.shape
    
    # Get velocity axis
    try:
        velocities = None
        vel_indices = np.arange(nv) + velocity_offset
        if data_wcs_gas_item.naxis == 3:
            vel_indices_wcs_T = data_wcs_gas_item.all_pix2world(vel_indices,vel_indices,vel_indices,0)
        elif data_wcs_gas_item.naxis == 4:
            vel_indices_wcs_T = data_wcs_gas_item.all_pix2world(vel_indices,vel_indices,vel_indices,0,0)
        velocities = (vel_indices_wcs_T[2]*data_wcs_gas_item.wcs.cunit[2]).to(u.km/u.s).value
    except:
        if velocity_range is None:
            velocity_range = (0, 10)
        velocities = np.linspace(velocity_range[0], velocity_range[1], nv)
    
    # Calculate velocity channel width
    delta_v = np.abs(velocities[1] - velocities[0])
    
    # Determine integration width
    if integration_width is None:
        # Auto-calculate: divide total range into n_slices regions
        total_range = velocities[-1] - velocities[0]
        integration_width = total_range / n_slices
    
    # Calculate number of channels to integrate
    n_channels_integrate = max(1, int(integration_width / delta_v))
    
    # Create velocity integration ranges
    integration_ranges = []
    integrated_data = []
    
    # Calculate how many integrated maps we can create
    n_actual_slices = min(n_slices, nv // n_channels_integrate)
    
    # Distribute channels evenly
    step = max(1, nv // n_actual_slices)
    
    for i in range(n_actual_slices):
        start_ch = i * step
        end_ch = min(start_ch + n_channels_integrate, nv)
        
        if start_ch >= nv:
            break
        
        # Integrate over velocity channels
        integrated = np.nansum(data_cube[start_ch:end_ch, :, :], axis=0)
        
        # Get velocity range for this integration
        v_start = velocities[start_ch]
        v_end = velocities[min(end_ch-1, nv-1)]
        
        integration_ranges.append((v_start, v_end))
        integrated_data.append(integrated)
    
    n_actual_slices = len(integrated_data)
    
    # Create subplots
    rows = int(np.ceil(np.sqrt(n_actual_slices)))
    cols = int(np.ceil(n_actual_slices / rows))
    
    fig = plt.figure(figsize=figsize)
    axes = []
    
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i+1, projection=data_wcs_gas_item.celestial)
        axes.append(ax)
    
    axes = np.array(axes).reshape(rows, cols)
    
    # Get ellipse coordinates if available
    if show_ellipse and hasattr(bubbleObj, 'ellipse_coords'):
        ellipse_coords = bubbleObj.ellipse_coords
        start_coords_gas_item_2 = bubbleObj.start_coords_gas_item_2
        ellipse_coords_i = np.c_[ellipse_coords[:, 0] - start_coords_gas_item_2[1],
                                 ellipse_coords[:, 1] - start_coords_gas_item_2[2]]
    else:
        show_ellipse = False
    
    # Determine global color range
    if vmin is None or vmax is None:
        all_data = np.concatenate([d[d > 0].flatten() for d in integrated_data if np.any(d > 0)])
        if len(all_data) > 0:
            if vmin is None:
                vmin = np.percentile(all_data, 1)
            if vmax is None:
                vmax = np.percentile(all_data, 99.5)
        else:
            vmin, vmax = 0, 1
    
    # Plot each integrated velocity map

    id_names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
    for i in range(n_actual_slices):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # Get integrated data and velocity range
        show_data = integrated_data[i] * bubbleObj.clumpsObj.delta_v
        v_start, v_end = integration_ranges[i]
        
        # Show ellipse
        if show_ellipse:
            ax.plot(ellipse_coords_i[:, 1], ellipse_coords_i[:, 0],
                   linewidth=linewidth, color=ellipse_color,
                   linestyle=ellipse_linestyle, zorder=4)
        
        # Plot image
        if cbar_logic:
            im = ax.imshow(show_data,origin='lower',interpolation='none',
                      cmap=cmap,aspect='auto',norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
        else:
            vmin = np.min(show_data[show_data != 0])
            vmax = np.nanpercentile(show_data[np.where(show_data != 0)], 99.5)
            im = ax.imshow(show_data,origin='lower',interpolation='none',
                      cmap=cmap,aspect='auto',norm=mcolors.Normalize(vmin=vmin, vmax=vmax))

        ax.text(0.05, 0.95, '({})'.format(id_names[i]), color='black',transform=ax.transAxes, \
                                    verticalalignment='top', fontsize=fontsize)
        
        # Mask zero values
        ax.contourf(show_data,levels=[0., 0.0001],colors='w',zorder=1)
        
        # Add contours
        if add_contours and np.any(show_data > 0):
            if contour_levels is None:
                data_min = np.nanmin(show_data[show_data > 0])
                data_max = np.nanmax(show_data)
                levels = np.linspace(data_min + 0.2*(data_max-data_min),
                                   data_min + 0.9*(data_max-data_min), 6)
            elif isinstance(contour_levels, int):
                levels = contour_levels
            else:
                levels = contour_levels
            
            contours = ax.contour(show_data, levels=levels,colors=contour_colors,
                                linewidths=contour_linewidths,alpha=contour_alpha,origin='lower',zorder=3)
        
        # Add velocity range label (improved text)
        velocity_text = f'{v_start:.2f} $\\sim$ {v_end:.2f} km s$^{{-1}}$'
        
        # Determine label position
        if v_label_position == 'bottom_right':
            text_x, text_y = 0.95, 0.05
            ha, va = 'right', 'bottom'
        elif v_label_position == 'top_right':
            text_x, text_y = 0.95, 0.95
            ha, va = 'right', 'top'
        elif v_label_position == 'top_left':
            text_x, text_y = 0.05, 0.95
            ha, va = 'left', 'top'
        else:  # bottom_left
            text_x, text_y = 0.05, 0.05
            ha, va = 'left', 'bottom'
        
        # Add text with improved styling
        ax.text(text_x, text_y, velocity_text,transform=ax.transAxes,fontsize=fontsize-2,
               fontweight='normal',ha=ha, va=va,
               bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='white',
                        edgecolor='black',
                        linewidth=0.5,
                        alpha=0.85), zorder=5)
        
        plt.rcParams['xtick.color'] = 'green'
        plt.rcParams['ytick.color'] = 'green'
        ax.coords[0].set_axislabel('Galactic Longitude', fontsize=fontsize)
        ax.coords[1].set_axislabel('Galactic Latitude', fontsize=fontsize)
        ax.coords[0].set_ticklabel(fontproperties={'family': 'DejaVu Sans'})
        ax.coords[1].set_ticklabel(fontproperties={'family': 'DejaVu Sans'})
        ax.coords[0].set_major_formatter('d.d')
        ax.coords[1].set_major_formatter('d.d')
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        
        # Only show labels on outer edges
        if row<rows-1 or col>0:#rows - 1:
            ax.coords[0].set_ticklabel_visible(False)
            ax.coords[0].set_axislabel('')
            ax.coords[1].set_ticklabel_visible(False)
            ax.coords[1].set_axislabel('')
    
    # Hide unused subplots
    for i in range(n_actual_slices, rows * cols):
        row = i // cols
        col = i % cols
        print(row, col)
        if row != 0 and col!=2:
            axes[row, col].axis('off')

    if show_legend and show_ellipse and n_actual_slices > 0:
        last_row = (n_actual_slices - 1) // cols
        last_col = (n_actual_slices - 1) % cols
        axes[last_row, last_col].plot([], [], linewidth=linewidth,
                                      color=ellipse_color,
                                      linestyle=ellipse_linestyle,
                                      label='Fitted Cavity Ellipse')
        axes[last_row, last_col].legend(fontsize=fontsize-4,
                                        loc='lower right',
                                        framealpha=0.8)
    
    # Adjust spacing
    plt.tight_layout()
    
    # Add colorbar
    if cbar_logic:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('K km s$^{-1}$', fontsize=fontsize-2)
        cbar.ax.tick_params(labelsize=fontsize-4)
        
    return axes


def Plot_Velocity_Channel_Maps_V2(bubbleObj,gas_type=2,velocity_range=None,n_slices=9,integration_width=None,figsize=(12, 11),
    cmap='viridis',vmin=None,vmax=None,v_label_position='top_right',percentile_value=90,fontsize=16,linewidth=2,add_contours=True,
    contour_levels=None,contour_colors='black',contour_linewidths=1.5,contour_alpha=0.8,show_ellipse=True,
    ellipse_color='blue',ellipse_linestyle='--',show_legend=True,cbar_logic=False):
    """
    Plot integrated velocity channel maps in Galactic coordinates
    
    Key Improvement: Instead of single velocity channels, shows integrated 
    intensity over velocity ranges
    
    Parameters:
    -----------
    bubbleObj : object
        Bubble object containing data_cube and wcs
    velocity_range : tuple, optional
        Velocity range (v_min, v_max) in km/s
    n_slices : int
        Number of integrated velocity maps to display (default: 9)
    integration_width : float, optional
        Width of velocity integration in km/s (default: auto-calculated)
    figsize : tuple
        Figure size (default: (12, 11))
    cmap : str
        Colormap name (default: 'viridis')
    vmin, vmax : float
        Color range limits
    v_label_position : str
        Position of velocity label: 'bottom_right', 'top_left', 'top_right', 'bottom_left'
    percentile_value : float
        Percentile for signal detection (default: 90)
    fontsize : int
        Font size for labels (default: 16)
    linewidth : float
        Line width for ellipse (default: 2)
    add_contours : bool
        Add contour lines (default: True)
    contour_levels : int or array, optional
        Contour levels
    contour_colors : str
        Contour color (default: 'black')
    contour_linewidths : float
        Contour line width (default: 1.5)
    contour_alpha : float
        Contour transparency (default: 0.8)
    show_ellipse : bool
        Show fitted ellipse (default: True)
    ellipse_color : str
        Ellipse color (default: 'blue')
    ellipse_linestyle : str
        Ellipse line style (default: '--')
    show_legend : bool
        Show legend (default: True)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    # Get data and WCS
    if gas_type == 1:
        data_cube = bubbleObj.bubble_gas_item_1
        data_wcs_gas_item = bubbleObj.data_wcs_gas_item_2
        start_coords_gas_item = bubbleObj.start_coords_gas_item_1
    elif gas_type == 2:
        data_cube = bubbleObj.bubble_gas_item_2
        data_wcs_gas_item = bubbleObj.data_wcs_gas_item_2
        start_coords_gas_item = bubbleObj.start_coords_gas_item_2
    else:
        print('Chose a gas type, 1 or 2.')
        
    # Detect signal range
    threshold = np.percentile(data_cube[data_cube > 0], percentile_value)
    data_coords = np.where(data_cube > threshold)
    start_idx, end_idx = data_coords[0].min(), data_coords[0].max()
    
    nv_original, ny, nx = data_cube.shape
    
    # Subset data to signal range if no velocity_range specified
    if velocity_range is None:
        data_cube = data_cube[start_idx:end_idx+1, :, :]
        velocity_offset = start_idx
    else:
        velocity_offset = 0
    
    nv, ny, nx = data_cube.shape
    
    # Get velocity axis
    try:
        velocities = None
        vel_indices = np.arange(nv) + velocity_offset
        if data_wcs_gas_item.naxis == 3:
            vel_indices_wcs_T = data_wcs_gas_item.all_pix2world(vel_indices,vel_indices,vel_indices,0)
        elif data_wcs_gas_item.naxis == 4:
            vel_indices_wcs_T = data_wcs_gas_item.all_pix2world(vel_indices,vel_indices,vel_indices,0,0)
        velocities = (vel_indices_wcs_T[2]*data_wcs_gas_item.wcs.cunit[2]).to(u.km/u.s).value
    except:
        if velocity_range is None:
            velocity_range = (0, 10)
        velocities = np.linspace(velocity_range[0], velocity_range[1], nv)
    
    # Calculate velocity channel width
    delta_v = np.abs(velocities[1] - velocities[0])
    
    # Determine integration width
    if integration_width is None:
        # Auto-calculate: divide total range into n_slices regions
        total_range = velocities[-1] - velocities[0]
        integration_width = total_range / n_slices
    
    # Calculate number of channels to integrate
    n_channels_integrate = max(1, int(integration_width / delta_v))
    
    # Create velocity integration ranges
    integration_ranges = []
    integrated_data = []
    
    # Calculate how many integrated maps we can create
    n_actual_slices = min(n_slices, nv // n_channels_integrate)
    
    # Distribute channels evenly
    step = max(1, nv // n_actual_slices)
    
    for i in range(n_actual_slices):
        start_ch = i * step
        end_ch = min(start_ch + n_channels_integrate, nv)
        
        if start_ch >= nv:
            break
        
        # Integrate over velocity channels
        integrated = np.nansum(data_cube[start_ch:end_ch, :, :], axis=0)
        
        # Get velocity range for this integration
        v_start = velocities[start_ch]
        v_end = velocities[min(end_ch-1, nv-1)]
        
        integration_ranges.append((v_start, v_end))
        integrated_data.append(integrated)
    
    n_actual_slices = len(integrated_data)
    
    # Create subplots
    rows = int(np.ceil(np.sqrt(n_actual_slices)))
    cols = int(np.ceil(n_actual_slices / rows))
    
    fig = plt.figure(figsize=figsize)
    axes = []
    
    for i in range(rows * cols-1):
        ax = fig.add_subplot(rows, cols, i+1, projection=data_wcs_gas_item.celestial)
        axes.append(ax)
    for i in [rows * cols-1]:
        ax = fig.add_subplot(rows, cols, i+1)
        axes.append(ax)
    
    axes = np.array(axes).reshape(rows, cols)
    
    # Get ellipse coordinates if available
    if show_ellipse and hasattr(bubbleObj, 'ellipse_coords'):
        ellipse_coords = bubbleObj.ellipse_coords
        start_coords_gas_item_2 = bubbleObj.start_coords_gas_item_2
        ellipse_coords_i = np.c_[ellipse_coords[:, 0] - start_coords_gas_item_2[1],
                                 ellipse_coords[:, 1] - start_coords_gas_item_2[2]]
    else:
        show_ellipse = False
    
    # Determine global color range
    if vmin is None or vmax is None:
        all_data = np.concatenate([d[d > 0].flatten() for d in integrated_data if np.any(d > 0)])
        if len(all_data) > 0:
            if vmin is None:
                vmin = np.percentile(all_data, 1)
            if vmax is None:
                vmax = np.percentile(all_data, 99.5)
        else:
            vmin, vmax = 0, 1
    
    # Plot each integrated velocity map

    id_names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
    for i in range(n_actual_slices):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # Get integrated data and velocity range
        show_data = integrated_data[i] * bubbleObj.clumpsObj.delta_v
        v_start, v_end = integration_ranges[i]
        
        # Show ellipse
        if show_ellipse:
            ax.plot(ellipse_coords_i[:, 1], ellipse_coords_i[:, 0],
                   linewidth=linewidth, color=ellipse_color,
                   linestyle=ellipse_linestyle, zorder=4)
        
        # Plot image
        if cbar_logic:
            im = ax.imshow(show_data,origin='lower',interpolation='none',
                      cmap=cmap,aspect='auto',norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
        else:
            vmin = np.min(show_data[show_data != 0])
            vmax = np.nanpercentile(show_data[np.where(show_data != 0)], 99.5)
            im = ax.imshow(show_data,origin='lower',interpolation='none',
                      cmap=cmap,aspect='auto',norm=mcolors.Normalize(vmin=vmin, vmax=vmax))

        ax.text(0.05, 0.95, '({})'.format(id_names[i]), color='black',transform=ax.transAxes, \
                                    verticalalignment='top', fontsize=fontsize)
        
        # Mask zero values
        ax.contourf(show_data,levels=[0., 0.0001],colors='w',zorder=1)
        
        # Add contours
        if add_contours and np.any(show_data > 0):
            if contour_levels is None:
                data_min = np.nanmin(show_data[show_data > 0])
                data_max = np.nanmax(show_data)
                levels = np.linspace(data_min + 0.2*(data_max-data_min),
                                   data_min + 0.9*(data_max-data_min), 6)
            elif isinstance(contour_levels, int):
                levels = contour_levels
            else:
                levels = contour_levels
            
            contours = ax.contour(show_data, levels=levels,colors=contour_colors,
                                linewidths=contour_linewidths,alpha=contour_alpha,origin='lower',zorder=3)
        
        # Add velocity range label (improved text)
        velocity_text = f'{v_start:.2f} $\\sim$ {v_end:.2f} km s$^{{-1}}$'
        
        # Determine label position
        if v_label_position == 'bottom_right':
            text_x, text_y = 0.95, 0.05
            ha, va = 'right', 'bottom'
        elif v_label_position == 'top_right':
            text_x, text_y = 0.95, 0.95
            ha, va = 'right', 'top'
        elif v_label_position == 'top_left':
            text_x, text_y = 0.05, 0.95
            ha, va = 'left', 'top'
        else:  # bottom_left
            text_x, text_y = 0.05, 0.05
            ha, va = 'left', 'bottom'
        
        # Add text with improved styling
        ax.text(text_x, text_y, velocity_text,transform=ax.transAxes,fontsize=fontsize-2,
               fontweight='normal',ha=ha, va=va,
               bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='white',
                        edgecolor='black',
                        linewidth=0.5,
                        alpha=0.85), zorder=5)
        
        plt.rcParams['xtick.color'] = 'green'
        plt.rcParams['ytick.color'] = 'green'
        ax.coords[0].set_axislabel('Galactic Longitude', fontsize=fontsize)
        ax.coords[1].set_axislabel('Galactic Latitude', fontsize=fontsize)
        ax.coords[0].set_ticklabel(fontproperties={'family': 'DejaVu Sans'})
        ax.coords[1].set_ticklabel(fontproperties={'family': 'DejaVu Sans'})
        ax.coords[0].set_major_formatter('d.d')
        ax.coords[1].set_major_formatter('d.d')
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        
        # Only show labels on outer edges
        if row<rows-1 or col>0:#rows - 1:
            ax.coords[0].set_ticklabel_visible(False)
            ax.coords[0].set_axislabel('')
            ax.coords[1].set_ticklabel_visible(False)
            ax.coords[1].set_axislabel('')
    
    # Hide unused subplots
    for i in range(n_actual_slices, rows * cols):
        row = i // cols
        col = i % cols
        print(row, col)
        if row != 0 and col!=2:
            axes[row, col].axis('off')

    if show_legend and show_ellipse and n_actual_slices > 0:
        last_row = (n_actual_slices - 1) // cols
        last_col = (n_actual_slices - 1) % cols
        axes[last_row, last_col].plot([], [], linewidth=linewidth,
                                      color=ellipse_color,
                                      linestyle=ellipse_linestyle,
                                      label='Fitted Cavity Ellipse')
        axes[last_row, last_col].legend(fontsize=fontsize-4,
                                        loc='lower right',
                                        framealpha=0.8)
    
    # Adjust spacing
    plt.tight_layout()
    
    # Add colorbar
    if cbar_logic:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('K km s$^{-1}$', fontsize=fontsize-2)
        cbar.ax.tick_params(labelsize=fontsize-4)
        
    return axes
    

def Plot_Bubble_Item(bubbleObj,line_index=0,ax0=None,figsize=(8,6),line_logics=[True,True],tick_logic=True,
                     label_l='Intensity and PV Slice',label_e='Cavity Ellipse',color_e='cyan',fontsize=12,spacing=12*u.arcmin):
    # bubble_com = bubbleObj.bubble_com
    # fbubble_com_wcs = bubbleObj.bubble_com_wcs
    # bubble_ratio = bubbleObj.bubble_ratio
    # bubble_angle = bubbleObj.bubble_angle
    bubble_item = bubbleObj.bubble_item
    start_coords = bubbleObj.start_coords
    data_wcs_item = bubbleObj.data_wcs_item
    dictionary_cuts_item = copy.deepcopy(bubbleObj.dictionary_cuts)
    
    # for key in ['plot_peaks', 'plot_cuts']:
    #     dictionary_cuts_item[key] = np.array(dictionary_cuts_item[key]) - start_coords[1:][::-1]
    # for i in range(len(dictionary_cuts_item['points'])):
    #     dictionary_cuts_item['points'][i] -= start_coords[1:][::-1]

    bubble_item_shape = bubble_item.shape

    if ax0 is None and tick_logic:
        fig = plt.figure(figsize=figsize)
        ax0 = fig.add_subplot(111,projection=data_wcs_item.celestial)
    else:
        fig, (ax0) = plt.subplots(1,1,figsize=figsize)
        ax0.set_xticks([]), ax0.set_yticks([])
    if tick_logic:
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
        if spacing != None:
            lon.set_ticks(spacing=spacing)
            lat.set_ticks(spacing=spacing)
        ax0.tick_params(axis='both', which='major', labelsize=fontsize)

    for points in dictionary_cuts_item['points_b']:
        ax0.plot(points[:, 0], points[:, 1], color_e, label="Fited {}".format(label_e),zorder=4,lw=1,marker='.', alpha=0.8, markersize=2.)

    if line_logics[0]:
        for cut_line_id in range(len(dictionary_cuts_item['plot_cuts'])):
            # if pp_distance[cut_line_id] < mask_width_mean / 2:
            start = dictionary_cuts_item['plot_cuts'][cut_line_id][0]
            end = dictionary_cuts_item['plot_cuts'][cut_line_id][1]
            ax0.plot([start[0], end[0]], [start[1], end[1]], color='yellow',linestyle='-', markersize=8., linewidth=1., alpha=0.6)

    if line_logics[1]:
        start = dictionary_cuts_item['plot_cuts'][line_index][0]
        end = dictionary_cuts_item['plot_cuts'][line_index][1]
        ax0.plot([start[0], end[0]], [start[1], end[1]], color='red',label=label_l,linestyle='-.', markersize=8., \
                 linewidth=1.2, alpha=0.6)
    else:
        start = dictionary_cuts_item['plot_cuts'][line_index][0]
        end = dictionary_cuts_item['plot_cuts'][line_index][1]
        ax0.plot([start[0], end[0]], [start[1], end[1]], color='yellow',label=label_l,linestyle='-', markersize=8., \
                 linewidth=1.2, alpha=0.6)

    vmin = np.min(bubble_item.sum(0)[np.where(bubble_item.sum(0) != 0)])
    vmax = np.nanpercentile(bubble_item.sum(0)[np.where(bubble_item.sum(0) != 0)], 98.)
    gci = ax0.imshow(bubble_item.sum(0)*bubbleObj.vel_resolution,
               origin='lower',
               cmap='gray',
               interpolation='none',
               norm=mcolors.Normalize(vmin=vmin*bubbleObj.vel_resolution, vmax=vmax*bubbleObj.vel_resolution))
    ax0.contourf(bubble_item.sum(0)*bubbleObj.vel_resolution,
                 levels=[0., .0001],
                 colors='w')
    if tick_logic:
        cbar=plt.colorbar(gci,pad=0)
        cbar.ax.tick_params(labelsize=fontsize) 
        cbar.set_label(label='K km s$^{-1}$',fontsize=fontsize) 
        
    # fig.tight_layout()
    plt.legend(fontsize=fontsize-2,loc='upper right')
    return ax0


def Plot_Profile(bubbleObj,ax0=None,figsize=(8, 6),fontsize=16):
    dictionary_cuts = bubbleObj.dictionary_cuts
    pix_scale_arcmin = bubbleObj.pix_scale_arcmin
    
    if ax0 is None:
        fig, (ax0) = plt.subplots(1,1,figsize=figsize)
    for i in range(0,len(dictionary_cuts['distance'])):
        # dists_i = dictionary_cuts['distance'][i][np.where(dictionary_cuts['profile'][i]!=0)]
        # delta_dist = dists_i[-1]-dists_i[0]
        # if delta_dist>filamentObj.EProfileLen:
        ax0.plot(dictionary_cuts['distance'][i]*pix_scale_arcmin, dictionary_cuts['profile'][i],c='gray',alpha=0.3)
        
    ax0.plot(bubbleObj.axis_coords_left*pix_scale_arcmin, bubbleObj.mean_profile_left,c='r',marker='.',alpha=1,label='Mean Profile')
    ax0.plot(bubbleObj.axis_coords_right*pix_scale_arcmin, bubbleObj.mean_profile_right,c='r',marker='.',alpha=1)
    ax0.plot(bubbleObj.axis_coords_right*pix_scale_arcmin, bubbleObj.mean_profile_left_r,c='b',marker='.',alpha=1,\
             label='Mean Profile of Left Part')
    
    # ax0.plot(bubbleObj.axis_coords, bubbleObj.profile_fited_G,c='gold',marker='.',alpha=1,label='Gaussian Profile')
    # ax0.plot(bubbleObj.axis_coords, bubbleObj.profile_fited_P,c='purple',marker='.',alpha=1,label='Plummer Profile')
    
    ax0.axvline(0, color='b', linestyle='dashed',alpha=0.3,label='Axis of Symmetry')
    
    # ax0.axvline(0, color='b', linestyle='dashed',alpha=0.5,label='Axis of Symmetry')
    # ax0.text(-25,70,'(a)',color='black',fontsize=fontsize+4)

    ax0.text(0.75, 0.45, f'SIOU: {bubbleObj.profile_IOU:.2f}', 
                transform=ax0.transAxes, verticalalignment='top', fontsize=fontsize,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ax0.text(8,30,'FWHM$_G$={}$\pm${}'.format(bubbleObj.FWHM_G,filamentObj.FWHM_error_G),color='black',fontsize=fontsize)
    # ax0.text(8,20,'FWHM$_P$={}$\pm${}'.format(bubbleObj.FWHM_P,filamentObj.FWHM_error_P),color='black',fontsize=fontsize)
    mean_profile_left_coords = np.where(bubbleObj.mean_profile_left>0)[0]
    mean_profile_right_coords = np.where(bubbleObj.mean_profile_right>0)[0]
    max_len = np.max([mean_profile_left_coords[-1],\
                      mean_profile_right_coords[-1]]) + 5
    plt.xlim(-max_len,max_len)
    fig.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    ticks = ax0.get_xticks()*pix_scale_arcmin
    ax0.set_xticklabels(np.int32(np.around(ticks)))
    
    plt.xlabel("Radial Distance (arcmin)",fontsize=fontsize)
    plt.ylabel(r"Integrated Intensity (K)",fontsize=fontsize)
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    
    plt.legend(fontsize=fontsize-4)
    return ax0


def Plot_Profile_Fit(bubbleObj,color_1='lime',color_2='cyan',ax0=None,figsize=(6,4),fontsize=14):
    dictionary_cuts = bubbleObj.dictionary_cuts
    pix_scale_arcmin = bubbleObj.pix_scale_arcmin
    fit_results = bubbleObj.fit_results
    if ax0 is None:
        fig, (ax0) = plt.subplots(1,1,figsize=figsize)
    for i in range(0,len(dictionary_cuts['distance'])):
        ax0.plot(dictionary_cuts['distance'][i], dictionary_cuts['profile'][i],c='gray',alpha=0.3)
        
    ax0.plot(bubbleObj.axis_coords_left, bubbleObj.mean_profile_left,c='r',marker='.',markersize=2,alpha=1,linewidth=1,label='Mean Profile')
    ax0.plot(bubbleObj.axis_coords_right, bubbleObj.mean_profile_right,c='r',marker='o',markersize=2,alpha=1,linewidth=1)

    x = fit_results['x']
    y_orig = fit_results['y_original']
    if not fit_results['success']:
        print("Fitting failed, cannot plot results")
    else:
        y_fit = fit_results['y_fitted']
        p = fit_results['params']
    
    # Get full data if available (for local fit visualization)
    if 'x_full' in fit_results:
        x_full = fit_results['x_full']
        y_full = fit_results['y_full']
        local_fit = fit_results.get('local_fit', False)
        intensity_threshold = fit_results.get('intensity_threshold', None)
    else:
        x_full, y_full = x, y_orig
        local_fit = False
        intensity_threshold = None
    
    
    # Plot full data in light gray if local fit was used
    # if local_fit and len(x_full) != len(x):
    #     plt.plot(x_full, y_full, 'o-', color='lightgray', markersize=2, alpha=0.5, label='Full Data (Not Fitted)')
    
    # Original fitted data - with customizable style options
    plt.plot(x, y_orig, 'ro-', label='Fitted Profile Points', markersize=4, alpha=0.7, linewidth=1.5)

    if fit_results['success']:
        # Fitted curve (extended over full range for visualization)
        x_smooth = np.linspace(np.min(x_full), np.max(x_full), 200)
        y_smooth = BFPr.Double_Gaussian(x_smooth, *list(p.values()))
        # plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Double Gaussian Fit')
        
        # Components - extending from zero instead of baseline
        comp1_from_zero = p['A1'] * np.exp(-0.5 * ((x_smooth - p['mu1']) / p['sigma1']) ** 2)+p['baseline']
        comp2_from_zero = p['A2'] * np.exp(-0.5 * ((x_smooth - p['mu2']) / p['sigma2']) ** 2)+p['baseline']
        comp_plot_coords_1 = comp1_from_zero>(p['baseline']+p['A2']*0.0001)
        comp_plot_coords_2 = comp2_from_zero>(p['baseline']+p['A2']*0.0001)
        ax0.plot(x_smooth[comp_plot_coords_1], comp1_from_zero[comp_plot_coords_1], \
                 c=color_1, linestyle='--', linewidth=2, label='Gauss Component 1')
        ax0.plot(x_smooth[comp_plot_coords_2], comp2_from_zero[comp_plot_coords_2], \
                 c=color_2, linestyle='--', linewidth=2, label='Gauss Component 2')
        
        if len(comp1_from_zero[comp_plot_coords_1]) != 0:
            peak_index = np.argmax(comp1_from_zero[comp_plot_coords_1])
            ax0.scatter(x_smooth[comp_plot_coords_1][peak_index], comp1_from_zero[comp_plot_coords_1][peak_index], \
                        color=color_1, s=50, marker='*',alpha=0.9, linewidth=1.5, label='Left Peak', zorder=5)
        if len(comp2_from_zero[comp_plot_coords_2]) != 0:
            peak_index = np.argmax(comp2_from_zero[comp_plot_coords_2])
            ax0.scatter(x_smooth[comp_plot_coords_2][peak_index], comp2_from_zero[comp_plot_coords_2][peak_index], \
                        color=color_2, s=50, marker='*',alpha=0.9, linewidth=1.5, label='Right Peak', zorder=5)
    
        bubble_diameter = abs(p['mu1']) + abs(p['mu2'])
        fwhm1 = 2.355 * p['sigma1']
        fwhm2 = 2.355 * p['sigma2']
        bubble_thickness = (fwhm1 + fwhm2) / 2
        half_height1 = p['A1']/2 + p['baseline']
        plt.annotate('', xy=(p['mu1'] - fwhm1/2, half_height1), xytext=(p['mu1'] + fwhm1/2, half_height1),
                    arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0, color=color_1, lw=1))
        ax0.plot((p['mu1'] - fwhm1/2, p['mu1'] + fwhm1/2), (half_height1, half_height1),\
                 color=color_1, linestyle='-', linewidth=1.5, label=f'FWHM$_1$: {fwhm1*pix_scale_arcmin:.2f}')
        
        half_height2 = p['A2']/2 + p['baseline']
        plt.annotate('', xy=(p['mu2'] - fwhm2/2, half_height2), xytext=(p['mu2'] + fwhm2/2, half_height2),
                    arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0, color=color_2, lw=1))
        ax0.plot((p['mu2'] - fwhm2/2, p['mu2'] + fwhm2/2), (half_height2, half_height2),\
                 color=color_2, linestyle='-', linewidth=1.5, label=f'FWHM$_2$: {fwhm2*pix_scale_arcmin:.2f}')
        
        plt.annotate('', xy=(p['mu2'], p['baseline']), xytext=(p['mu1'], p['baseline']),
                    arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,color='green', lw=1))
        ax0.plot((p['mu2'], p['mu1']), (p['baseline'], p['baseline']),\
                 color='green', linestyle='-', linewidth=1.5, label=f'Baseline&Diameter: {bubble_diameter*pix_scale_arcmin:.2f}')
    
        plt.annotate('', xy=(p['mu2']+bubble_thickness/2, p['baseline']-p['baseline']/6), \
                     xytext=(p['mu1']-bubble_thickness/2, p['baseline']-p['baseline']/6),
                     arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,color='orange', lw=1))
        ax0.plot((p['mu2']+bubble_thickness/2, p['mu1']-bubble_thickness/2), \
                 (p['baseline']-p['baseline']/6, p['baseline']-p['baseline']/6),\
                 color='orange', linestyle='-', linewidth=1.5, \
                 label=f'Outer Diameter: {(bubble_diameter+bubble_thickness)*pix_scale_arcmin:.2f}')
    
    # plt.axvline(x=p['mu2'], color='orange', linewidth=2, alpha=0.8)
    ax0.axvline(0, color='b', linestyle='dashed',alpha=0.3,label='Axis of Symmetry')

    ax0.text(0.8, 0.2, r'S$_{{\text{{sym}}}}$: {}'.format(np.around(fit_results['symmetry_score'],2)), 
                transform=ax0.transAxes, verticalalignment='top', fontsize=fontsize-2,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    mean_profile_left_coords = np.where(bubbleObj.mean_profile_left>0)[0]
    mean_profile_right_coords = np.where(bubbleObj.mean_profile_right>0)[0]
    max_len = np.max([mean_profile_left_coords[-1],\
                      mean_profile_right_coords[-1]]) + 5
    plt.xlim(-max_len,max_len)
    fig.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    ticks = ax0.get_xticks()*pix_scale_arcmin
    ax0.set_xticklabels(np.int32(np.around(ticks)))
    
    plt.xlabel("Radial Distance (arcmin)",fontsize=fontsize)
    plt.ylabel(r"Integrated Intensity (K)",fontsize=fontsize)
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.legend(fontsize=fontsize-4,loc='upper right')
    return ax0


def Plot_Contour_Ellipse_IOU(contour_ellipse_IOU,visualization,bubble_contour,ellipse_coords,figsize=(6, 4),fontsize=12,linewidth=2):
    fig,(ax1,ax2)= plt.subplots(1,2, figsize=figsize)
    ax1.set_title('Original Contour (Red) vs Ellipse (Green)', fontsize=fontsize)
    ax1.imshow(visualization)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1.invert_yaxis()
    ax1.axis('off')
    
    ax2.set_title('Contour and Ellipse', fontsize=fontsize)
    ax2.plot(bubble_contour[:,0], bubble_contour[:,1], 'r-', label='Original Contour',linewidth=linewidth)
    ax2.plot(ellipse_coords[:,0], ellipse_coords[:,1], 'g-', label='Ellipse',linewidth=linewidth)
    ax2.text(0.72, 0.1, f'CIOU: {contour_ellipse_IOU:.2f}', verticalalignment='top', fontsize=fontsize,
                    transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xticks([]),plt.yticks([])
    plt.tight_layout()

    return ax1,ax2
    

def Plot_Slice_Line_Coords(bubbleObj,line_coords_index,add_con=False,plot_clump=False,plot_line=True,text_com=True,\
                           ax0=None,figsize=(8,6),fontsize=14):
    bubble_clump_ids = bubbleObj.bubble_clump_ids 
    bubble_clump_ids_con = bubbleObj.bubble_clump_ids_con  
    bubble_coords = bubbleObj.bubble_coords 
    bubble_com_item = bubbleObj.bubble_com_item
    bubble_com_item_wcs = bubbleObj.bubble_com_item_wcs
    bubble_contour = bubbleObj.bubble_contour
    ellipse_infor = bubbleObj.ellipse_infor 
    ellipse_coords = bubbleObj.ellipse_coords 
    dictionary_cuts = bubbleObj.dictionary_cuts 
    # for line_coords_index in range(len(dictionary_cuts['lines_coords'])):
    line_coords = dictionary_cuts['lines_coords'][line_coords_index]
    mask_lines =  dictionary_cuts['mask_lines'][line_coords_index]
    line_coords = line_coords[mask_lines]
    
    if not add_con:
        clump_ids = list(bubble_clump_ids) 
        colors = ['red']*len(bubble_clump_ids)
    else :
        clump_ids = list(bubble_clump_ids) + list(bubble_clump_ids_con)
        colors = ['red']*len(bubble_clump_ids) + ['green']*len(bubble_clump_ids_con)
    bubble_item,start_coords,ax0 = Plot_Clumps_Infor_By_Ids(bubbleObj.clumpsObj,clump_ids,\
                                        ax0=ax0,figsize=figsize,fontsize=fontsize,plot_clump=plot_clump,colors=colors,line_scale=3)

    if plot_line:
        for i in range(len(line_coords)):
            ax0.plot(line_coords[i][0],line_coords[i][1],color='r',marker='.',markersize=4)
    
    ax0.scatter(bubble_com_item[2],bubble_com_item[1],color='yellow',marker='*',s=12,label="Cavity Com")
    ax0.plot(bubble_contour[:,1],bubble_contour[:,0],linewidth=2,color='green',label="Cavity Contours")
    
    ax0.scatter(ellipse_infor[1]-start_coords[2],ellipse_infor[0]-start_coords[1],color='orange',marker='*',s=12,label="Fit Cavity Center")
    ax0.plot(ellipse_coords[:,1]-start_coords[2],ellipse_coords[:,0]-start_coords[1],linewidth=2,color='cyan',label="Fit Cavity Ellipse")

    if text_com:
        ax0.text(bubble_item.shape[2]/20,bubble_item.shape[1]-bubble_item.shape[1]/15,r'Com = [{}, {}, {}]'.\
             format(bubble_com_item_wcs[0],bubble_com_item_wcs[1],bubble_com_item_wcs[2]),color='blue',fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(fontsize=fontsize,loc='upper right')
    return ax0


def Plot_Bubble_PV_Slice(bubbleObj,ax0=None,figsize=(6,4),fontsize=12):
    bubble_item_pv = bubbleObj.bubble_item_pv
    pv_path = bubbleObj.pv_path
    bubble_com_item = bubbleObj.bubble_com_item
    bubble_com_item_wcs = bubbleObj.bubble_com_item_wcs
    position_axis = bubbleObj.position_axis
    velocity_axis = bubbleObj.velocity_axis
    if ax0 is None:
        fig, (ax0) = plt.subplots(1,1,figsize=figsize)
    
    extent = [position_axis[0], position_axis[-1],
              velocity_axis[0], velocity_axis[-1]]

    vmin = np.min(bubble_item_pv[np.where(bubble_item_pv != 0)])
    vmax = np.nanpercentile(bubble_item_pv[np.where(bubble_item_pv != 0)], 98.)
    im = ax0.imshow(bubble_item_pv,
                   origin='lower',
                   cmap='gray',
                   aspect='auto',
                   extent=extent,
                   interpolation='bilinear',
                   norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    ax0.contourf(bubble_item_pv,
                 levels=[0., .01],extent=extent,
                 colors='w')
    
    X, Y = np.meshgrid(position_axis, velocity_axis)
    data_pos = bubble_item_pv[bubble_item_pv > 0]
    if len(data_pos) > 0:
        levels = np.linspace(np.percentile(data_pos, 20), np.percentile(data_pos, 90), 8)
        ax0.contour(X, Y, bubble_item_pv, levels=levels,
                  colors='cyan', linewidths=0.8, alpha=0.5,zorder=1)

    ax0.scatter(0,bubble_com_item_wcs[2],color='green',marker='o',s=40,label="Cavity Com",zorder=3)
    
    ax0.plot((0, 0), (bubbleObj.ranges_v_wcs[bubbleObj.index][0], bubbleObj.ranges_v_wcs[bubbleObj.index][1]),
            color='blue', linestyle='-.', linewidth=1.5, label='Cavity V Range',zorder=2)

    if hasattr(bubbleObj, 'bubble_params_RFWHM'):
        if 'bubble_params_type' in bubbleObj.bubble_params_RFWHM and bubbleObj.bubble_params_RFWHM['bubble_params_type'] == 1:
            ax0.plot((-bubbleObj.bubble_params_RFWHM['radius']/2, bubbleObj.bubble_params_RFWHM['radius']/2), 
                     (bubble_com_item_wcs[2], bubble_com_item_wcs[2]),
                    color='blue', linestyle='--', linewidth=1.5, label='Intensity Diameter',zorder=2)

    ax0.set_xlabel('Offset from Bubble Com (arcmin)', fontsize=fontsize)
    ax0.set_ylabel(r'V$_\text{LSR}$ (km s$^{-1}$)', fontsize=fontsize)
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'

    ax0.tick_params(axis='both', which='major', labelsize=fontsize)
    ax0.legend(loc='upper right', fontsize=fontsize)
    
    cbar = plt.colorbar(im, ax=ax0, shrink=1, pad=0)
    cbar.set_label('Intensity (K)', fontsize=fontsize)
    cbar.ax.tick_params(axis='y', colors='black', labelsize=fontsize)
    # plt.tight_layout()
    
    return ax0


def Plot_Bubble_Inner_Contours(bubbleObj,bg_data=1,bubble_valid_ids=None,plot_contour=True,num_text=True,\
               cmap='gray',ax0=None,figsize=(5,4),tick_logic=True,cbar_logic=True,linewidth=2,markersize=2,fontsize=12,spacing=None):
    
    data_wcs = bubbleObj.clumpsObj.data_wcs
    contours = bubbleObj.contours
    bubble_coms = bubbleObj.bubble_coms
    origin_data = bubbleObj.clumpsObj.origin_data
    regions_data = bubbleObj.clumpsObj.regions_data
    clumps_data = np.zeros_like(origin_data)
    clumps_data[regions_data>0] = origin_data[regions_data>0]
    bubble_weight_data = bubbleObj.bubble_weight_data
    bubble_regions_data = bubbleObj.bubble_regions_data

    if ax0 is None and tick_logic:
        fig = plt.figure(figsize=figsize)
        ax0 = fig.add_subplot(111,projection=data_wcs.celestial)
    else:
        fig, (ax0) = plt.subplots(1,1,figsize=figsize)
        ax0.set_xticks([]), ax0.set_yticks([])
    if tick_logic:
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
        if spacing != None:
            lon.set_ticks(spacing=spacing)
            lat.set_ticks(spacing=spacing)
        ax0.tick_params(axis='both', which='major', labelsize=fontsize)

    if plot_contour:
        colors_T = Distinct_Dark_Colors(len(bubble_coms))
        np.random.seed(0)
        colors_T = np.random.permutation(colors_T)
        if bubble_valid_ids is None:
            for i in range(len(contours)):
                ax0.plot(contours[i][:,1],contours[i][:,0],color=colors_T[i],linewidth=linewidth)
            for i in range(len(bubble_coms)):
                ax0.plot(bubble_coms[i][2],bubble_coms[i][1],color=colors_T[i],marker='o',markersize=markersize)
                if num_text==True:
                    ax0.text(bubble_coms[i][2],bubble_coms[i][1],"{}".format(i+1),color='r',fontsize=fontsize-2)
        else:
            for i in bubble_valid_ids:
                ax0.plot(contours[i][:,1],contours[i][:,0],color=colors_T[i],linewidth=linewidth)
            for i in bubble_valid_ids:
                ax0.plot(bubble_coms[i][2],bubble_coms[i][1],color=colors_T[i],marker='o',markersize=markersize)
                if num_text==True:
                    ax0.text(bubble_coms[i][2],bubble_coms[i][1],"{}".format(i+1),color='r',fontsize=fontsize-2)
                    
    if bg_data == 1:
        show_data = clumps_data.sum(0)*bubbleObj.vel_resolution
        cbar_name = 'K km s$^{-1}$'
    elif bg_data == 2:
        show_data = bubble_weight_data*(bubble_regions_data>0)
        show_data = show_data.sum(0)
        show_data[show_data != 0] = np.sqrt(show_data[show_data != 0])
        cbar_name = ''
        ax0.text(0.85, 0.1, r'$\sqrt{W_{l,b,v}}$', 
                transform=ax0.transAxes, verticalalignment='top', fontsize=fontsize-2,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    vmin = np.min(show_data[show_data != 0])
    vmax = np.nanpercentile(show_data[np.where(show_data != 0)], 99.5)
    
    gci = ax0.imshow(show_data,
               origin='lower',
               cmap=cmap,
               interpolation='none',
               norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    ax0.contourf(show_data,
                 levels=[0., .0001],
                 colors='w')
    if cbar_logic:
        cbar=plt.colorbar(gci,pad=0)
        cbar.ax.tick_params(labelsize=fontsize) 
        cbar.set_label(label=cbar_name,fontsize=fontsize)
    
    return ax0


def Plot_Radial_Velocity_Profile(bubbleObj,fontsize=12,ax0=None,figsize=(6,4)):
    pv_path = bubbleObj.pv_path
    bubble_item_pv = bubbleObj.bubble_item_pv
    bubble_position_arcmin = bubbleObj.bubble_position_arcmin
    position_axis = bubbleObj.position_axis
    velocity_axis = bubbleObj.velocity_axis
    intensity_weights = bubbleObj.intensity_weights
    expansion_analysis = bubbleObj.expansion_analysis
    bubble_outer_radius = bubbleObj.bubble_outer_radius
    velocity_dispersion = bubbleObj.velocity_dispersion
    weighted_velocity = bubbleObj.weighted_velocity
    bubble_com_item_wcs = bubbleObj.bubble_com_item_wcs
    wv_red = bubbleObj.wv_red
    wv_blue = bubbleObj.wv_blue
    iw_red = bubbleObj.iw_red
    iw_blue = bubbleObj.iw_blue
    systemic_v = bubbleObj.systemic_v
    central_v = bubbleObj.central_v
    
    if ax0 is None:
        fig, (ax0) = plt.subplots(1,1,figsize=figsize)
    
    used_ids = intensity_weights>0
    ax0.plot(position_axis[used_ids], weighted_velocity[used_ids], color='black',linestyle='-',linewidth=2,label='Weighted Velocity')
    ax0.fill_between(position_axis[used_ids], 
                   (weighted_velocity - velocity_dispersion)[used_ids],
                   (weighted_velocity + velocity_dispersion)[used_ids],
                   alpha=0.3, color='green', label='$\pm1\sigma$ Dispersion')

    used_ids = iw_red>0
    ax0.plot(position_axis[used_ids], wv_red[used_ids],color='red',linestyle='-.',\
            linewidth=1.5,label='Red Weighted Velocity')
    
    used_ids = iw_blue>0
    ax0.plot(position_axis[used_ids], wv_blue[used_ids],color='blue',linestyle='-.',\
            linewidth=1.5,label='Blue Weighted Velocity')

    plt.annotate('', xy=(-bubble_outer_radius/2, systemic_v), xytext=(bubble_outer_radius/2, systemic_v),
                arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,color='orange', lw=1))
    ax0.plot((-bubble_outer_radius/2, bubble_outer_radius/2), (systemic_v, systemic_v),\
             color='orange', linestyle='-', linewidth=1.5, label='Outer Diameter')
    # if show_band and np.isfinite(R_arcmin):
    ax0.axvspan(-bubble_outer_radius/2, bubble_outer_radius/2, color='orange', alpha=0.08, linewidth=0, zorder=0)
        
    ax0.axvline(x=0, color='b', linestyle='dashed', alpha=0.3)
    ax0.scatter(0,systemic_v,color='green',marker='x',s=50, alpha=1, \
               label=f'Systemic V: {systemic_v:.2f} '+'km s$^{-1}$',zorder=3)
    ax0.scatter(0,central_v,color='black',marker='x',s=50, alpha=1, \
               label=f'Central V: {central_v:.2f} '+'km s$^{-1}$',zorder=3)
    ax0.set_xlabel('Offset from Bubble Com (arcmin)', fontsize=fontsize)
    ax0.set_ylabel(r'V$_\text{LSR}$ (km s$^{-1}$)', fontsize=fontsize)
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    ax0.legend(fontsize=fontsize-4,loc='upper right')
    ax0.grid(True, alpha=0.3)
    ax0.tick_params(axis='both', which='major', labelsize=fontsize)
    return ax0


def Plot_Radial_Velocity_Profile_Mean(bubbleObj, fontsize=12, ax0=None, figsize=(6,4)):
    if ax0 is None:
        fig, (ax0) = plt.subplots(1,1,figsize=figsize)
    intensity_weights_record = bubbleObj.intensity_weights_record
    position_axis_record = bubbleObj.position_axis_record
    weighted_velocity_record = bubbleObj.weighted_velocity_record
    weighted_velocity_means = bubbleObj.weighted_velocity_means 
    weighted_velocity_mean_stds = bubbleObj.weighted_velocity_mean_stds 
    position_axis_mean = bubbleObj.position_axis_mean 
    pix_scale_arcmin = bubbleObj.pix_scale_arcmin
    bubble_outer_radius = bubbleObj.bubble_outer_radius
    bubble_outer_radius = bubble_outer_radius*pix_scale_arcmin

    weighted_velocity_mean      = weighted_velocity_means[0]
    weighted_velocity_mean_red  = weighted_velocity_means[1]
    weighted_velocity_mean_blue = weighted_velocity_means[2]
    weighted_velocity_mean_std      = weighted_velocity_mean_stds[0]
    weighted_velocity_mean_std_red  = weighted_velocity_mean_stds[1]
    weighted_velocity_mean_std_blue = weighted_velocity_mean_stds[2]

    expansion_analysis_mean      = bubbleObj.expansion_analysis_mean
    expansion_analysis_mean_red  = bubbleObj.expansion_analysis_mean_red
    expansion_analysis_mean_blue = bubbleObj.expansion_analysis_mean_blue
    systemic_v     = bubbleObj.systemic_v
    central_v_mean = bubbleObj.central_v_mean
    exp_central_delta_v_arg_max = bubbleObj.exp_central_delta_v_arg_max

    for i in range(len(position_axis_record)):
        position_axis = position_axis_record[i]
        weighted_velocity = weighted_velocity_record[i].copy()
        weighted_velocity[np.where(weighted_velocity == 0)] = np.nan
        ax0.plot(position_axis, weighted_velocity, c='gray', linewidth=1, alpha=0.2)
    ax0.plot(position_axis, weighted_velocity, c='gray', linewidth=1, alpha=0.2, label='Weighted Velocity (WV)')
    
    position_axis = position_axis_record[exp_central_delta_v_arg_max]
    weighted_velocity = weighted_velocity_record[exp_central_delta_v_arg_max]
    weighted_velocity[np.where(weighted_velocity == 0)] = np.nan
    ax0.plot(position_axis, weighted_velocity, c='black', linewidth=2, alpha=0.5, label='WV With Max $\Delta$V')

    wv_mean = weighted_velocity_mean.copy().astype(float)
    wv_mean[np.where(wv_mean == 0)] = np.nan
    ax0.plot(position_axis_mean, wv_mean, color='black', linestyle='-', linewidth=2, label='Mean WV')

    wv_mean_red = weighted_velocity_mean_red.copy().astype(float)
    wv_mean_red[(wv_mean_red == systemic_v) | (wv_mean_red == 0)] = np.nan
    ax0.plot(position_axis_mean, wv_mean_red, color='red', linestyle='-.', linewidth=1.5, label='Mean Red Velocity')

    wv_mean_blue = weighted_velocity_mean_blue.copy().astype(float)
    wv_mean_blue[(wv_mean_blue == systemic_v) | (wv_mean_blue == 0)] = np.nan
    ax0.plot(position_axis_mean, wv_mean_blue, color='blue', linestyle='-.', linewidth=1.5, label='Mean Blue Velocity')

    plt.annotate('', xy=(-bubble_outer_radius, systemic_v), xytext=(bubble_outer_radius, systemic_v),
                 arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0, color='orange', lw=1))
    ax0.plot((-bubble_outer_radius, bubble_outer_radius), (systemic_v, systemic_v),
            color='orange', linestyle='-', linewidth=1.5, label='Outer Diameter')
    ax0.axvspan(-bubble_outer_radius, bubble_outer_radius, color='orange', alpha=0.08, linewidth=0, zorder=0)
    
    ax0.axvline(x=0, color='b', linestyle='dashed', alpha=0.3)
    ax0.scatter(0, systemic_v, color='green', marker='x', s=50, alpha=1,
               label=f'Systemic V: {systemic_v:.2f} km s$^{{-1}}$', zorder=3)
    ax0.scatter(0, central_v_mean, color='black', marker='x', s=50, alpha=1,
               label=f'Central V: {central_v_mean:.2f} km s$^{{-1}}$', zorder=3)

    def _put_box(y, color_text, color_bbox, label_mean, label_max, analysis):
        if 'expansion_left' in analysis:
            exp_left = analysis['expansion_left'];     exp_right = analysis['expansion_right']
            exp_mean = (exp_left + exp_right)/2
            exp_left_max = analysis['expansion_left_max']; exp_right_max = analysis['expansion_right_max']
            exp_max = (exp_left_max + exp_right_max)/2
            ax0.text(0.14, y, f'{label_mean}={exp_mean:.2f} km s$^{{-1}}$', transform=ax0.transAxes,
                    ha='center', va='center', color=color_text, bbox=dict(boxstyle='round', facecolor=color_bbox, alpha=0.8),
                    fontsize=fontsize-3)
            ax0.text(0.86, y, f'{label_max}={exp_max:.2f} km s$^{{-1}}$', transform=ax0.transAxes,
                    ha='center', va='center', color=color_text, bbox=dict(boxstyle='round', facecolor=color_bbox, alpha=0.8),
                    fontsize=fontsize-3)
    _put_box(0.15, 'white','black', 'Mean', 'Max', expansion_analysis_mean)
    _put_box(0.25, 'black', 'red',  'Mean', 'Max', expansion_analysis_mean_red)
    _put_box(0.05, 'black', 'blue', 'Mean', 'Max', expansion_analysis_mean_blue)

    # ax0.text(0.14, 0.35, f'Com Type={bubbleObj.pv_type}',
    #         transform=ax0.transAxes, ha='center', va='center',
    #         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
    #         fontsize=fontsize-3)

    ax0.text(
        0.5, 0.25, 
        rf"S$_{{\text{{exp}}}}$(light black)={bubbleObj.exp_results['light black']['S_exp']:.2f}, "
        rf"{bubbleObj.exp_signs_light_black_str[0]}",
        transform=ax0.transAxes, ha='center', va='center', color='white',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),fontsize=fontsize - 3)

    ax0.text(
        0.5, 0.15, 
        rf"P:${bubbleObj.exp_sign_positive_mean_per:.3g}$, "
        rf"N:${bubbleObj.exp_sign_negative_mean_per:.3g}$, "
        rf"D:${bubbleObj.exp_sign_diff_mean_per:.3g}$",
        transform=ax0.transAxes, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='green', alpha=0.5),fontsize=fontsize - 3)

    colors = ["black", "red", "blue"]
    color_name = colors[bubbleObj.max_Sexp_index]
    ax0.text(0.5, 0.05,
            rf"S$_{{\text{{exp}}}}$({color_name})={bubbleObj.max_Sexp:.2f}",
            transform=ax0.transAxes, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.5),
            fontsize=fontsize-3)

    ax0.set_xlabel('Offset from Bubble Com (arcmin)', fontsize=fontsize)
    ax0.set_ylabel(r'V$_\text{LSR}$ (km s$^{-1}$)', fontsize=fontsize)
    ax0.legend(fontsize=fontsize-4, loc='upper right')
    ax0.grid(True, alpha=0.3)
    ax0.tick_params(axis='both', which='major', labelsize=fontsize)

    x_min, x_max = ax0.get_xlim()
    x_max_abs = np.max([x_min, x_max])
    new_x_max = x_max_abs + 0.1 * x_max_abs 
    ax0.set_xlim(-new_x_max, new_x_max)

    y_min, y_max = ax0.get_ylim()
    y_range = y_max - y_min
    new_y_min = y_min - 0.2 * y_range  
    new_y_max = y_max + 0.1 * y_range 
    ax0.set_ylim(new_y_min, new_y_max)

    return ax0


def Plot_Radial_Velocity_Model_Fitting(bubbleObj,fontsize=12,ax0=None,figsize=(6,4)):
    position_axis = bubbleObj.position_axis
    velocity_axis = bubbleObj.velocity_axis
    intensity_weights = bubbleObj.intensity_weights
    bubble_outer_radius = bubbleObj.bubble_outer_radius
    weighted_velocity = bubbleObj.weighted_velocity
    models = bubbleObj.models 
    best_model_key = bubbleObj.best_model_key 
    
    if ax0 is None:
        fig, (ax0) = plt.subplots(1,1,figsize=figsize)
    used_ids = intensity_weights>0
    pos_plot = position_axis[used_ids]
    vel_plot = weighted_velocity[used_ids]
    
    ax0.scatter(pos_plot, vel_plot, c='black', s=20, alpha=0.7, label='Observed')
    ax0.axvline(x=0, color='b', linestyle='dashed', alpha=0.3)
    
    if models and best_model_key:
        # Show best fit model
        best_model = models[best_model_key]
        pos_model = np.linspace(-bubble_outer_radius/2, bubble_outer_radius/2, 100)
        vel_model = best_model['function'](pos_model, *best_model['params'])
        
        ax0.plot(pos_model, vel_model, 'r-', linewidth=2, 
               label=f"Best Fit: {best_model['name']}\nR$^2$ = {best_model['r2']:.3f}")
        
        # Show other models if available
        colors = ['orange', 'purple']
        color_idx = 0
        for key, model in models.items():
            if key != best_model_key and color_idx < len(colors):
                vel_alt = model['function'](pos_model, *model['params'])
                ax.plot(pos_model, vel_alt, '--', color=colors[color_idx], linewidth=1.5,
                       alpha=0.7, label=f"{model['name']} (R$^2$={model['r2']:.3f})")
                color_idx += 1
    
    ax0.set_xlabel('Position (arcmin)', fontsize=fontsize)
    ax0.set_ylabel(r'V$_\text{LSR}$ (km s$^{-1}$)', fontsize=fontsize)
    # ax.set_title('Model Fitting', fontsize=fontsize, fontweight='bold')
    ax0.legend(fontsize=fontsize-2)
    ax0.grid(True, alpha=0.3)
    ax0.tick_params(axis='both', which='major', labelsize=fontsize)
    return ax0
    

def Plot_Radial_Velocity_Gradient(bubbleObj,fontsize=12,ax0=None,figsize=(6,4)):
    position_axis = bubbleObj.position_axis
    velocity_axis = bubbleObj.velocity_axis
    intensity_weights = bubbleObj.intensity_weights
    bubble_outer_radius = bubbleObj.bubble_outer_radius
    weighted_velocity = bubbleObj.weighted_velocity
    velocity_dispersion = bubbleObj.velocity_dispersion
    
    used_ids = intensity_weights>0
    pos_plot = position_axis[used_ids]
    vel_plot = weighted_velocity[used_ids]

    if ax0 is None:
        fig, (ax0) = plt.subplots(1,1,figsize=figsize)
    if len(pos_plot) > 3:
        gradients = []
        gradient_positions = []
        
        window_size = max(3, len(pos_plot) // 10)
        
        for i in range(window_size, len(pos_plot) - window_size):
            start_idx = i - window_size
            end_idx = i + window_size
            
            pos_window = pos_plot[start_idx:end_idx]
            vel_window = vel_plot[start_idx:end_idx]
            
            if len(pos_window) > 2:
                gradient = np.polyfit(pos_window, vel_window, 1)[0]
                gradients.append(gradient)
                gradient_positions.append(pos_plot[i])
        
        if gradients:
            ax0.plot(gradient_positions, gradients, 'purple', linewidth=2, 
                   marker='o', markersize=3, label='Local Velocity Gradient')
            ax0.axhline(y=0, color='black', linestyle='-', alpha=0.8)
            ax0.axvline(x=0, color='red', linestyle='-', alpha=0.6)
            
            mean_gradient = np.mean(gradients)
            ax0.axhline(y=mean_gradient, color='orange', linestyle='--', 
                      label=f'Mean: {mean_gradient:.3f} km/s/arcmin')
    
    ax0.set_xlabel('Position (arcmin)', fontsize=fontsize)
    ax0.set_ylabel('dv/dr (km/s/arcmin)', fontsize=fontsize)
    # ax.set_title('Velocity Gradient', fontsize=fontsize, fontweight='bold')
    ax0.legend(fontsize=fontsize-2)
    ax0.grid(True, alpha=0.3)
    ax0.tick_params(axis='both', which='major', labelsize=fontsize)
    return ax0
    

def Plot_Radial_Velocity_Dispersion(bubbleObj,fontsize=12,ax0=None,figsize=(6,4)):
    position_axis = bubbleObj.position_axis
    velocity_axis = bubbleObj.velocity_axis
    intensity_weights = bubbleObj.intensity_weights
    bubble_outer_radius = bubbleObj.bubble_outer_radius
    weighted_velocity = bubbleObj.weighted_velocity
    velocity_dispersion = bubbleObj.velocity_dispersion
    expansion_analysis = bubbleObj.expansion_analysis
    used_ids = intensity_weights>0
    
    if ax0 is None:
        fig, (ax0) = plt.subplots(1,1,figsize=figsize)

    # Velocity dispersion vs position
    ax0.plot(position_axis[used_ids], velocity_dispersion[used_ids], 'purple', linewidth=2, 
           marker='o', markersize=4, label='Velocity Dispersion')
    
    # Mark high turbulence regions
    high_turb_threshold = np.mean(velocity_dispersion) + np.std(velocity_dispersion)
    high_turb_mask = velocity_dispersion > high_turb_threshold
    
    if np.any(high_turb_mask):
        ax0.scatter(position_axis[high_turb_mask], velocity_dispersion[high_turb_mask],
                  c='red', s=50, marker='^', label='High Turbulence', zorder=5)
    
    ax0.axvline(x=0, color='red', linestyle='-', alpha=0.6)
    mean_disp = expansion_analysis['mean_dispersion']
    ax0.axhline(y=mean_disp, color='orange', linestyle='--', 
              label=f'Mean: {mean_disp:.2f} '+'km s$^{-1}$')
    
    ax0.set_xlabel('Position (arcmin)', fontsize=12)
    ax0.set_ylabel('Velocity Dispersion (km s$^{-1}$)', fontsize=fontsize)
    ax0.set_title('Turbulence Distribution', fontsize=fontsize, fontweight='bold')
    ax0.legend(fontsize=fontsize-2)
    ax0.grid(True, alpha=0.3)
    ax0.tick_params(axis='both', which='major', labelsize=fontsize)
    return ax0
    

def Plot_Kinematic_Text(bubbleObj,fontsize=12,ax0=None,figsize=(6,4)):
    expansion_analysis = bubbleObj.expansion_analysis
    models = bubbleObj.models
    best_model_key = bubbleObj.best_model_key
    if ax0 is None:
        fig, (ax0) = plt.subplots(1,1,figsize=figsize)
    ax0.axis('off')
    
    # Create analysis report
    report_text = "=== BUBBLE KINEMATIC ANALYSIS ===\n\n"
    
    if expansion_analysis:
        report_text += f"Target Parameters:\n"
        if 'classification' in expansion_analysis:
            report_text += f"  Type: {expansion_analysis['classification']}\n"
            report_text += f"  Confidence: {expansion_analysis['confidence']}\n"
        
        if 'mean_expansion' in expansion_analysis:
            report_text += f"  Expansion vel: {expansion_analysis['mean_expansion']:.2f} km/s\n"
            report_text += f"  Asymmetry: {expansion_analysis['asymmetry']:.2f}\n"
        
        report_text += f"\nTurbulence Features:\n"
        if 'mean_dispersion' in expansion_analysis:
            report_text += f"  Mean dispersion: {expansion_analysis['mean_dispersion']:.2f} km/s\n"
            report_text += f"  Impact level: {expansion_analysis['turbulence_impact']}\n"
        
        report_text += f"\nPhysical Meaning:\n"
        report_text += f"  • Systemic vel: {expansion_analysis['systemic_velocity']:.2f} km/s\n"
        report_text += f"  • Main process: bubble expansion\n"
        report_text += f"  • Driver: internal energy source\n"
        
        if models and best_model_key:
            best_model = models[best_model_key]
            report_text += f"\nBest Model:\n"
            report_text += f"  {best_model['formula']}\n"
            report_text += f"  Fit quality: {best_model['r2']:.3f}\n"
    
    ax0.text(0.05, 0.95, report_text, transform=ax0.transAxes,
           verticalalignment='top', fontsize=fontsize, fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9))
    
    plt.tight_layout()
    return ax0


def Plot_Unwrap_Bubble_Infor(bubbleObj,add_con=False,plot_clump=False,plot_ellipse=True,plot_contour=True,
                             plot_circles=[True,True],plot_annotate=True,plot_skeleton=False,plot_skeleton_ellipse=False,
                             tick_logic=True,cbar_logic=True,text_com=False,
                             skeleton_types=['scatter','line'],linewidth=1.5,ax0=None,figsize=(8,6),fontsize=14):
    angle_offset = bubbleObj.angle_offset
    bubble_clump_ids = bubbleObj.bubble_clump_ids
    bubble_clump_ids_con = bubbleObj.bubble_clump_ids_con
    bubble_com_item = bubbleObj.bubble_com_item
    bubble_contour = bubbleObj.bubble_contour
    ellipse_coords = bubbleObj.ellipse_coords
    ellipse_infor = bubbleObj.ellipse_infor
    start_coords_item = bubbleObj.start_coords
    unwrap_width = bubbleObj.unwrap_width
    unwrap_radius = bubbleObj.unwrap_radius
    skeleton_coords_original = bubbleObj.skeleton_coords_original 
    skeleton_coords_ellipse = bubbleObj.skeleton_coords_ellipse 
    
    clump_ids = list(bubble_clump_ids) 
    if not add_con:
        clump_ids = list(bubble_clump_ids) 
        colors = ['red']*len(bubble_clump_ids)
        bubble_gas_com = bubbleObj.bubble_gas_com_wcs_1
        bubble_gas_ranges_lbv_min = bubbleObj.bubble_gas_ranges_lbv_min_1
        bubble_gas_ranges_lbv_max = bubbleObj.bubble_gas_ranges_lbv_max_1
    else :
        clump_ids = list(bubble_clump_ids) + list(bubble_clump_ids_con)
        colors = ['red']*len(bubble_clump_ids) + ['blue']*len(bubble_clump_ids_con)
        bubble_gas_com = bubbleObj.bubble_gas_com_wcs_2
        bubble_gas_ranges_lbv_min = bubbleObj.bubble_gas_ranges_lbv_min_2
        bubble_gas_ranges_lbv_max = bubbleObj.bubble_gas_ranges_lbv_max_2
    
    bubble_item,start_coords,ax0 = Plot_Clumps_Infor_By_Ids(bubbleObj.clumpsObj,clump_ids,tick_logic=tick_logic,cbar_logic=cbar_logic,\
                                ax0=ax0,figsize=figsize,fontsize=fontsize,plot_clump=plot_clump,colors=colors,num_text=False,line_scale=3)
    
    bubble_com_item_i = np.array([bubble_com_item[1]+start_coords_item[1]-start_coords[1],\
                                        bubble_com_item[2]+start_coords_item[2]-start_coords[2]])
    ax0.scatter(bubble_com_item_i[1],bubble_com_item_i[0],\
                 color='green',marker='o',s=20,label="Cavity Com")
    if plot_contour:
        bubble_contour_i = np.c_[bubble_contour[:,0]+start_coords_item[1]-start_coords[1],\
                                    bubble_contour[:,1]+start_coords_item[2]-start_coords[2]]
        ax0.plot(bubble_contour_i[:,1],bubble_contour_i[:,0],\
                 linewidth=2,color='green',label="Cavity Contour",alpha=0.6)
    
    if plot_ellipse:
        ellipse_coords_i = np.c_[ellipse_coords[:,0]-start_coords[1],ellipse_coords[:,1]-start_coords[2]]
        ax0.scatter(ellipse_infor[1]-start_coords[2],ellipse_infor[0]-start_coords[1],color='cyan',marker='*',s=20,label="Fited Cavity Center")
        ax0.plot(ellipse_coords_i[:,1],ellipse_coords_i[:,0],linewidth=linewidth,color='cyan',label="Fited Cavity Ellipse")
    
    if plot_circles[0]:
        bubble_circle = Circle((bubble_com_item_i[1], bubble_com_item_i[0]), unwrap_radius, fill=False, 
                       color='green', linewidth=linewidth, linestyle='-.', label='Radius Circle')
        
        ax0.add_patch(bubble_circle)
    if plot_circles[1]:
        outer_circle = Circle((bubble_com_item_i[1], bubble_com_item_i[0]), unwrap_radius + unwrap_width, fill=False, 
                           color='yellow', linewidth=linewidth, linestyle='-.', label='Outer Unwrap Ellipse')
        inner_circle = Circle((bubble_com_item_i[1], bubble_com_item_i[0]), np.max([0.5,unwrap_radius - unwrap_width]), fill=False, 
                         color='yellow', linewidth=linewidth, linestyle='--', label='Inner Unwrap Ellipse')
        ax0.add_patch(outer_circle)
        ax0.add_patch(inner_circle)
        
    # Add arrow showing direction
    if plot_annotate:
        arrow_x_1 = bubble_com_item_i[1] + np.max([0.5,unwrap_radius - unwrap_width])*np.cos(np.radians(angle_offset))
        arrow_x_2 = bubble_com_item_i[1] + (unwrap_radius+unwrap_width)*np.cos(np.radians(angle_offset))
        arrow_y_1 = bubble_com_item_i[0] + np.max([0.5,unwrap_radius - unwrap_width])*np.sin(np.radians(angle_offset))
        arrow_y_2 = bubble_com_item_i[0] + (unwrap_radius+unwrap_width)*np.sin(np.radians(angle_offset))
        plt.annotate('', xy=(arrow_x_2, arrow_y_2), xytext=(arrow_x_1, arrow_y_1),
                           arrowprops=dict(arrowstyle='->', shrinkA=0, shrinkB=0,color='orange', lw=2))
        ax0.plot((arrow_x_1, arrow_x_2),(arrow_y_1, arrow_y_2),color='orange',linestyle='-',linewidth=1.5,label='Unwrap Line')

    if plot_skeleton and skeleton_types[0]=='line':
        ax0.plot(skeleton_coords_original[:,1]-start_coords[2],
                 skeleton_coords_original[:,0]-start_coords[1],
                 linewidth=linewidth,color='red',linestyle='-',label="Intensity Skeleton")
    elif plot_skeleton and skeleton_types[0]=='scatter':
        ax0.scatter(skeleton_coords_original[:,1]-start_coords[2],
                    skeleton_coords_original[:,0]-start_coords[1],
                    marker='.',s=20,color='red',label="Intensity Skeleton")
    if plot_skeleton_ellipse and skeleton_types[1]=='line':
        ax0.scatter(bubbleObj.skeleton_ellipse_infor[1]-start_coords[2],bubbleObj.skeleton_ellipse_infor[0]-start_coords[1],\
                    color='lime',marker='*',s=20,label="Fited Intensity Center")
        ax0.plot(skeleton_coords_ellipse[:,1]-start_coords[2],
                 skeleton_coords_ellipse[:,0]-start_coords[1],
                 linewidth=linewidth,color='lime',linestyle='-',label="Fited Intensity Skeleton")
    elif plot_skeleton_ellipse and skeleton_types[1]=='scatter':
        ax0.scatter(bubbleObj.skeleton_ellipse_infor[1]-start_coords[2],bubbleObj.skeleton_ellipse_infor[0]-start_coords[1],\
                    color='lime',marker='*',s=20,label="Fited Intensity Center")
        ax0.scatter(skeleton_coords_ellipse[:,1]-start_coords[2],
                    skeleton_coords_ellipse[:,0]-start_coords[1],
                    marker='.',s=20,color='lime',label="Fited Intensity Skeleton")

    if text_com:
        ax0.text( 0.05, 0.88, r'Gas Com = [{}$^\circ$, {}$^\circ$, {}km s $^{{-1}}$]'.format(
                             bubble_gas_com[0],bubble_gas_com[1],bubble_gas_com[2]),\
                             color='green',transform=ax0.transAxes, verticalalignment='top', fontsize=10)
        ax0.text(0.05, 0.81, r'Gas V Range = ({}$\sim${}) km s$^{{-1}}$'.format(bubble_gas_ranges_lbv_min[2],bubble_gas_ranges_lbv_max[2]),\
                             color='green',transform=ax0.transAxes, verticalalignment='top', fontsize=10)
                
    plt.legend(loc='upper right',labelcolor='black',fontsize=fontsize-2)
    return ax0


def Plot_Ellipse_Width_Region(ax, ellipse_x0, ellipse_y0, ellipse_angle, semi_major, semi_minor, \
                              width, target_angle, n_samples=1000, linewidth=1):

    ellipse_angle = np.radians(90-ellipse_angle)
    cos_rot = np.cos(ellipse_angle)
    sin_rot = np.sin(ellipse_angle)
    t = np.linspace(0, 2*np.pi, n_samples, endpoint=True)
    
    # ---------------------- 1. 椭圆中心线坐标（全局） ----------------------
    # 椭圆局部坐标（中心在原点，未旋转）
    x_local = semi_major * np.cos(t)
    y_local = semi_minor * np.sin(t)
    
    # 转换到全局坐标（中心+旋转）
    x_ellipse = ellipse_x0 + cos_rot * x_local - sin_rot * y_local
    y_ellipse = ellipse_y0 + sin_rot * x_local + cos_rot * y_local
    
    # ---------------------- 2. 阴影区域内外边界计算 ----------------------
    # 椭圆切向量（局部）
    dx_dt = -semi_major * np.sin(t)
    dy_dt = semi_minor * np.cos(t)
    tangent_len = np.sqrt(dx_dt**2 + dy_dt**2)
    # 法向量（局部，向外）
    nx_l_outer = dy_dt / tangent_len
    ny_l_outer = -dx_dt / tangent_len

    x_l_outer = x_local + width * nx_l_outer
    y_l_outer = y_local + width * ny_l_outer
    x_outer = ellipse_x0 + cos_rot * x_l_outer - sin_rot * y_l_outer
    y_outer = ellipse_y0 + sin_rot * x_l_outer + cos_rot * y_l_outer
    
    x_l_inner = x_local - width * nx_l_outer
    y_l_inner = y_local - width * ny_l_outer
    x_inner = ellipse_x0 + cos_rot * x_l_inner - sin_rot * y_l_inner
    y_inner = ellipse_y0 + sin_rot * x_l_inner + cos_rot * y_l_inner
    
    x_end_outer, y_end_outer = BFS.Get_Expanded_Boundary_Intersection(
        target_angle, ellipse_x0, ellipse_y0, ellipse_angle,
        semi_major, semi_minor, width, is_outer=True)
    
    x_end_inner, y_end_inner = BFS.Get_Expanded_Boundary_Intersection(
        target_angle, ellipse_x0, ellipse_y0, ellipse_angle,
        semi_major, semi_minor, width, is_outer=False)

    ax.fill(np.concatenate([x_outer, x_inner[::-1]]), 
            np.concatenate([y_outer, y_inner[::-1]]), 
            color='lightblue', alpha=0.3, label='Unwrap Region')

    ax.annotate('', xy=(x_end_outer, y_end_outer), xytext=(x_end_inner, y_end_inner),
                arrowprops=dict(arrowstyle='->', shrinkA=0, shrinkB=0, color='orange', lw=linewidth))
    ax.plot([x_end_inner, x_end_outer], [y_end_inner, y_end_outer],
            color='orange', linestyle='-', linewidth=linewidth, label='Unwrap Line')
    
    ax.plot(x_ellipse, y_ellipse, color='darkblue', linewidth=linewidth+0.5, label='Unwrap Ellipse')
    ax.plot(x_outer, y_outer, color='yellow', linestyle='--', linewidth=linewidth, label='Outer Boundary')
    ax.plot(x_inner, y_inner, color='yellow', linestyle='-', linewidth=linewidth, label='Inner Boundary')
    
    return ax
    

def Plot_Unwrapped_Bub_Data(bubbleObj,fontsize=12,ax0=None,figsize=(15,2)):
    pix_scale_arcmin = bubbleObj.pix_scale_arcmin
    vel_resolution = bubbleObj.vel_resolution
    unwrapped_bub_data = bubbleObj.unwrapped_bub_data 
    skeleton_coords_unwrapped = bubbleObj.skeleton_coords_unwrapped
    unwrap_width = bubbleObj.unwrap_width
    
    if ax0 is None:
        fig, (ax0) = plt.subplots(1,1,figsize=figsize)
    
    skeleton_coords = skeleton_coords_unwrapped
    for i in range(len(skeleton_coords)):
        ax0.plot(skeleton_coords[i][1],skeleton_coords[i][0],color='r',marker='.',markersize=4)
    
    gci = ax0.imshow(unwrapped_bub_data*vel_resolution,
               origin='lower',
               cmap='gray',
               interpolation='none')
    ax0.contourf(unwrapped_bub_data*vel_resolution,
                 levels = [0., .01],
                 colors = 'w')
    
    cbar=plt.colorbar(gci,pad=0)
    cbar.ax.tick_params(labelsize=fontsize) 
    cbar_name = 'K km s$^{-1}$'
    cbar.set_label(label=cbar_name,fontsize=fontsize)
    
    ax0.set_xlabel('Azimuth Angle (degrees)',fontsize=fontsize)
    ax0.set_ylabel('Radial D (arcmin)',fontsize=fontsize)
    ax0.tick_params(axis='both', which='major',labelsize=fontsize)
    fig.tight_layout()
    
    xticks = ax0.get_xticks()
    xticks = xticks/xticks.max()*360
    ax0.set_xticklabels(np.int32(np.around(xticks)))
    
    yticks = ax0.get_yticks()
    yticks_arcmin = (yticks - unwrap_width) * pix_scale_arcmin
    ax0.set_yticklabels([f"{y:.1f}" for y in yticks_arcmin])
    return ax0


def Plot_Bubble_Flow_Imgs(bubbleObj, index=None, fontsize=12, img_name='N19-1',save=True, save_folder='../Images', show=False):
    print(img_name)

    # ===== 1) 创建画布 + 三个子图（ax0 带 WCS 投影）=====
    fig = plt.figure(figsize=(20, 6), constrained_layout=True)

    # 第一个：WCS projection
    ax0 = fig.add_subplot(1, 3, 1, projection=bubbleObj.data_wcs_item.celestial)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    # ===== 2) (c) Radial Velocity Profile =====
    ax2.tick_params(axis='both', colors='black')
    Plot_Radial_Velocity_Profile_Mean(bubbleObj, ax0=ax2)
    ax2.text(0.05, 0.95, '(c)', color='black', transform=ax2.transAxes,
             va='top', fontsize=20)

    # ===== 3) (b) PV Slice =====
    line_index = bubbleObj.exp_central_delta_v_arg_max
    BFPV.Cal_Radial_Velocity_Profile(bubbleObj, line_index, width=2)

    Plot_Bubble_PV_Slice(bubbleObj, ax0=ax1)
    leg1 = ax1.legend(loc='upper right', fontsize=fontsize-2)
    legend_style = [('green', 1.0), ('blue', 0.5)]
    for text, (color, alpha) in zip(leg1.get_texts(), legend_style):
        text.set_color(color)
        text.set_alpha(alpha)

    ax1.text(0.05, 0.95, '(b)', color='black', transform=ax1.transAxes,
             va='top', fontsize=20)

    # ===== 4) (a) Unwrap Bubble (WCS) =====
    ax0.tick_params(axis='both', colors='green')

    Plot_Unwrap_Bubble_Infor(
        bubbleObj,
        add_con=True, plot_clump=True, plot_ellipse=False, plot_contour=False,
        plot_circles=[False, False], plot_annotate=False, plot_skeleton=False,
        plot_skeleton_ellipse=True, tick_logic=True, cbar_logic=True, text_com=True,
        skeleton_types=['scatter', 'line'], linewidth=1.5, fontsize=12, ax0=ax0
    )

    dictionary_cuts_item = copy.deepcopy(bubbleObj.dictionary_cuts)
    start = dictionary_cuts_item['plot_cuts'][line_index][0]
    end   = dictionary_cuts_item['plot_cuts'][line_index][1]

    ax0.plot([start[0], end[0]], [start[1], end[1]],
             color='red', label="PV Slice",
             linestyle='-', markersize=8., linewidth=1.2, alpha=0.6)

    leg0 = ax0.legend(loc='upper right', fontsize=fontsize-2)
    legend_style = [('green', 1.0), ('lime', 1.0), ('lime', 1.0), ('red', 0.6)]
    for text, (color, alpha) in zip(leg0.get_texts(), legend_style):
        text.set_color(color)
        text.set_alpha(alpha)

    parts = img_name.split('_')
    tag = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else img_name
    ax0.text(0.05, 0.95, f'(a) {tag}', color='black',
             transform=ax0.transAxes, va='top', fontsize=20)

    if save and save_folder:
        save_path_pdf = f"{save_folder}/Combined_Plot_{img_name}.pdf"
        save_path_png = f"{save_folder}/Combined_Plot_{img_name}.png"
        fig.savefig(save_path_pdf, format='pdf', dpi=500, bbox_inches='tight')
        fig.savefig(save_path_png, format='png', dpi=500, bbox_inches='tight')

    if show:
        plt.show()

    plt.close(fig)


def Plot_Bubble_Flow_Imgs_Separate(bubbleObj,index,fontsize=12,img_name='N19-1',save=True,save_folder='../Images',show=False):

    print(img_name)
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    ax0 = BFPl.Plot_Radial_Velocity_Profile_Mean(bubbleObj,figsize=(6,4))
    ax0.text(0.05, 0.95, '(c)', color='black',transform=ax0.transAxes, verticalalignment='top', fontsize=20)
    save_path_pdf = save_folder + '/Imgs_PV/Velocity_Profiles_{}.pdf'.format(img_name)
    save_path_png = save_folder + '/Imgs_PV/Velocity_Profiles_{}.png'.format(img_name)
    if save_path_pdf != None and save:
        plt.savefig(save_path_pdf, format='pdf', dpi=500)
        plt.savefig(save_path_png, format='png', dpi=500)
    if show:
        plt.show()
    
    line_index = bubbleObj.exp_central_delta_v_arg_max
    BFPV.Cal_Radial_Velocity_Profile(bubbleObj,line_index,width=2)
    ax0 = Plot_Bubble_PV_Slice(bubbleObj)
    # ax0.set_ylim(bubble_gas_lbv_min_2_group[2],bubble_gas_lbv_max_2_group[2])
    # ax0.axhline(bubble_com_item_wcs_group[2], color='b', linestyle='dashed',alpha=0.5,label='Systemic V of N19 Group')

    legend = ax0.legend(loc='upper right',fontsize=fontsize-2)
    legend_style = [('green', 1.0),('blue', 0.5)]
    for text, (color, alpha) in zip(legend.get_texts(), legend_style):
        text.set_color(color)
        text.set_alpha(alpha)
        
    ax0.text(0.05, 0.95, '(b)', color='black',transform=ax0.transAxes, verticalalignment='top', fontsize=20)
    save_path_pdf = save_folder + '/Imgs_PV/PV_Slice_{}.pdf'.format(img_name)
    save_path_png = save_folder + '/Imgs_PV/PV_Slice_{}.png'.format(img_name)
    if save_path_pdf != None and save:
        plt.savefig(save_path_pdf, format='pdf', dpi=500)
        plt.savefig(save_path_png, format='png', dpi=500)
    if show:
        plt.show()
    
    plt.rcParams['xtick.color'] = 'green'
    plt.rcParams['ytick.color'] = 'green'
    ax0 = Plot_Unwrap_Bubble_Infor(bubbleObj,add_con=True,plot_clump=True,plot_ellipse=False,plot_contour=False,\
                                        plot_circles=[False,False],plot_annotate=False,plot_skeleton=False,plot_skeleton_ellipse=True,\
                                        tick_logic=True,cbar_logic=True,text_com=True,\
                                        skeleton_types=['scatter','line'],linewidth=1.5,figsize=(8,6),fontsize=12)
    dictionary_cuts_item = copy.deepcopy(bubbleObj.dictionary_cuts)
    start = dictionary_cuts_item['plot_cuts'][line_index][0]
    end = dictionary_cuts_item['plot_cuts'][line_index][1]
    ax0.plot([start[0], end[0]], [start[1], end[1]], color='red',label="PV Slice",linestyle='-', markersize=8., \
             linewidth=1.2, alpha=0.6)
    legend = ax0.legend(loc='upper right',fontsize=fontsize-2)
    legend_style = [('green', 1.0),('lime', 1.0),('lime', 1.0),('red', 0.6)]
    for text, (color, alpha) in zip(legend.get_texts(), legend_style):
        text.set_color(color)
        text.set_alpha(alpha)

    ax0.text(0.05, 0.95, r'(a) {}-{}'.format(img_name.split('_')[0],img_name.split('_')[1]), \
             color='black',transform=ax0.transAxes, verticalalignment='top', fontsize=20)
    save_path_pdf = save_folder + '/Imgs_Skt/Fited_Skt_{}.pdf'.format(img_name)
    save_path_png = save_folder + '/Imgs_Skt/Fited_Skt_{}.png'.format(img_name)
    if save_path_pdf != None and save:
        plt.savefig(save_path_pdf, format='pdf', dpi=500)
        plt.savefig(save_path_png, format='png', dpi=500)
    if show:
        plt.show()











