import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from collections import defaultdict

# Import custom modules related to Bubble analysis
from . import Bubble_Class
from . import Bubble_Funs_Morphology as BFM
from . import Bubble_Funs_PV as BFPV      
from . import Bubble_Funs_Table as BFT  
from . import Bubble_Funs_Profile as BFPr
from . import Bubble_Funs_Skeleton as BFS
from . import Bubble_Funs_Plot as BFPl
from . import Bubble_Funs_Tools as BFTools
from . import MeerKAT_Code as MKCode


def Bubble_Detect_Determine_By_MorProVel_Table(bubbleObj, bub_ids=None, bubble_infor_provided=None, add_cons=[False, True], 
                                               fixed_ec=True, half_ecoords=True, systemic_v_type=1, SymScore=0.8, ExpSign=1, 
                                               pv_type=None, pv_width=2, unwrap_radius=None, unwrap_width=None, angle_offset=None, 
                                               plot_logic=False, update_contour=False,mean_line_num=2):
    """
    Detect and determine bubble properties using morphology, profile fitting, and velocity analysis.

    Parameters
    ----------
    bubbleObj : BubbleClass
        The bubble object containing bubble data.
    bub_ids : list or None
        List of bubble indices to process. If None, all bubbles will be processed.
    bubble_infor_provided : dict or None
        Optional bubble information to be provided.
    add_cons : list of bool
        Whether to add additional constraints to the bubble analysis.
    fixed_ec : bool
        Whether to fix the expansion center during the analysis.
    half_ecoords : bool
        Whether to use half coordinates for the bubble analysis.
    systemic_v_type : int
        Type of systemic velocity.
    SymScore : float
        Symmetry score threshold for considering a bubble valid.
    ExpScore : float
        Expansion score threshold for considering a bubble valid.
    pv_type : str or None
        Type of profile velocity to analyze.
    pv_width : int
        Width of the velocity profile.
    unwrap_radius : float or None
        Radius for unwrapping the bubble skeleton.
    unwrap_width : float or None
        Width for unwrapping the bubble skeleton.
    angle_offset : float or None
        Offset angle for the bubble skeleton unwrapping.
    plot_logic : bool
        Whether to generate plots during the analysis.
    update_contour : bool
        Whether to update the contour and ellipse information after analysis.
    mean_line_num : int
        The minimum intensity profile number for a point on mean intensity profile
    """
    # Initialize variables to store data
    bubble_used_ids = []
    bubble_high_exp_ids = []
    skeleton_ellipse_infor_record = []
    skeleton_coords_ellipse_record = []
    exp_sign_pers = []
    bubble_gas_coms_wcs_2 = []
    thicknesss = []
    thicknesss_error = []
    skeleton_ellipse_coms_wcs = []
    skeleton_ellipse_angle_abs = []
    exp_maxs = []
    sym_scores = []
    exp_significances = []
    
    bubble_coms = bubbleObj.bubble_coms

    updata_records = False
    if bub_ids is None:
        updata_records = True
        bub_ids = np.arange(len(bubble_coms))
        
    for index in bub_ids:
        print('Index:', index + 1)
        add_con = add_cons[0]
        dictionary_cuts = defaultdict(list)
        
        # Get bubble morphology and fit the profile
        bubbleObj.Get_Bubble_Item_Morphology(index, dictionary_cuts, add_con, bubble_infor_provided, 
                                             systemic_v_type=systemic_v_type, fixed_ec=fixed_ec, half_ecoords=half_ecoords)
        if len(bubbleObj.bubble_clump_ids) > bubbleObj.ClumpNum:  
            BFPr.Cal_Mean_Profile(bubbleObj, mean_line_num=0, mean_line_range=[-30, 30], ExtendRange=0)
            fit_results = BFPr.Fit_Double_Gaussian(bubbleObj, local_fit=True, left_constraint_width=10, right_constraint_width=10)

            # If the fit was successful, proceed with further analysis
        
            symmetry_score_0 = bubbleObj.fit_results['symmetry_score']
            if symmetry_score_0 < SymScore or bubbleObj.fit_results['local_fit'] == False:
                # Reprocess if the symmetry score is below threshold
                dictionary_cuts = defaultdict(list)
                fixed_ec_T = not fixed_ec
                bubbleObj.Get_Bubble_Item_Morphology(index, dictionary_cuts, add_con, bubble_infor_provided, 
                                                     systemic_v_type=systemic_v_type, fixed_ec=fixed_ec_T, half_ecoords=half_ecoords)
                BFPr.Cal_Mean_Profile(bubbleObj, mean_line_num=mean_line_num, mean_line_range=[-30, 30], ExtendRange=0)
                fit_results = BFPr.Fit_Double_Gaussian(bubbleObj, local_fit=True, left_constraint_width=6, right_constraint_width=6)
                
            symmetry_score_1 = bubbleObj.fit_results['symmetry_score']
            if symmetry_score_1 <= symmetry_score_0 or bubbleObj.fit_results['local_fit'] == False:
                # Reprocess if the symmetry score is lower
                dictionary_cuts = defaultdict(list)
                bubbleObj.Get_Bubble_Item_Morphology(index, dictionary_cuts, add_con, bubble_infor_provided, 
                                                     systemic_v_type=systemic_v_type, fixed_ec=fixed_ec, half_ecoords=half_ecoords)
                BFPr.Cal_Mean_Profile(bubbleObj, mean_line_num=mean_line_num, mean_line_range=[-30, 30], ExtendRange=0)
                fit_results = BFPr.Fit_Double_Gaussian(bubbleObj, local_fit=True, left_constraint_width=6, right_constraint_width=6)

            if plot_logic:
                ax0 = BFPl.Plot_Bubble_Gas_Region(bubbleObj, add_con=False, plot_clump=True, text_num=False, text_com=True)
                ax0 = BFPl.Plot_Bubble_Gas_Region(bubbleObj, add_con=True, plot_clump=True, text_num=False, text_com=True)
                # ax0 = BFPl.Plot_Bubble_Item(bubbleObj, figsize=(8, 6), fontsize=12, spacing=12*u.arcmin)
                ax0 = BFPl.Plot_Profile_Fit(bubbleObj, figsize=(6, 4))
                plt.show()

        if len(bubbleObj.bubble_clump_ids) > bubbleObj.ClumpNum and bubbleObj.dictionary_cuts['empty_logic'] == False:
            bubble_params_RFWHM = BFPr.Cal_Bubble_RFWHM(bubbleObj, ref_ellipse='cavity', SymScore=SymScore)
                
            BFS.Get_Bubble_Skeleton_Weighted(bubbleObj, unwrap_radius=unwrap_radius, unwrap_width=unwrap_width, 
                                             angle_offset=angle_offset, unwrap_value='max', trim_logic=True)
            
            add_con = add_cons[1]
            fixed_ec_T = False
            dictionary_cuts = defaultdict(list)
            bubbleObj.Get_Bubble_Item_Morphology(
                                index, dictionary_cuts, add_con, bubble_infor_provided,
                                systemic_v_type=systemic_v_type, fixed_ec=fixed_ec_T, half_ecoords=half_ecoords,
                                skeleton_ellipse_infor=bubbleObj.skeleton_ellipse_infor,
                                skeleton_coords_ellipse=bubbleObj.skeleton_coords_ellipse)
            
            if bubbleObj.dictionary_cuts['empty_logic'] == False:
                BFPr.Cal_Mean_Profile(bubbleObj, mean_line_num=mean_line_num, mean_line_range=[-30, 30], ExtendRange=0)
                fit_results = BFPr.Fit_Double_Gaussian(bubbleObj, local_fit=True, ref_ellipse='skeleton', bounds_per=2)
                bubble_params_RFWHM = BFPr.Cal_Bubble_RFWHM(bubbleObj, ref_ellipse='skeleton', SymScore=SymScore)
                symmetry_score = bubbleObj.fit_results['symmetry_score']
                sym_scores.append(symmetry_score)

                if plot_logic:
                    ax0 = BFPl.Plot_Unwrap_Bubble_Infor(bubbleObj, add_con=add_con, plot_clump=False, plot_ellipse=False, 
                                                plot_contour=True, plot_circles=[False, False], plot_annotate=False, 
                                                plot_skeleton=True, plot_skeleton_ellipse=True, skeleton_types=['scatter', 'line'], 
                                                linewidth=1.5, figsize=(8, 6), fontsize=12)

                    ax0 = BFPl.Plot_Unwrapped_Bub_Data(bubbleObj, fontsize=12, figsize=(15, 3))

                    ax0 = BFPl.Plot_Profile_Fit(bubbleObj, figsize=(6, 4))
                    plt.show()

                add_con = add_cons[2]
                dictionary_cuts = defaultdict(list)
                bubbleObj.Get_Bubble_Item_Morphology(
                                    index, dictionary_cuts, add_con, bubble_infor_provided,
                                    systemic_v_type=systemic_v_type, fixed_ec=fixed_ec_T, half_ecoords=half_ecoords,
                                    skeleton_ellipse_infor=bubbleObj.skeleton_ellipse_infor,
                                    skeleton_coords_ellipse=bubbleObj.skeleton_coords_ellipse)

                if update_contour:
                    BFS.Update_Contour_And_Ellipse_Infor(bubbleObj, add_con)

                BFPV.Cal_Radial_Velocity_Profile_Mean(bubbleObj, pv_type=pv_type, width=pv_width)
                exp_significances.append(bubbleObj.max_Sexp)

                if bubbleObj.max_Sexp >= ExpSign:
                    bubble_high_exp_ids.append(index)

                if plot_logic:
                    line_index = bubbleObj.exp_central_delta_v_arg_max
                    ax0 = BFPl.Plot_Bubble_Item(
                        bubbleObj, line_index=line_index, line_logics=[True,True],tick_logic=False, label_e='Skeleton', 
                        figsize=(8, 6), fontsize=12, spacing=12*u.arcmin)

                    ax0 = BFPl.Plot_Radial_Velocity_Profile_Mean(bubbleObj)
                    plt.show()

                # Store results in a structured table format
                skeleton_ellipse_infor_record.append(bubbleObj.skeleton_ellipse_infor)
                skeleton_coords_ellipse_record.append(bubbleObj.skeleton_coords_ellipse)
                exp_sign_pers.append([bubbleObj.exp_sign_positive_mean_per, bubbleObj.exp_sign_negative_mean_per, 
                                      bubbleObj.exp_sign_diff_mean_per])
                bubble_gas_coms_wcs_2.append(bubbleObj.bubble_gas_com_wcs_2)
                thicknesss.append(np.around(bubbleObj.bubble_params_RFWHM['thickness'], 3))
                thicknesss_error.append(np.around(bubbleObj.bubble_params_RFWHM['thickness'], 3))

                # Translate the skeleton ellipse coordinates to world coordinates (WCS)
                skeleton_ellipse_com_wcs_lb = BFTools.Translate_Coords_LBV([np.r_[bubbleObj.skeleton_ellipse_infor[:2][::-1], 0]], 
                                                                            bubbleObj.clumpsObj.data_wcs, pix2world=True)
                skeleton_ellipse_coms_wcs.append(np.r_[skeleton_ellipse_com_wcs_lb[0][:2], bubbleObj.systemic_v])
                skeleton_ellipse_angle_abs.append(bubbleObj.skeleton_ellipse_infor[2:])

                # Perform expansion analysis and store results
                expansion_analysis_mean = bubbleObj.expansion_analysis_mean
                expansion_analysis_mean_red = bubbleObj.expansion_analysis_mean_red
                expansion_analysis_mean_blue = bubbleObj.expansion_analysis_mean_blue
                exp_left_max = expansion_analysis_mean_red['expansion_left_max']
                exp_right_max = expansion_analysis_mean_red['expansion_right_max']
                exp_max_red = (exp_left_max + exp_right_max) / 2
                exp_left_max = expansion_analysis_mean_blue['expansion_left_max']
                exp_right_max = expansion_analysis_mean_blue['expansion_right_max']
                exp_max_blue = (exp_left_max + exp_right_max) / 2
                exp_max_red_blue = np.around(exp_max_red - exp_max_blue, 3)
                exp_maxs.append(exp_max_red_blue)
                bubble_used_ids.append(index)
            else:
                sym_scores.append([])
                exp_significances.append([])
                skeleton_ellipse_infor_record.append([])
                skeleton_coords_ellipse_record.append([])
                exp_sign_pers.append([[], [], []])
                bubble_gas_coms_wcs_2.append([])
                thicknesss.append([])
                thicknesss_error.append([])
                skeleton_ellipse_coms_wcs.append([])
                skeleton_ellipse_angle_abs.append([])
                exp_maxs.append([])
            
        else:
            # If no gas clumps, record empty results
            sym_scores.append([])
            exp_significances.append([])
            skeleton_ellipse_infor_record.append([])
            skeleton_coords_ellipse_record.append([])
            exp_sign_pers.append([[], [], []])
            bubble_gas_coms_wcs_2.append([])
            thicknesss.append([])
            thicknesss_error.append([])
            skeleton_ellipse_coms_wcs.append([])
            skeleton_ellipse_angle_abs.append([])
            exp_maxs.append([])

    # Update the bubble object with the results in a table format
    if updata_records:
        bubbleObj.sym_scores = sym_scores
        bubbleObj.exp_significances = exp_significances
        bubbleObj.bubble_high_exp_ids = bubble_high_exp_ids
        bubbleObj.skeleton_ellipse_infor_record = skeleton_ellipse_infor_record
        bubbleObj.skeleton_coords_ellipse_record = skeleton_coords_ellipse_record
        bubbleObj.exp_sign_pers = exp_sign_pers
        bubbleObj.bubble_gas_coms_wcs_2 = bubble_gas_coms_wcs_2
        bubbleObj.thicknesss = thicknesss
        bubbleObj.thicknesss_error = thicknesss_error
        bubbleObj.skeleton_ellipse_coms_wcs = skeleton_ellipse_coms_wcs
        bubbleObj.skeleton_ellipse_angle_abs = skeleton_ellipse_angle_abs
        bubbleObj.exp_maxs = exp_maxs
    bubbleObj.bubble_used_ids = bubble_used_ids


def Filter_Bubble(bubbleObj, bubble_valid_ids):
    """
    Filter out valid bubble regions and store their data in filtered arrays.

    Parameters
    ----------
    bubbleObj : BubbleClass
        The bubble object containing bubble data.
    bubble_valid_ids : list
        List of valid bubble indices.

    Returns
    -------
    bubble_weight_data_filtered : ndarray
        Filtered bubble weight data.
    bubble_regions_data_filtered : ndarray
        Filtered bubble regions data.
    """
    # Extract bubble regions and data
    bubble_coms = bubbleObj.bubble_coms
    bubble_weight_data_filtered = bubbleObj.bubble_weight_data.copy()
    bubble_regions_data_filtered = bubbleObj.bubble_regions_data.copy()
    bubble_regions = bubbleObj.bubble_regions

    k = 1
    # Loop over all bubbles
    for index in range(len(bubble_coms)):
        # Only keep bubbles whose index is in bubble_valid_ids
        if index in bubble_valid_ids:
            # Get voxel coordinates (x, y, v) belonging to this bubble region
            bub_coords = bubble_regions[index].coords

            # Restore weight data for valid bubble voxels
            bubble_weight_data_filtered[
                bub_coords[:, 0], bub_coords[:, 1], bub_coords[:, 2]
            ] = bubbleObj.bubble_weight_data[
                bub_coords[:, 0], bub_coords[:, 1], bub_coords[:, 2]
            ]

            # Restore region label data for valid bubble voxels
            bubble_regions_data_filtered[
                bub_coords[:, 0], bub_coords[:, 1], bub_coords[:, 2]
            ] = bubbleObj.bubble_regions_data[
                bub_coords[:, 0], bub_coords[:, 1], bub_coords[:, 2]
            ]  # k (commented label placeholder)

            k += 1

    # Save filtered data back into the bubble object
    bubbleObj.bubble_weight_data_filtered = bubble_weight_data_filtered
    bubbleObj.bubble_regions_data_filtered = bubble_regions_data_filtered

    return bubble_weight_data_filtered, bubble_regions_data_filtered


def Get_Exp_Marker(bubbleObj,bubble_valid_id):
    exp_sign_index = np.argmax(bubbleObj.exp_sign_pers[bubble_valid_id])
    if exp_sign_index == 0:
        marker='v'
        fillstyle='full'
    elif exp_sign_index == 1:
        marker='^'
        fillstyle='full'
    elif exp_sign_index == 2:
        marker='d'
        fillstyle='full'
    else:
        print('Exp Sign:',bubbleObj.exp_sign_pers[bubble_valid_id])
    return exp_sign_index,marker





