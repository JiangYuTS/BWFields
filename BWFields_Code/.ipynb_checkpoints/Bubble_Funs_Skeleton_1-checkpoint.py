import time
import numpy as np
import scipy.ndimage as ndimage
from skimage import measure,morphology
from scipy.spatial.distance import cdist
from collections import deque
import networkx as nx
from collections import defaultdict

from DPConCFil import Filament_Class_Funs_Analysis as FCFA 

from . import Bubble_Funs_Morphology as BFM
from . import Bubble_Funs_Profile as BFPr
from . import Bubble_Funs_Tools as BFTools
    

def Trim_Skeleton_Coords_2D(skeleton_coords_2D, max_3x3=3, SmallSkeleton=3):
    """
    Trim and clean up skeleton coordinates by removing redundancies and loops.
    
    This function removes overlapping or redundant points in the skeleton and
    ensures it forms a clean, non-branching path through the filament.
    
    Parameters:
    -----------
    skeleton_coords_2D : ndarray
        Input skeleton coordinates
    fil_mask : ndarray
        Binary mask 
    max_3x3 : int
        Maximum points in 3x3 neighborhood
    SmallSkeleton : int
        Small skeleton threshold
        
    Returns:
    --------
    coords : ndarray
        Trimmed skeleton coordinates
    small_sc : bool
        Whether skeleton is small
    """
    
    def is_connected(coords):
        """Robust connectivity check"""
        if len(coords) < 2:
            return True
        distances = cdist(coords, coords)
        adj = distances <= np.sqrt(2) + 1e-6
        np.fill_diagonal(adj, False)
        
        visited = set([0])
        queue = deque([0])
        while queue:
            current = queue.popleft()
            for i in np.where(adj[current])[0]:
                if i not in visited:
                    visited.add(i)
                    queue.append(i)
        return len(visited) == len(coords)
    
    def find_main_path_with_priorities(coords):
        """Find main skeleton path and assign priorities"""
        if len(coords) < 3:
            return list(range(len(coords))), np.ones(len(coords))
        
        distances = cdist(coords, coords)
        adj_matrix = distances <= 2 #np.sqrt(2) + 1e-6
        np.fill_diagonal(adj_matrix, False)
        
        # Find endpoints (points with few neighbors)
        neighbor_counts = np.sum(adj_matrix, axis=1)
        endpoints = np.where(neighbor_counts <= 2)[0]
        
        if len(endpoints) < 2:
            # Use furthest apart points as endpoints
            max_dist = 0
            start_idx, end_idx = 0, len(coords)-1
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    if distances[i, j] > max_dist:
                        max_dist = distances[i, j]
                        start_idx, end_idx = i, j
        else:
            start_idx, end_idx = endpoints[0], endpoints[-1]
        
        # Build path from start to end
        path = [start_idx]
        current = start_idx
        visited = {start_idx}
        
        while current != end_idx and len(path) < len(coords):
            neighbors = np.where(adj_matrix[current])[0]
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            if not unvisited_neighbors:
                break
                
            # Choose neighbor closest to end point
            best_neighbor = min(unvisited_neighbors, 
                              key=lambda n: distances[n, end_idx])
            path.append(best_neighbor)
            visited.add(best_neighbor)
            current = best_neighbor
        
        # Assign priority scores (main path gets higher scores)
        priorities = np.zeros(len(coords))
        for i, idx in enumerate(path):
            priorities[idx] = len(path) - i + 10  # Boost main path points
        
        # Give remaining points lower priority based on connectivity
        for i in range(len(coords)):
            if priorities[i] == 0:
                priorities[i] = neighbor_counts[i]
        
        return path, priorities
    
    def get_box_neighbors(center_coord, coords, box_size):
        """Get neighbors within box"""
        half = box_size // 2
        neighbors = []
        for i, coord in enumerate(coords):
            if (abs(coord[0] - center_coord[0]) <= half and 
                abs(coord[1] - center_coord[1]) <= half):
                neighbors.append(i)
        return neighbors
    
    def find_best_bridge_points(current_coords, original_coords):
        """Find optimal bridge points for reconnection"""
        if is_connected(current_coords):
            return []
        
        # Find disconnected components
        distances = cdist(current_coords, current_coords)
        adj = distances <= np.sqrt(2) + 1e-6
        np.fill_diagonal(adj, False)
        
        visited = set()
        components = []
        for i in range(len(current_coords)):
            if i not in visited:
                component = []
                stack = [i]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        neighbors = np.where(adj[current])[0]
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                stack.append(neighbor)
                components.append(component)
        
        if len(components) <= 1:
            return []
        
        # Find missing points that can bridge components
        current_set = set(map(tuple, current_coords))
        original_set = set(map(tuple, original_coords))
        missing_points = [np.array(p) for p in original_set - current_set]
        
        bridge_candidates = []
        for missing_point in missing_points:
            connected_components = set()
            
            # Check which components this point connects
            for comp_idx, component in enumerate(components):
                for point_idx in component:
                    dist = np.linalg.norm(current_coords[point_idx] - missing_point)
                    if dist <= np.sqrt(2) + 1e-6:
                        connected_components.add(comp_idx)
                        break
            
            if len(connected_components) >= 2:
                bridge_candidates.append((len(connected_components), missing_point))
        
        # Sort by number of components connected (prefer points that connect more)
        bridge_candidates.sort(reverse=True, key=lambda x: x[0])
        return [point for _, point in bridge_candidates]
    
    if len(skeleton_coords_2D) <= SmallSkeleton:
        return skeleton_coords_2D, True
    
    original_coords = skeleton_coords_2D.copy()
    coords = skeleton_coords_2D.copy()
    
    # Find main path and priorities
    main_path, priorities = find_main_path_with_priorities(coords)
    
    # Iterative trimming with priority-based removal
    max_iterations = 20
    for iteration in range(max_iterations):
        coords_changed = False
        points_to_remove = set()
        
        # Update priorities after each iteration
        if iteration > 0:
            _, priorities = find_main_path_with_priorities(coords)
        
        # Check density constraints
        for i in range(len(coords)):
            # Check 3x3 neighborhood
            neighbors_3x3 = get_box_neighbors(coords[i], coords, 3)
            if len(neighbors_3x3) > max_3x3:
                excess = len(neighbors_3x3) - max_3x3
                # Sort by priority (remove lowest priority first)
                neighbor_priorities = [(priorities[j] if j < len(priorities) else 0, j) 
                                     for j in neighbors_3x3 if j != i]
                neighbor_priorities.sort()
                
                for _, remove_idx in neighbor_priorities[:excess]:
                    points_to_remove.add(remove_idx)
        
        # Apply removals while checking connectivity
        final_removals = []
        for remove_idx in sorted(points_to_remove, reverse=True):
            temp_coords = np.delete(coords, remove_idx, axis=0)
            if len(temp_coords) > SmallSkeleton and is_connected(temp_coords):
                final_removals.append(remove_idx)
        
        if final_removals:
            # Remove points and update priorities
            for remove_idx in final_removals:
                coords = np.delete(coords, remove_idx, axis=0)
                if remove_idx < len(priorities):
                    priorities = np.delete(priorities, remove_idx)
            coords_changed = True
        
        if not coords_changed:
            break
    
    # Repair connectivity if broken
    max_repair_attempts = len(coords)
    for repair_attempt in range(max_repair_attempts):
        if is_connected(coords):
            break
            
        bridge_points = find_best_bridge_points(coords, original_coords)
        if not bridge_points:
            # print("No bridge points found")
            break
        
        # Add bridge points until connected
        for bridge_point in bridge_points:  # Try up to 5 bridge points
            test_coords = np.vstack([coords, bridge_point.reshape(1, -1)])
            # if is_connected(test_coords):
            coords = test_coords
            break
        else:
            # If no single bridge works, try combinations
            if len(bridge_points) >= 1:
                test_coords = np.vstack([coords] + [bp.reshape(1, -1) for bp in bridge_points[:2]])
                # if is_connected(test_coords):
                coords = test_coords
                break
    
    # Final fallback
    # if not is_connected(coords):
    #     print("Warning: Could not restore full connectivity, using original skeleton")
    #     coords = original_coords

    # Create a graph from the skeleton coordinates
    G_sorted_skeleton, T_sorted_skeleton = FCFA.Graph_Infor(coords)
    
    # Find the longest path through the skeleton
    max_path, max_edges = FCFA.Get_Max_Path_Weight(T_sorted_skeleton)
    
    sorted_skeleton_coords = coords[max_path]
    small_sc = len(sorted_skeleton_coords) <= SmallSkeleton
    return sorted_skeleton_coords, small_sc


def Cal_Angle_Between_Com_And_MaxValue(bubble_item,bubble_com_item,bubble_radius,unwrap_value='max'):
    bubble_item_sum = bubble_item.sum(0)
    circle_points, circumference = BFTools.Generate_Circle_Points(bubble_com_item[2], bubble_com_item[1], bubble_radius)
    if (np.int64(np.around(circle_points))[:,1].max() < bubble_item_sum.shape[1] - 1) and \
            (np.int64(np.around(circle_points))[:,0].max() < bubble_item_sum.shape[0] - 1): 
        circle_values = []
        for circle_point in circle_points:
            neighbor_coords = BFTools.Generate_Neighbor_coords(np.int64(np.around(circle_point)),bubble_item_sum.shape)
            circle_value = bubble_item_sum[neighbor_coords[:,1],neighbor_coords[:,0]].sum()
            circle_values.append(circle_value)
        circle_values = np.array(circle_values)

        # circle_values = bubble_item_sum[neighbor_coords[:,1],neighbor_coords[:,0]]
        
        if unwrap_value == 'min':
            point_a = circle_points[np.argmin(circle_values)]
        elif unwrap_value == 'max':
            point_a = circle_points[np.argmax(circle_values)]
        else:
            print('Choose the type of unwrap_value, min or max')

    else:
        if unwrap_value == 'min':
            point_a = np.where(bubble_item_sum==bubble_item_sum[bubble_item_sum>0].min())
            point_a = np.c_[point_a[0],point_a[1]][0]
        elif unwrap_value == 'max':
            point_a = np.where(bubble_item_sum==bubble_item_sum[bubble_item_sum>0].max())
            point_a = np.c_[point_a[0],point_a[1]][0]
        else:
            print('Choose the type of unwrap_value, min or max')
            
    point_b = bubble_com_item[1:][::-1]
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    rad_angle = np.arctan2(dy, dx)
    delta_angle = np.degrees(rad_angle) % 360
    return circle_points,point_a,delta_angle


def Unwrap_Circle_Data(data, center_x, center_y, radius, width, angle_offset=0):
    """
    Unwrap circular data within specified width range
    
    Parameters:
    - data: 2D array data
    - center_x, center_y: Circle center coordinates  
    - radius: Circle radius (green circle)
    - width: Radial width for unwrapping (red circle range)
    - angle_offset: Starting angle offset in degrees (default: 0)
                   0° = East (right), 90° = North (up), 180° = West (left), 270° = South (down)
    
    Returns:
    - unwrapped: Unwrapped data array [circumference × width_samples]
    - angles: Angle array (degrees)
    - radial_dist: Radial distance array
    """
    
    # Calculate sampling points
    circumference = 2 * np.pi * radius
    n_angles = int(round(circumference))

    n_angles = int(round(circumference))
    n_radial = int(width * 2 + 1) # 2 samples per unit width
    
    # Create sampling arrays with offset
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False) + np.radians(angle_offset)
    # angles = np.r_[angles,angles[0]]
    radial_dist = np.linspace(-width, width, n_radial)
    
    # Initialize output
    unwrapped = np.zeros((n_angles, n_radial))
    
    # Sample data
    for i, angle in enumerate(angles):
        for j, r_offset in enumerate(radial_dist):
            # Calculate sampling coordinates
            r = radius + r_offset
            x = center_x + r * np.cos(angle)
            y = center_y + r * np.sin(angle)
            
            # Bilinear interpolation if within bounds
            if 0 <= x < data.shape[1]-1 and 0 <= y < data.shape[0]-1:
                x0, y0 = int(x), int(y)
                x1, y1 = x0 + 1, y0 + 1
                
                # Interpolation weights
                wx, wy = x - x0, y - y0
                
                # Interpolate
                value = ((1-wx)*(1-wy)*data[y0, x0] + 
                        wx*(1-wy)*data[y0, x1] + 
                        (1-wx)*wy*data[y1, x0] + 
                        wx*wy*data[y1, x1])
                
                unwrapped[i, j] = value
    
    angles_deg = np.degrees(angles - np.radians(angle_offset))  # Convert back to 0-360 range
    unwrapped = unwrapped.T

    if width > radius:
        delta_width = int(round(width - radius)) + 1
        unwrapped[:delta_width] = 0
    return unwrapped, angles_deg, radial_dist
    

def Graph_Infor_Connected_Unwrapped_Bub(unwrapped_bub_data,mask_coords):
    bub_image = unwrapped_bub_data
    bub_image_filtered = ndimage.uniform_filter(unwrapped_bub_data, size=3)
    
    # Calculate distances between neighboring coordinates
    dist_matrix = FCFA.Dists_Array(mask_coords, mask_coords)
    mask_coords_in_dm = np.where(np.logical_and(dist_matrix > 0.5, dist_matrix < 2))
    
    # Create a graph with edges between neighboring pixels
    Graph_find_skeleton = nx.Graph()
    common_mask_coords_id = []
    
    # Add edges with weights inversely proportional to intensity
    for i, j in zip(mask_coords_in_dm[0], mask_coords_in_dm[1]):
        weight_ij = bub_image_filtered[mask_coords[i][0], mask_coords[i][1]] + \
                    bub_image_filtered[mask_coords[j][0], mask_coords[j][1]]
        
        # mask_coord_delta = np.abs(mask_coords[i] - mask_coords[j])
        # if mask_coord_delta[0] < mask_coord_delta[1]:
        #     weight_ij = weight_ij * 0.5
        
        # Set weight to 0 if intensity is 0, otherwise make it inversely proportional
        if weight_ij != 0:
            weight_ij = dist_matrix[i, j] / weight_ij
        
        # Add edge to graph
        Graph_find_skeleton.add_edge(i, j, weight=weight_ij)
        
    # Find minimum spanning tree
    Tree = nx.minimum_spanning_tree(Graph_find_skeleton)
    return Graph_find_skeleton,Tree


def Get_Splice_Nodes_Pair(unwrapped_bub_data,mask_coords,Tree):
    splice_points_start = np.c_[np.arange(mask_coords[:,0].min(),mask_coords[:,0].max()+1),\
                                np.ones(mask_coords[:,0].max()-mask_coords[:,0].min()+1)*(mask_coords[:,1].min())]
    splice_points_end = np.c_[np.arange(mask_coords[:,0].min(),mask_coords[:,0].max()+1),
                              np.ones(mask_coords[:,0].max()-mask_coords[:,0].min()+1)*(mask_coords[:,1].max())]
    splice_points = np.r_[splice_points_start,splice_points_end]
    contour_data = np.zeros_like(unwrapped_bub_data,dtype='int32')
    for splice_point in splice_points:
        contour_data[np.int32(splice_point[0]),np.int32(splice_point[1])] = 1
    
    splice_nodes = [node for node in Tree.nodes if Tree.degree(node) > 0 and \
                         contour_data[mask_coords[node][0], mask_coords[node][1]]]
    
    splice_nodes_pair = {}
    for node_1 in splice_nodes:
        for node_2 in splice_nodes:
            if mask_coords[node_1][0] == mask_coords[node_2][0] and mask_coords[node_1][1] != mask_coords[node_2][1]:
                splice_nodes_pair[node_1] = node_2
    splice_nodes_pair = {k: v for k, v in splice_nodes_pair.items() if k < v}
    
    if len(splice_nodes_pair) == 0:
        splice_nodes_pair = {}
        for node_1 in splice_nodes:
            for node_2 in splice_nodes:
                splice_nodes_pair[node_1] = node_2
        splice_nodes_pair = {k: v for k, v in splice_nodes_pair.items() if k < v}
            
    return splice_nodes_pair


def Get_Bubble_Skeleton_Weighted_Unwrapped(unwrapped_bub_data,Tree,splice_nodes_pair,mask_coords,trim_logic=True):
    path_weights = []
    touch_edge_splice_nodes_nums = []
    skeleton_coords_2D_record = []
    unwrapped_data_shape = unwrapped_bub_data.shape
    bub_mask = unwrapped_bub_data>0
    
    paths_and_weights = []
    edge_len = 1
    while len(paths_and_weights) == 0:
        for key in splice_nodes_pair:
            if nx.has_path(Tree, key, splice_nodes_pair[key]):
                path = nx.shortest_path(Tree, key, splice_nodes_pair[key])
                path_weight = 0
                for k in range(len(path) - 1):
                    if Tree[path[k]][path[k + 1]]['weight'] != 0:
                        path_weight += 1 / Tree[path[k]][path[k + 1]]['weight']
                        
                skeleton_coords_2D = mask_coords[path]
                if skeleton_coords_2D[0][0] >= int(unwrapped_data_shape[0]/3 ) and \
                   skeleton_coords_2D[0][0] <= int(unwrapped_data_shape[0] - unwrapped_data_shape[0]/3) and \
                   0 not in skeleton_coords_2D[edge_len:][:,1] and \
                   skeleton_coords_2D[:,1].max() not in skeleton_coords_2D[:-edge_len-1][:,1]:
                    paths_and_weights.append((path, path_weight))
                    
        edge_len += 1
        if edge_len > unwrapped_data_shape[1]/2:
            for key in splice_nodes_pair:
                if nx.has_path(Tree, key, splice_nodes_pair[key]):
                    path = nx.shortest_path(Tree, key, splice_nodes_pair[key])
                    path_weight = 0
                    for k in range(len(path) - 1):
                        if Tree[path[k]][path[k + 1]]['weight'] != 0:
                            path_weight += 1 / Tree[path[k]][path[k + 1]]['weight']
                        paths_and_weights.append((path, path_weight))
        
    max_weight, max_path, max_edges = FCFA.Search_Max_Path_And_Edges(paths_and_weights)
    skeleton_coords_2D = mask_coords[max_path]
    
    if trim_logic:
        skeleton_coords_2D, small_sc = Trim_Skeleton_Coords_2D(skeleton_coords_2D)
    return skeleton_coords_2D


def Get_Mapping_Skeleton_Coords(skeleton_coords_2D,unwrap_width):
    skeleton_radial = skeleton_coords_2D[:,0]
    skeleton_angles = skeleton_coords_2D[:,1]
    skeleton_radial = skeleton_radial - unwrap_width 
    skeleton_angles = skeleton_angles/skeleton_angles.max()*360
    
    # skeleton_angles_mask = (skeleton_angles != 0) & (skeleton_angles != 360)
    # skeleton_angles_label = measure.label(skeleton_angles_mask,connectivity=1)
    # used_angel_coords = np.where(skeleton_angles_label == skeleton_angles_label[np.int32(len(skeleton_angles_label)/2)])[0]

    # skeleton_radial = skeleton_radial[np.max([0,used_angel_coords[0]-1]):used_angel_coords[-1]+2]
    # skeleton_angles = skeleton_angles[np.max([0,used_angel_coords[0]-1]):used_angel_coords[-1]+2]
    # skeleton_coords_2D = skeleton_coords_2D[np.max([0,used_angel_coords[0]-1]):used_angel_coords[-1]+2]
    
    skeleton_radial = np.r_[skeleton_radial,skeleton_radial[0]]
    skeleton_angles = np.r_[skeleton_angles,skeleton_angles[0]]
    
    return skeleton_coords_2D,skeleton_radial,skeleton_angles
    

def Map_Curve_To_Original(curve_radial, curve_angles, center_x, center_y, radius, angle_offset=0):
    """
    Map a curve from unwrapped coordinates back to original image coordinates
    
    Parameters:
    - curve_radial: Array of radial distances (from unwrapped image x-axis)
    - curve_angles: Array of angles in degrees (from unwrapped image y-axis)
    - center_x, center_y: Circle center in original image
    - radius: Base circle radius
    - angle_offset: Angle offset used in unwrapping (degrees)
    
    Returns:
    - curve_x, curve_y: Coordinates of curve in original image
    """
    
    # Convert angles to radians and add offset
    angles_rad = np.radians(curve_angles) + np.radians(angle_offset)
    
    # Calculate actual radius for each point
    actual_radius = radius + curve_radial
    
    # Convert to Cartesian coordinates
    curve_x = center_x + actual_radius * np.cos(angles_rad)
    curve_y = center_y + actual_radius * np.sin(angles_rad)
    
    return curve_x, curve_y


def Get_Bubble_Skeleton_Weighted(bubbleObj,unwrap_radius=None,unwrap_width=None,angle_offset=None,unwrapped_zero_col=6,\
                                 unwrap_value='max',trim_logic=True):
    bubble_item = bubbleObj.bubble_item 
    bubble_com_item = bubbleObj.bubble_com_item 
    bubble_radius = bubbleObj.bubble_params_RFWHM['radius']
    bubble_thickness = bubbleObj.bubble_params_RFWHM['thickness']
    skeleton_coords_unwrapped = np.array([[0,0]])
    
    circle_points,point_mvalue,delta_angle = Cal_Angle_Between_Com_And_MaxValue(bubble_item,bubble_com_item,bubble_radius,unwrap_value)
    if unwrap_radius == None:
        unwrap_radius = bubble_radius 
    if unwrap_width == None:
        unwrap_width = int(round(bubble_thickness/2))
    if angle_offset == None:
        angle_offset = delta_angle
    
    unwrapped_bub_data,angles,radial_distances = Unwrap_Circle_Data(
                bubble_item.sum(0),bubble_com_item[2],bubble_com_item[1],unwrap_radius,unwrap_width,angle_offset)

    unwrapped_bub_data_shape = unwrapped_bub_data.shape
    zero_width = np.int16(np.around(unwrapped_bub_data_shape[0]/unwrapped_zero_col))
    unwrapped_bub_data_T = unwrapped_bub_data.copy()
    unwrapped_bub_data_T[:zero_width,:] = 0
    unwrapped_bub_data_T[unwrapped_bub_data_shape[0]-1-zero_width,:] = 0
    bub_mask = unwrapped_bub_data_T>0
    
    bub_mask_label = measure.label(bub_mask,connectivity=2)
    regions_list = measure.regionprops(bub_mask_label)
    
    for i in range(len(regions_list)):
        mask_coords = regions_list[i].coords
        if unwrap_width in mask_coords[:,0]:
            Graph_find_skeleton,Tree = Graph_Infor_Connected_Unwrapped_Bub(unwrapped_bub_data,mask_coords)
            splice_nodes_pair = Get_Splice_Nodes_Pair(unwrapped_bub_data,mask_coords,Tree)
            skeleton_coords_unwrapped_i = Get_Bubble_Skeleton_Weighted_Unwrapped(unwrapped_bub_data,Tree,splice_nodes_pair,mask_coords,trim_logic)
    
            skeleton_coords_unwrapped_i,skeleton_radial,skeleton_angles = Get_Mapping_Skeleton_Coords(skeleton_coords_unwrapped_i,unwrap_width)
            skeleton_coords_unwrapped = np.r_[skeleton_coords_unwrapped,skeleton_coords_unwrapped_i]

    skeleton_coords_unwrapped = skeleton_coords_unwrapped[1:]
    skeleton_coords_unwrapped,skeleton_radial,skeleton_angles = Get_Mapping_Skeleton_Coords(skeleton_coords_unwrapped,unwrap_width)
    skeleton_x, skeleton_y = Map_Curve_To_Original(
                skeleton_radial,skeleton_angles,bubble_com_item[2],bubble_com_item[1],unwrap_radius,angle_offset)
    skeleton_coords_original = np.c_[skeleton_y,skeleton_x]

    paras = BFM.Solve_Ellipse(skeleton_coords_original[:,0],skeleton_coords_original[:,1])
    skeleton_ellipse_infor,coords_fit,ellipse_perimeter = BFM.Solve_Ellipse_Infor(paras)
    skeleton_coords_ellipse = np.c_[coords_fit[0],coords_fit[1]]

    start_coords_item = bubbleObj.start_coords
    skeleton_ellipse_infor[0] = skeleton_ellipse_infor[0]+start_coords_item[1]
    skeleton_ellipse_infor[1] = skeleton_ellipse_infor[1]+start_coords_item[2]
    skeleton_coords_original = np.c_[skeleton_y,skeleton_x] + start_coords_item[1:]
    skeleton_coords_ellipse = skeleton_coords_ellipse + start_coords_item[1:]
        
    bubbleObj.circle_points = circle_points
    bubbleObj.point_mvalue = point_mvalue
    bubbleObj.unwrap_radius = unwrap_radius
    bubbleObj.unwrap_width = unwrap_width
    bubbleObj.angle_offset = angle_offset
    bubbleObj.unwrapped_bub_data = unwrapped_bub_data
    bubbleObj.skeleton_coords_unwrapped = skeleton_coords_unwrapped
    bubbleObj.skeleton_coords_original = skeleton_coords_original
    bubbleObj.skeleton_ellipse_infor = skeleton_ellipse_infor
    bubbleObj.skeleton_coords_ellipse = skeleton_coords_ellipse


def Update_Contour_And_Ellipse_Infor(bubbleObj,add_con):
    origin_data = bubbleObj.clumpsObj.origin_data
    regions_data = bubbleObj.clumpsObj.regions_data
    data_wcs = bubbleObj.clumpsObj.data_wcs
    connected_ids_dict = bubbleObj.clumpsObj.connected_ids_dict
    clump_coords_dict = bubbleObj.clumpsObj.clump_coords_dict
    bubbleObj.ellipse_infor = bubbleObj.skeleton_ellipse_infor
    bubbleObj.ellipse_coords = bubbleObj.skeleton_coords_ellipse
    bubbleObj.bubble_contour = bubbleObj.skeleton_coords_original - bubbleObj.start_coords[1:]
    bubble_clump_ids = bubbleObj.bubble_clump_ids
    bubble_clump_ids_con = bubbleObj.bubble_clump_ids_con
    dictionary_cuts = defaultdict(list)

    bubble_com_item = bubbleObj.bubble_com_item
    ellipse_com = np.mean(bubbleObj.ellipse_coords - bubbleObj.start_coords[1:],0)
    bubbleObj.bubble_com_item = np.array([bubble_com_item[0],ellipse_com[0],ellipse_com[1]])
    
    clump_ids = list(bubble_clump_ids) 
    if add_con:
        clump_ids = list(bubble_clump_ids) + list(bubble_clump_ids_con)
        
    bubble_coords,bubble_item,data_wcs_item,regions_data_i,start_coords,bubble_item_mask_2D,lb_area = \
                                    FCFA.Filament_Coords(origin_data,regions_data,data_wcs,clump_coords_dict,clump_ids)
    
    ellipse_x0,ellipse_y0,ellipse_angle,a_res,b_res = bubbleObj.skeleton_ellipse_infor
    points,fprime,points_b = BFPr.Cal_Derivative(ellipse_x0,ellipse_y0,bubbleObj.skeleton_coords_ellipse,start_coords)
    dictionary_cuts = BFPr.Cal_Dictionary_Cuts(regions_data,clump_ids,connected_ids_dict,clump_coords_dict, \
                            points,points_b,fprime,bubble_item.sum(0),bubble_item_mask_2D,dictionary_cuts,start_coords)
    
    bubbleObj.dictionary_cuts = dictionary_cuts









# import numpy as np
# from scipy.special import ellipe

# def Unwrap_Ellipse_Data(data, center_x, center_y, semi_major, semi_minor, theta, width, angle_offset=0):
#     """
#     Unwrap elliptical data within specified width range
    
#     Parameters:
#     - data: 2D array data
#     - center_x, center_y: Ellipse center coordinates
#     - semi_major: Semi-major axis (a)
#     - semi_minor: Semi-minor axis (b)
#     - theta: Rotation angle of ellipse in radians
#     - width: Normal width for unwrapping (perpendicular to ellipse)
#     - angle_offset: Starting angle offset in degrees (default: 0)
    
#     Returns:
#     - unwrapped: Unwrapped data array [perimeter_samples × width_samples]
#     - arc_lengths: Arc length array (normalized 0-1)
#     - normal_dist: Normal distance array
#     """
    
#     # Calculate ellipse perimeter using elliptic integral
#     e_squared = 1 - (semi_minor**2 / semi_major**2)
#     E = ellipe(e_squared)
#     perimeter = 4 * semi_major * E
    
#     # Number of sampling points along ellipse
#     n_angles = int(round(perimeter))
#     n_normal = int(width * 2 + 1)  # 2 samples per unit width
    
#     # Parametric angles with offset
#     t_angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False) + np.radians(angle_offset)
#     normal_dist = np.linspace(-width, width, n_normal)
    
#     # Rotation matrix
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)
    
#     # Initialize output
#     unwrapped = np.zeros((n_angles, n_normal))
#     arc_lengths = np.zeros(n_angles)
    
#     # Calculate ellipse points and normals
#     for i, t in enumerate(t_angles):
#         # Ellipse point in local coordinates
#         x_local = semi_major * np.cos(t)
#         y_local = semi_minor * np.sin(t)
        
#         # Tangent vector in local coordinates
#         dx_local = -semi_major * np.sin(t)
#         dy_local = semi_minor * np.cos(t)
        
#         # Rotate tangent to global coordinates
#         dx_global = cos_theta * dx_local - sin_theta * dy_local
#         dy_global = sin_theta * dx_local + cos_theta * dy_local
        
#         # Normal vector (perpendicular to tangent, pointing outward)
#         tangent_length = np.sqrt(dx_global**2 + dy_global**2)
#         nx = -dy_global / tangent_length  # Perpendicular to tangent
#         ny = dx_global / tangent_length
        
#         # Ellipse point in global coordinates
#         x_ellipse = center_x + cos_theta * x_local - sin_theta * y_local
#         y_ellipse = center_y + sin_theta * x_local + cos_theta * y_local
        
#         # Calculate arc length (approximate)
#         if i > 0:
#             arc_lengths[i] = arc_lengths[i-1] + tangent_length * (2*np.pi / n_angles)
        
#         # Sample along normal direction
#         for j, n_offset in enumerate(normal_dist):
#             # Sampling coordinates along normal
#             x = x_ellipse + n_offset * nx
#             y = y_ellipse + n_offset * ny
            
#             # Bilinear interpolation if within bounds
#             if 0 <= x < data.shape[1]-1 and 0 <= y < data.shape[0]-1:
#                 x0, y0 = int(x), int(y)
#                 x1, y1 = x0 + 1, y0 + 1
                
#                 # Interpolation weights
#                 wx, wy = x - x0, y - y0
                
#                 # Interpolate
#                 value = ((1-wx)*(1-wy)*data[y0, x0] + 
#                         wx*(1-wy)*data[y0, x1] + 
#                         (1-wx)*wy*data[y1, x0] + 
#                         wx*wy*data[y1, x1])
                
#                 unwrapped[i, j] = value
    
#     # Normalize arc lengths to 0-1
#     if arc_lengths[-1] > 0:
#         arc_lengths = arc_lengths / arc_lengths[-1]
    
#     unwrapped = unwrapped.T
    
#     return unwrapped, arc_lengths, normal_dist


# def Map_Curve_From_Unwrapped_Ellipse(curve_normal, curve_arc, center_x, center_y, 
#                                       semi_major, semi_minor, theta, angle_offset=0):
#     """
#     Map a curve from unwrapped ellipse coordinates back to original image coordinates
    
#     Parameters:
#     - curve_normal: Array of normal distances (from unwrapped image x-axis)
#     - curve_arc: Array of arc length positions 0-1 (from unwrapped image y-axis)
#     - center_x, center_y: Ellipse center in original image
#     - semi_major: Semi-major axis (a)
#     - semi_minor: Semi-minor axis (b)
#     - theta: Rotation angle of ellipse in radians
#     - angle_offset: Angle offset used in unwrapping (degrees)
    
#     Returns:
#     - curve_x, curve_y: Coordinates of curve in original image
#     """
    
#     # Convert arc lengths to parametric angles
#     t_angles = curve_arc * 2 * np.pi + np.radians(angle_offset)
    
#     # Rotation matrix
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)
    
#     curve_x = np.zeros_like(curve_arc)
#     curve_y = np.zeros_like(curve_arc)
    
#     for i, (t, n_offset) in enumerate(zip(t_angles, curve_normal)):
#         # Ellipse point in local coordinates
#         x_local = semi_major * np.cos(t)
#         y_local = semi_minor * np.sin(t)
        
#         # Tangent vector in local coordinates
#         dx_local = -semi_major * np.sin(t)
#         dy_local = semi_minor * np.cos(t)
        
#         # Rotate tangent to global coordinates
#         dx_global = cos_theta * dx_local - sin_theta * dy_local
#         dy_global = sin_theta * dx_local + cos_theta * dy_local
        
#         # Normal vector
#         tangent_length = np.sqrt(dx_global**2 + dy_global**2)
#         nx = -dy_global / tangent_length
#         ny = dx_global / tangent_length
        
#         # Ellipse point in global coordinates
#         x_ellipse = center_x + cos_theta * x_local - sin_theta * y_local
#         y_ellipse = center_y + sin_theta * x_local + cos_theta * y_local
        
#         # Point along normal
#         curve_x[i] = x_ellipse + n_offset * nx
#         curve_y[i] = y_ellipse + n_offset * ny
    
#     return curve_x, curve_y


# # Modified version for your bubble analysis code
# def Get_Bubble_Skeleton_Weighted_Ellipse(bubbleObj, unwrap_width=None, angle_offset=None, 
#                                          unwrap_value='min', trim_logic=True):
#     """
#     Get bubble skeleton using ellipse unwrapping instead of circle
    
#     Parameters:
#     - bubbleObj: Bubble object containing bubble information
#     - unwrap_width: Width for unwrapping (default: bubble thickness)
#     - angle_offset: Starting angle offset (default: auto-calculated)
#     - unwrap_value: 'min' or 'max' for determining unwrap start point
#     - trim_logic: Whether to trim skeleton
#     """
#     bubble_item = bubbleObj.bubble_item 
#     bubble_com_item = bubbleObj.bubble_com_item
    
#     # Get ellipse parameters from skeleton_ellipse_infor
#     # [x0, y0, angle, a, b]
#     if hasattr(bubbleObj, 'skeleton_ellipse_infor'):
#         ellipse_x0, ellipse_y0, ellipse_angle, semi_major, semi_minor = bubbleObj.skeleton_ellipse_infor
#         theta = -np.radians(ellipse_angle)  # Convert to radians
#     else:
#         # Fallback to circle parameters
#         bubble_radius = bubbleObj.bubble_params_RFWHM['radius']
#         semi_major = bubble_radius
#         semi_minor = bubble_radius
#         theta = 0
#         ellipse_x0, ellipse_y0 = bubble_com_item[2], bubble_com_item[1]
    
#     # Set unwrap width
#     if unwrap_width is None:
#         bubble_thickness = bubbleObj.bubble_params_RFWHM['thickness']
#         unwrap_width = int(round(bubble_thickness))
    
#     # Calculate angle offset if not provided
#     if angle_offset is None:
#         bubble_item_sum = bubble_item.sum(0)
#         if unwrap_value == 'min':
#             point_a = np.where(bubble_item_sum == bubble_item_sum[bubble_item_sum > 0].min())
#         elif unwrap_value == 'max':
#             point_a = np.where(bubble_item_sum == bubble_item_sum[bubble_item_sum > 0].max())
#         point_a = np.c_[point_a[0], point_a[1]][0]
#         point_b = bubble_com_item[1:][::-1]
#         dx = point_a[0] - point_b[0]
#         dy = point_a[1] - point_b[1]
#         rad_angle = np.arctan2(dy, dx)
#         angle_offset = np.degrees(rad_angle) % 360
    
#     # Unwrap using ellipse
#     unwrapped_bub_data, arc_lengths, normal_distances = Unwrap_Ellipse_Data(
#         bubble_item.sum(0), ellipse_x0, ellipse_y0, semi_major, semi_minor, 
#         theta, unwrap_width, angle_offset
#     )
    
#     # Rest of the skeleton finding logic remains similar
#     bub_mask = unwrapped_bub_data > 0
#     bub_mask_label = measure.label(bub_mask, connectivity=2)
#     regions_list = measure.regionprops(bub_mask_label)
    
#     skeleton_coords_unwrapped = np.array([[0, 0]])
    
#     for i in range(len(regions_list)):
#         mask_coords = regions_list[i].coords
#         if unwrap_width in mask_coords[:, 0]:
#             Graph_find_skeleton, Tree = Graph_Infor_Connected_Unwrapped_Bub(
#                 unwrapped_bub_data, mask_coords
#             )
#             splice_nodes_pair = Get_Splice_Nodes_Pair(
#                 unwrapped_bub_data, mask_coords, Tree
#             )
#             skeleton_coords_unwrapped_i = Get_Bubble_Skeleton_Weighted_Unwrapped(
#                 unwrapped_bub_data, Tree, splice_nodes_pair, mask_coords, trim_logic
#             )
#             skeleton_coords_unwrapped = np.r_[skeleton_coords_unwrapped, skeleton_coords_unwrapped_i]
    
#     skeleton_coords_unwrapped = skeleton_coords_unwrapped[1:]
    
#     # Convert unwrapped coordinates to arc length positions
#     skeleton_normal = skeleton_coords_unwrapped[:, 0] - unwrap_width
#     skeleton_arc = skeleton_coords_unwrapped[:, 1] / skeleton_coords_unwrapped[:, 1].max()
    
#     # Map back to original coordinates using ellipse
#     skeleton_x, skeleton_y = Map_Curve_From_Unwrapped_Ellipse(
#         skeleton_normal, skeleton_arc, ellipse_x0, ellipse_y0,
#         semi_major, semi_minor, theta, angle_offset
#     )
    
#     skeleton_coords_original = np.c_[skeleton_y, skeleton_x]
    
#     # Store results
#     bubbleObj.unwrap_width = unwrap_width
#     bubbleObj.angle_offset = angle_offset
#     bubbleObj.unwrapped_bub_data = unwrapped_bub_data
#     bubbleObj.skeleton_coords_unwrapped = skeleton_coords_unwrapped
#     bubbleObj.skeleton_coords_original = skeleton_coords_original
    
#     return skeleton_coords_original


# # Example usage
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
    
#     # Create test ellipse data
#     y, x = np.ogrid[0:200, 0:200]
#     center_x, center_y = 100, 100
#     semi_major, semi_minor = 50, 30
#     theta = np.pi / 6  # 30 degrees rotation
    
#     # Create rotated ellipse mask
#     cos_t, sin_t = np.cos(theta), np.sin(theta)
#     x_rot = cos_t * (x - center_x) + sin_t * (y - center_y)
#     y_rot = -sin_t * (x - center_x) + cos_t * (y - center_y)
    
#     ellipse_mask = ((x_rot / semi_major)**2 + (y_rot / semi_minor)**2) < 1
#     ellipse_data = ellipse_mask.astype(float) * 100
    
#     # Add some pattern
#     ellipse_data += 50 * np.sin(x / 10) * ellipse_mask
    
#     # Unwrap the ellipse
#     unwrapped, arc_lengths, normal_dist = Unwrap_Ellipse_Data(
#         ellipse_data, center_x, center_y, semi_major, semi_minor, theta, width=20
#     )
    
#     # Visualize
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
#     # Original data
#     axes[0].imshow(ellipse_data, cmap='gray', origin='lower')
#     axes[0].plot(center_x, center_y, 'r*', markersize=15, label='Center')
#     axes[0].set_title('Original Ellipse Data')
#     axes[0].legend()
#     axes[0].axis('equal')
    
#     # Unwrapped data
#     im = axes[1].imshow(unwrapped, aspect='auto', cmap='gray', origin='lower')
#     axes[1].set_title('Unwrapped Ellipse Data')
#     axes[1].set_xlabel('Arc Length Position')
#     axes[1].set_ylabel('Normal Distance')
#     plt.colorbar(im, ax=axes[1])
    
#     plt.tight_layout()
#     plt.show()

