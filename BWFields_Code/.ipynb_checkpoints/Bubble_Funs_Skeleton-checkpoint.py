import time
import numpy as np
import scipy.ndimage as ndimage
from skimage import measure, morphology
from scipy.spatial.distance import cdist
from scipy.special import ellipe
from scipy.optimize import fsolve
from collections import deque
import networkx as nx
from collections import defaultdict

from DPConCFil import Filament_Class_Funs_Analysis as FCFA

from . import Bubble_Funs_Morphology as BFM
from . import Bubble_Funs_Profile as BFPr
from . import Bubble_Funs_Tools as BFTools


def Trim_Skeleton_Coords_2D(skeleton_coords_2D, max_3x3=3, SmallSkeleton=3):
    """
    Trim and clean a 2D skeleton (polyline-like) represented by pixel coordinates.

    Goals:
      - Remove overly dense local clusters (too many points in a small neighborhood),
      - Preserve connectivity of the skeleton,
      - Preferentially keep the main/longest path rather than side loops/branches,
      - Optionally repair connectivity by re-adding missing points from the original set,
      - Finally re-order points along the longest path.

    Parameters
    ----------
    skeleton_coords_2D : (N, 2) ndarray
        Skeleton pixel coordinates in 2D (row/col or y/x depending on upstream conventions).
    max_3x3 : int
        Maximum allowed points in a 3x3 neighborhood (controls local density).
    SmallSkeleton : int
        If skeleton length <= this threshold, treat as small and skip heavy trimming.

    Returns
    -------
    sorted_skeleton_coords : (M, 2) ndarray
        Trimmed and ordered skeleton coordinates along the main path.
    small_sc : bool
        True if the final skeleton is considered small (<= SmallSkeleton).
    """

    def is_connected(coords):
        """
        Check whether all points are mutually connected under 8-neighborhood adjacency.

        Implementation:
          - Build adjacency by pairwise distances <= sqrt(2),
          - BFS from the first node to see if all nodes are reachable.
        """
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
        """
        Heuristically determine a main path through the skeleton and assign per-point priorities.

        Strategy:
          - Build a looser adjacency (distance <= 2) to allow slightly gapped points,
          - Pick endpoints as nodes with low degree (<=2); if not found, use farthest pair,
          - Greedily grow a path from start to end, choosing neighbors closest to the end,
          - Assign higher priority to points on the main path (so trimming removes side points first).
        """
        if len(coords) < 3:
            return list(range(len(coords))), np.ones(len(coords))

        distances = cdist(coords, coords)
        adj_matrix = distances <= 2
        np.fill_diagonal(adj_matrix, False)

        neighbor_counts = np.sum(adj_matrix, axis=1)
        endpoints = np.where(neighbor_counts <= 2)[0]

        if len(endpoints) < 2:
            # If no clear endpoints (loop-like), pick farthest pair as endpoints
            max_dist = 0
            start_idx, end_idx = 0, len(coords) - 1
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    if distances[i, j] > max_dist:
                        max_dist = distances[i, j]
                        start_idx, end_idx = i, j
        else:
            start_idx, end_idx = endpoints[0], endpoints[-1]

        # Build a path from start to end
        path = [start_idx]
        current = start_idx
        visited = {start_idx}

        while current != end_idx and len(path) < len(coords):
            neighbors = np.where(adj_matrix[current])[0]
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            if not unvisited_neighbors:
                break

            # Greedy choice: next neighbor that is closest to end_idx
            best_neighbor = min(unvisited_neighbors, key=lambda n: distances[n, end_idx])
            path.append(best_neighbor)
            visited.add(best_neighbor)
            current = best_neighbor

        # Priority: main path points get a high score; earlier points even higher
        priorities = np.zeros(len(coords))
        for i, idx in enumerate(path):
            priorities[idx] = len(path) - i + 10

        # Remaining points get a low priority based on local connectivity (degree)
        for i in range(len(coords)):
            if priorities[i] == 0:
                priorities[i] = neighbor_counts[i]

        return path, priorities

    def get_box_neighbors(center_coord, coords, box_size):
        """
        Collect indices of points that fall inside a square neighborhood around center_coord.

        box_size=3 corresponds to a 3x3 window (Chebyshev distance <=1).
        """
        half = box_size // 2
        neighbors = []
        for i, coord in enumerate(coords):
            if (abs(coord[0] - center_coord[0]) <= half and
                abs(coord[1] - center_coord[1]) <= half):
                neighbors.append(i)
        return neighbors

    def find_best_bridge_points(current_coords, original_coords):
        """
        If current_coords become disconnected after trimming, try to recover connectivity by
        re-adding points that were removed (from original_coords).

        Strategy:
          - Find connected components in current_coords using 8-neighborhood,
          - Search for 'missing' points that connect >= 2 components (as bridge candidates),
          - Return candidates sorted by how many components they can connect.
        """
        if is_connected(current_coords):
            return []

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

        current_set = set(map(tuple, current_coords))
        original_set = set(map(tuple, original_coords))
        missing_points = [np.array(p) for p in original_set - current_set]

        bridge_candidates = []
        for missing_point in missing_points:
            connected_components = set()

            # Check which components this missing_point touches
            for comp_idx, component in enumerate(components):
                for point_idx in component:
                    dist = np.linalg.norm(current_coords[point_idx] - missing_point)
                    if dist <= np.sqrt(2) + 1e-6:
                        connected_components.add(comp_idx)
                        break

            if len(connected_components) >= 2:
                bridge_candidates.append((len(connected_components), missing_point))

        bridge_candidates.sort(reverse=True, key=lambda x: x[0])
        return [point for _, point in bridge_candidates]

    # If skeleton is already very small, skip trimming to avoid destroying it
    if len(skeleton_coords_2D) <= SmallSkeleton:
        return skeleton_coords_2D, True

    original_coords = skeleton_coords_2D.copy()
    coords = skeleton_coords_2D.copy()

    # Establish an initial "main path" and per-point priorities
    main_path, priorities = find_main_path_with_priorities(coords)

    # Iteratively remove low-priority points in locally dense neighborhoods
    max_iterations = 20
    for iteration in range(max_iterations):
        coords_changed = False
        points_to_remove = set()

        # Recompute priorities after each removal cycle
        if iteration > 0:
            _, priorities = find_main_path_with_priorities(coords)

        # Identify points violating density constraints in a 3x3 neighborhood
        for i in range(len(coords)):
            neighbors_3x3 = get_box_neighbors(coords[i], coords, 3)
            if len(neighbors_3x3) > max_3x3:
                excess = len(neighbors_3x3) - max_3x3

                # Remove the lowest-priority neighbors first (excluding itself)
                neighbor_priorities = [
                    (priorities[j] if j < len(priorities) else 0, j)
                    for j in neighbors_3x3 if j != i
                ]
                neighbor_priorities.sort()

                for _, remove_idx in neighbor_priorities[:excess]:
                    points_to_remove.add(remove_idx)

        # Apply removals conservatively: only remove if connectivity is preserved
        final_removals = []
        for remove_idx in sorted(points_to_remove, reverse=True):
            temp_coords = np.delete(coords, remove_idx, axis=0)
            if len(temp_coords) > SmallSkeleton and is_connected(temp_coords):
                final_removals.append(remove_idx)

        if final_removals:
            for remove_idx in final_removals:
                coords = np.delete(coords, remove_idx, axis=0)
                if remove_idx < len(priorities):
                    priorities = np.delete(priorities, remove_idx)
            coords_changed = True

        if not coords_changed:
            break

    # If trimming broke connectivity, try to repair by re-adding bridge points
    max_repair_attempts = len(coords)
    for repair_attempt in range(max_repair_attempts):
        if is_connected(coords):
            break

        bridge_points = find_best_bridge_points(coords, original_coords)
        if not bridge_points:
            break

        # Add a bridge point (or a small combination) to restore connectivity
        for bridge_point in bridge_points:
            coords = np.vstack([coords, bridge_point.reshape(1, -1)])
            break
        else:
            if len(bridge_points) >= 1:
                coords = np.vstack([coords] + [bp.reshape(1, -1) for bp in bridge_points[:2]])
                break

    # Convert skeleton coords to a graph and extract the longest path ordering
    G_sorted_skeleton, T_sorted_skeleton = FCFA.Graph_Infor(coords)
    max_path, max_edges = FCFA.Get_Max_Path_Weight(T_sorted_skeleton)

    sorted_skeleton_coords = coords[max_path]
    small_sc = len(sorted_skeleton_coords) <= SmallSkeleton
    return sorted_skeleton_coords, small_sc


def Graph_Infor_Connected_Unwrapped_Bub(unwrapped_bub_data, mask_coords):
    """
    Build a connectivity graph over pixels in an unwrapped bubble mask, and compute an MST.

    Nodes:
      - Each pixel in mask_coords is a node (indexed by its row in mask_coords).

    Edges:
      - Connect nodes whose pixel distance is within a local neighborhood (0.5 < d < 2).
      - Edge weight is designed to prefer high-intensity paths:
            weight = dist / (I_i + I_j)
        so minimum spanning tree tends to go through bright pixels.

    Parameters
    ----------
    unwrapped_bub_data : 2D ndarray
        Unwrapped bubble intensity image (e.g., ellipse unwrapping result).
    mask_coords : (N, 2) ndarray
        Coordinates of candidate pixels (typically from a connected component mask).

    Returns
    -------
    Graph_find_skeleton : networkx.Graph
        Full graph with weighted edges.
    Tree : networkx.Graph
        Minimum spanning tree of Graph_find_skeleton.
    """
    bub_image = unwrapped_bub_data

    # Slight smoothing to stabilize edge weights (avoid pixel noise dominating)
    bub_image_filtered = ndimage.uniform_filter(unwrapped_bub_data, size=3)

    # Compute pairwise distances between all mask pixels (via helper)
    dist_matrix = FCFA.Dists_Array(mask_coords, mask_coords)

    # Candidate neighbor pairs (exclude self and far pairs)
    mask_coords_in_dm = np.where(np.logical_and(dist_matrix > 0.5, dist_matrix < 2))

    Graph_find_skeleton = nx.Graph()

    # Build weighted edges
    for i, j in zip(mask_coords_in_dm[0], mask_coords_in_dm[1]):
        weight_ij = (bub_image_filtered[mask_coords[i][0], mask_coords[i][1]] +
                     bub_image_filtered[mask_coords[j][0], mask_coords[j][1]])

        # If intensity sum is non-zero, set weight inversely proportional to brightness
        if weight_ij != 0:
            weight_ij = dist_matrix[i, j] / weight_ij

        Graph_find_skeleton.add_edge(i, j, weight=weight_ij)

    # Extract minimum spanning tree (gives a loop-free structure)
    Tree = nx.minimum_spanning_tree(Graph_find_skeleton)
    return Graph_find_skeleton, Tree


def Get_Splice_Nodes_Pair(unwrapped_bub_data, mask_coords, Tree):
    """
    Identify candidate "splice" node pairs along the left and right boundaries of an unwrapped image.

    Motivation:
      In the unwrapped ellipse image, angle=0 and angle=360 correspond to opposite edges.
      A skeleton that represents a "wrapped" structure may enter/exit at these boundaries.
      This function selects nodes on both edges that share the same radial index (same row),
      and pairs them.

    Parameters
    ----------
    unwrapped_bub_data : 2D ndarray
        Unwrapped bubble image.
    mask_coords : (N, 2) ndarray
        Pixel coordinates belonging to a candidate region.
    Tree : networkx.Graph
        MST graph over mask pixels.

    Returns
    -------
    splice_nodes_pair : dict
        Mapping node_id -> paired_node_id (node_id < paired_node_id).
        If no strict row-matched pairs are found, a fallback pairing is attempted.
    """
    # Build candidate boundary pixel coordinates on leftmost and rightmost columns
    splice_points_start = []
    splice_points_end = []
    wrapped_radius = np.arange(mask_coords[:, 0].min(), mask_coords[:, 0].max() + 1)
    for i in wrapped_radius:
        wrapped_angle = mask_coords[np.where(mask_coords[:,0]==i)][:,1]
        if len(wrapped_angle) != 0:
            splice_points_start.append([i,wrapped_angle.min()])
            splice_points_end.append([i,wrapped_angle.max()])
    
    splice_points = np.r_[splice_points_start, splice_points_end]

    # Mark boundary pixels in a contour mask
    contour_data = np.zeros_like(unwrapped_bub_data, dtype='int32')
    for splice_point in splice_points:
        contour_data[np.int32(splice_point[0]), np.int32(splice_point[1])] = 1

    # Select nodes in the Tree that fall on the boundary
    splice_nodes = [
        node for node in Tree.nodes
        if Tree.degree(node) > 0 and contour_data[mask_coords[node][0], mask_coords[node][1]]
    ]

    # Pair nodes that share the same row but different column (left vs right edge)
    splice_nodes_pair = {}
    for node_1 in splice_nodes:
        for node_2 in splice_nodes:
            if mask_coords[node_1][0] == mask_coords[node_2][0] and mask_coords[node_1][1] != mask_coords[node_2][1]:
                splice_nodes_pair[node_1] = node_2
    splice_nodes_pair = {k: v for k, v in splice_nodes_pair.items() if k < v}

    # Fallback: if no strict pairs found, allow arbitrary pairing (still filtered by k < v)
    if len(splice_nodes_pair) == 0:
        splice_nodes_pair = {}
        for node_1 in splice_nodes:
            for node_2 in splice_nodes:
                splice_nodes_pair[node_1] = node_2
        splice_nodes_pair = {k: v for k, v in splice_nodes_pair.items() if k < v}

    return splice_nodes_pair


def Get_Bubble_Skeleton_Weighted_Unwrapped(unwrapped_bub_data, Tree, splice_nodes_pair,
                                          mask_coords, trim_logic=True):
    """
    Compute a weighted skeleton path in the unwrapped bubble domain.

    High-level:
      - For candidate boundary splice pairs, compute shortest paths on the MST,
      - Score each path by summing inverse edge weights (favor bright/intense ridges),
      - Choose the path with maximal score and optionally trim it.

    Parameters
    ----------
    unwrapped_bub_data : 2D ndarray
        Unwrapped intensity map (angles x radial-offset or similar).
    Tree : networkx.Graph
        Minimum spanning tree over the region pixels.
    splice_nodes_pair : dict
        Candidate start/end node pairs across unwrapped boundaries.
    mask_coords : (N, 2) ndarray
        Pixel coordinates corresponding to Tree nodes.
    trim_logic : bool
        Whether to apply Trim_Skeleton_Coords_2D() to clean the path.

    Returns
    -------
    skeleton_coords_2D : (M, 2) ndarray
        Skeleton path coordinates in unwrapped image coordinates.
    """
    unwrapped_data_shape = unwrapped_bub_data.shape

    if len(splice_nodes_pair.keys()) == 0:
        print('No splice_nodes_pair value.')

    paths_and_weights = []
    edge_len = 1
    break_logic = False

    # Search for valid boundary-crossing paths with additional edge constraints
    while len(paths_and_weights) == 0:
        for key in splice_nodes_pair:
            if nx.has_path(Tree, key, splice_nodes_pair[key]):
                path = nx.shortest_path(Tree, key, splice_nodes_pair[key])

                # Path score: sum of (1 / edge_weight) → larger = brighter preference
                path_weight = 0
                for k in range(len(path) - 1):
                    if Tree[path[k]][path[k + 1]]['weight'] != 0:
                        path_weight += 1 / Tree[path[k]][path[k + 1]]['weight']

                skeleton_coords_2D = mask_coords[path]

                # Heuristic constraints to avoid selecting paths that hug the boundaries too much
                if (skeleton_coords_2D[0][0] >= int(unwrapped_data_shape[0] / 3) and
                    skeleton_coords_2D[0][0] <= int(unwrapped_data_shape[0] - unwrapped_data_shape[0] / 3) and
                    0 not in skeleton_coords_2D[edge_len:][:, 1] and
                    skeleton_coords_2D[:, 1].max() not in skeleton_coords_2D[:-edge_len - 1][:, 1]):
                    paths_and_weights.append((path, path_weight))

        edge_len += 1

        # Fallback: relax constraints if no paths found
        if edge_len > unwrapped_data_shape[1] / 2:
            for key in splice_nodes_pair:
                if nx.has_path(Tree, key, splice_nodes_pair[key]):
                    path = nx.shortest_path(Tree, key, splice_nodes_pair[key])
                    path_weight = 0
                    for k in range(len(path) - 1):
                        if Tree[path[k]][path[k + 1]]['weight'] != 0:
                            path_weight += 1 / Tree[path[k]][path[k + 1]]['weight']
                        paths_and_weights.append((path, path_weight))

        # Hard break condition (rare, last resort)
        if edge_len > unwrapped_data_shape[1]:
            # NOTE: This branch looks suspicious in the original code (iterating over an int),
            # but we keep it unchanged as requested.
            for key in list(splice_nodes_pair.keys())[np.int64(len(splice_nodes_pair.keys()) / 2)]:
                if nx.has_path(Tree, key, splice_nodes_pair[key]):
                    path = nx.shortest_path(Tree, key, splice_nodes_pair[key])
                    path_weight = 0
                    for k in range(len(path) - 1):
                        if Tree[path[k]][path[k + 1]]['weight'] != 0:
                            path_weight += 1 / Tree[path[k]][path[k + 1]]['weight']
                        paths_and_weights.append((path, path_weight))
            break_logic = True
            break

    if not break_logic:
        # Choose the maximum-scoring candidate path
        max_weight, max_path, max_edges = FCFA.Search_Max_Path_And_Edges(paths_and_weights)
        skeleton_coords_2D = mask_coords[max_path]
    else:
        # NOTE: bubbleObj is not defined in this function in the original code.
        # This branch likely never executes in practice; kept unchanged.
        skeleton_coords_2D = np.c_[
            [np.int32(bubbleObj.unwrapped_bub_data.shape[0] / 2)] * bubbleObj.unwrapped_bub_data.shape[1],
            np.arange(bubbleObj.unwrapped_bub_data.shape[1])
        ]

    if trim_logic:
        skeleton_coords_2D, small_sc = Trim_Skeleton_Coords_2D(skeleton_coords_2D)

    return skeleton_coords_2D


def Get_Mapping_Skeleton_Coords(skeleton_coords_2D, unwrap_width):
    """
    Convert skeleton coordinates in unwrapped image space into
    (radial_offset, angle_degrees) arrays.

    Parameters
    ----------
    skeleton_coords_2D : (N, 2) ndarray
        Skeleton pixels in unwrapped image coords: [radial_index, angle_index]
    unwrap_width : int
        The centerline radial index offset used during unwrapping.

    Returns
    -------
    skeleton_coords_2D : (N, 2) ndarray
        Original unwrapped skeleton coords (unchanged).
    skeleton_radial : (N+1,) ndarray
        Radial offsets centered at 0 (radial_index - unwrap_width), closed by repeating first value.
    skeleton_angles : (N+1,) ndarray
        Angles in degrees mapped from angle_index to [0, 360], closed by repeating first value.
    """
    skeleton_radial = skeleton_coords_2D[:, 0]
    skeleton_angles = skeleton_coords_2D[:, 1]

    # Convert to offsets (center radial at 0) and angles in degrees
    skeleton_radial = skeleton_radial - unwrap_width
    skeleton_angles = skeleton_angles / skeleton_angles.max() * 360

    # Close the curve by appending the first point at the end
    skeleton_radial = np.r_[skeleton_radial, skeleton_radial[0]]
    skeleton_angles = np.r_[skeleton_angles, skeleton_angles[0]]

    return skeleton_coords_2D, skeleton_radial, skeleton_angles


def Cal_Angle_Between_Com_And_MaxValue(bubble_item, bubble_com_item, semi_major, semi_minor, ellipse_angle, unwrap_value='max'):
    """
    Determine the angular offset (in degrees) between the bubble center and the
    strongest (or weakest) point along the ellipse boundary.

    This is used to set a stable "starting angle" when unwrapping the ellipse.

    Parameters
    ----------
    bubble_item : 3D ndarray
        Bubble sub-cube (v,b,l). This function uses bubble_item.sum(0) as a 2D image.
    bubble_com_item : array-like
        Bubble center in item coordinates, likely [v, b, l] (or similar ordering).
    semi_major, semi_minor : float
        Ellipse semi-axes in pixels.
    ellipse_angle : float
        Ellipse rotation angle in radians (as used by Generate_Ellipse_Points).
    unwrap_value : {'max','min'}
        Whether to align angle offset to the maximum or minimum intensity location on the ellipse.

    Returns
    -------
    ellipse_points : (M, 2) ndarray
        Sampled ellipse boundary points.
    point_a : (2,) ndarray
        Selected boundary point (max/min intensity).
    delta_angle : float
        Angle offset (degrees in [0,360)).
    """
    bubble_item_sum = bubble_item.sum(0)

    # Note: center here is [x, y] for Generate_Ellipse_Points
    center = np.array([bubble_com_item[2], bubble_com_item[1]])
    ellipse_points, ellipse_infor = BFTools.Generate_Ellipse_Points(
        semi_major, semi_minor, ellipse_angle, center
    )

    # If ellipse boundary stays within image bounds, sample local neighborhood values
    if (np.int64(np.around(ellipse_points))[:, 1].max() < bubble_item_sum.shape[1] - 1) and \
       (np.int64(np.around(ellipse_points))[:, 0].max() < bubble_item_sum.shape[0] - 1):

        circle_values = []
        for circle_point in ellipse_points:
            neighbor_coords = BFTools.Generate_Neighbor_coords(
                np.int64(np.around(circle_point)),
                bubble_item_sum.shape
            )
            # Neighborhood sum around each ellipse point
            if len(neighbor_coords) != 0:
                circle_value = bubble_item_sum[neighbor_coords[:, 1], neighbor_coords[:, 0]].sum()
            else:
                circle_value = 0
            circle_values.append(circle_value)

        circle_values = np.array(circle_values)

        if unwrap_value == 'min':
            point_a = ellipse_points[np.argmin(circle_values[np.where(circle_values!=0)])]
        elif unwrap_value == 'max':
            point_a = ellipse_points[np.argmax(circle_values)]
        else:
            print('Choose the type of unwrap_value, min or max')

    else:
        # Fallback: if ellipse exceeds image boundary, pick global min/max of the non-zero bubble image
        if unwrap_value == 'min':
            point_a = np.where(bubble_item_sum == bubble_item_sum[bubble_item_sum > 0].min())
            point_a = np.c_[point_a[0], point_a[1]][0]
        elif unwrap_value == 'max':
            point_a = np.where(bubble_item_sum == bubble_item_sum[bubble_item_sum > 0].max())
            point_a = np.c_[point_a[0], point_a[1]][0]
        else:
            print('Choose the type of unwrap_value, min or max')

    # point_b is the center, reversed to [x, y] ordering to match point_a
    point_b = bubble_com_item[1:][::-1]

    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]

    # Convert to [0,360) degrees
    rad_angle = np.arctan2(dy, dx)
    delta_angle = np.degrees(rad_angle) % 360

    return ellipse_points, point_a, delta_angle


def Unwrap_Ellipse_Data(data, center_x, center_y, semi_major, semi_minor, theta,
                        width, angle_offset=0):
    """
    Unwrap a 2D image in an elliptical coordinate system around a given ellipse.

    Concept:
      - Parameterize the ellipse by angle t in [0, 2π),
      - At each t, compute the outward normal direction,
      - Sample data along normal offsets in [-width, +width] using bilinear interpolation,
      - Output is an "unwrapped" image (normal_offset x angle_position).

    Parameters
    ----------
    data : 2D ndarray
        Original 2D image (e.g., bubble_item.sum(0)).
    center_x, center_y : float
        Ellipse center (in image x/y coordinates).
    semi_major, semi_minor : float
        Ellipse semi-axes (a, b).
    theta : float
        Ellipse rotation angle in radians.
    width : float
        Half-width for sampling along the normal direction.
    angle_offset : float
        Starting angle shift in degrees (sets where unwrapped angle=0 begins).

    Returns
    -------
    unwrapped : 2D ndarray
        Unwrapped image of shape [n_normal, n_angles] after transpose (see code).
    arc_lengths : 1D ndarray
        Normalized arc-length position (0..1) along the ellipse.
    normal_dist : 1D ndarray
        Normal offsets sampled in [-width, +width].
    """
    # Ellipse perimeter approximation using complete elliptic integral
    e_squared = 1 - (semi_minor ** 2 / semi_major ** 2)
    E = ellipe(e_squared)
    perimeter = 4 * semi_major * E

    # Sampling counts: one sample per pixel along perimeter; normal samples per unit width
    n_angles = int(round(perimeter))
    n_normal = int(width * 2 + 1)

    # Parametric angles with starting offset
    t_angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False) + np.radians(angle_offset)
    normal_dist = np.linspace(-width, width, n_normal)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    unwrapped = np.zeros((n_angles, n_normal))
    arc_lengths = np.zeros(n_angles)

    for i, t in enumerate(t_angles):
        # Ellipse point in local coordinates
        x_local = semi_major * np.cos(t)
        y_local = semi_minor * np.sin(t)

        # Tangent vector in local coordinates
        dx_local = -semi_major * np.sin(t)
        dy_local = semi_minor * np.cos(t)

        # Rotate tangent into global coordinates
        dx_global = cos_theta * dx_local - sin_theta * dy_local
        dy_global = sin_theta * dx_local + cos_theta * dy_local

        # Unit normal vector perpendicular to tangent
        tangent_length = np.sqrt(dx_global ** 2 + dy_global ** 2)
        nx = -dy_global / tangent_length
        ny = dx_global / tangent_length

        # Ellipse point in global image coordinates
        x_ellipse = center_x + cos_theta * x_local - sin_theta * y_local
        y_ellipse = center_y + sin_theta * x_local + cos_theta * y_local

        # Approximate arc-length progression
        if i > 0:
            arc_lengths[i] = arc_lengths[i - 1] + tangent_length * (2 * np.pi / n_angles)

        # Sample along the normal direction using bilinear interpolation
        for j, n_offset in enumerate(normal_dist):
            x = x_ellipse + n_offset * nx
            y = y_ellipse + n_offset * ny

            if 0 <= x < data.shape[1] - 1 and 0 <= y < data.shape[0] - 1:
                x0, y0 = int(x), int(y)
                x1, y1 = x0 + 1, y0 + 1
                wx, wy = x - x0, y - y0

                value = ((1 - wx) * (1 - wy) * data[y0, x0] +
                         wx * (1 - wy) * data[y0, x1] +
                         (1 - wx) * wy * data[y1, x0] +
                         wx * wy * data[y1, x1])

                unwrapped[i, j] = value

    # Normalize arc-length to [0,1]
    if arc_lengths[-1] > 0:
        arc_lengths = arc_lengths / arc_lengths[-1]

    # Transpose to shape [normal, angle] (matches later usage)
    unwrapped = unwrapped.T

    # If width exceeds semi_minor, remove invalid inner rows (heuristic masking)
    if width > semi_minor:
        delta_width = int(round(width - semi_minor)) + 1
        unwrapped[:delta_width] = 0

    return unwrapped, arc_lengths, normal_dist


def Map_Curve_From_Unwrapped_Ellipse(curve_normal, curve_arc, center_x, center_y,
                                     semi_major, semi_minor, theta, angle_offset=0):
    """
    Map a curve defined in unwrapped-ellipse coordinates back to original image coordinates.

    Inputs describe a curve in:
      - curve_arc   : position along ellipse perimeter (0..1)
      - curve_normal: offset along the outward normal (pixels)

    Parameters
    ----------
    curve_normal : 1D ndarray
        Normal offsets (same unit as pixels).
    curve_arc : 1D ndarray
        Arc-length positions normalized to [0,1].
    center_x, center_y : float
        Ellipse center in the original image.
    semi_major, semi_minor : float
        Ellipse semi-axes.
    theta : float
        Ellipse rotation in radians.
    angle_offset : float
        Angle offset in degrees used during unwrapping.

    Returns
    -------
    curve_x, curve_y : 1D ndarray
        Mapped curve coordinates in the original image (float).
    """
    # Convert normalized arc position back to parametric angle t
    t_angles = curve_arc * 2 * np.pi + np.radians(angle_offset)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    curve_x = np.zeros_like(curve_arc)
    curve_y = np.zeros_like(curve_arc)

    for i, (t, n_offset) in enumerate(zip(t_angles, curve_normal)):
        # Ellipse point in local coords
        x_local = semi_major * np.cos(t)
        y_local = semi_minor * np.sin(t)

        # Tangent in local coords
        dx_local = -semi_major * np.sin(t)
        dy_local = semi_minor * np.cos(t)

        # Rotate tangent to global
        dx_global = cos_theta * dx_local - sin_theta * dy_local
        dy_global = sin_theta * dx_local + cos_theta * dy_local

        # Unit normal
        tangent_length = np.sqrt(dx_global ** 2 + dy_global ** 2)
        nx = -dy_global / tangent_length
        ny = dx_global / tangent_length

        # Ellipse point in global coords
        x_ellipse = center_x + cos_theta * x_local - sin_theta * y_local
        y_ellipse = center_y + sin_theta * x_local + cos_theta * y_local

        # Offset along normal
        curve_x[i] = x_ellipse + n_offset * nx
        curve_y[i] = y_ellipse + n_offset * ny

    return curve_x, curve_y


def Get_Bubble_Skeleton_Weighted(bubbleObj, unwrap_radius=None, unwrap_width=None, angle_offset=None,
                                 unwrapped_zero_col=11, unwrap_value='max', trim_logic=True):
    """
    High-level pipeline: compute bubble skeleton by unwrapping the bubble onto an ellipse.

    Steps:
      1) Determine ellipse parameters (from bubbleObj.ellipse_infor or fallback to circle),
      2) Choose a stable starting angle (align to max/min intensity on ellipse boundary),
      3) Unwrap bubble 2D projection into (normal_offset x arc_position) image,
      4) Find skeleton in unwrapped domain via MST + max-weight path,
      5) Map skeleton back to original image coordinates,
      6) Fit an ellipse to the mapped skeleton curve,
      7) Store all derived products into bubbleObj.

    Parameters
    ----------
    bubbleObj : object
        Should contain bubble_item, bubble_com_item, bubble_params_RFWHM, start_coords, etc.
    unwrap_radius : float or None
        If provided, forces circle radius for unwrapping.
    unwrap_width : int or None
        Half-width in pixels for normal sampling during unwrapping.
    angle_offset : float or None
        Starting angle in degrees for unwrapping.
    unwrapped_zero_col : int
        Used to zero out a fraction of radial rows near the top/bottom of unwrapped image.
    unwrap_value : {'max','min'}
        Choose starting angle by brightest/dimmest boundary point.
    trim_logic : bool
        Whether to trim the unwrapped skeleton path.

    Side Effects
    ------------
    Writes many attributes back into bubbleObj (unwrapped data, skeleton coords, fitted ellipse, etc.).
    """
    bubble_item = bubbleObj.bubble_item
    bubble_com_item = bubbleObj.bubble_com_item
    bubble_radius = bubbleObj.bubble_params_RFWHM['radius']
    bubble_thickness = bubbleObj.bubble_params_RFWHM['thickness']
    skeleton_coords_unwrapped = np.array([[0, 0]])

    # Determine ellipse parameters for unwrapping
    if unwrap_radius is not None:
        semi_major = unwrap_radius
        semi_minor = unwrap_radius
        ellipse_angle_rad = 0
    elif hasattr(bubbleObj, 'ellipse_infor'):
        ellipse_x00, ellipse_y00, ellipse_angle, semi_major_0, semi_minor_0 = bubbleObj.ellipse_infor
        semi_major = bubble_radius
        semi_minor = bubble_radius * np.abs(semi_minor_0 / semi_major_0)
        ellipse_angle_rad = np.radians(90 - ellipse_angle)
    else:
        semi_major = bubble_radius
        semi_minor = bubble_radius
        ellipse_angle_rad = 0

    # Unwrapping center in image coords (x=l, y=b ordering is assumed here)
    ellipse_x0, ellipse_y0 = bubble_com_item[2], bubble_com_item[1]

    # Choose angle_offset by locating the max/min value on ellipse boundary
    unwrap_ellipse_points, point_mvalue, delta_angle = Cal_Angle_Between_Com_And_MaxValue(
        bubble_item, bubble_com_item, semi_major, semi_minor, ellipse_angle_rad, unwrap_value
    )

    if unwrap_radius is None:
        unwrap_radius = bubble_radius
    if unwrap_width is None:
        unwrap_width = int(round(bubble_thickness / 2))
    if angle_offset is None:
        angle_offset = delta_angle

    # Unwrap the 2D projection (sum over v-axis)
    unwrapped_bub_data, arc_lengths, normal_distances = Unwrap_Ellipse_Data(
        bubble_item.sum(0), ellipse_x0, ellipse_y0, semi_major, semi_minor,
        ellipse_angle_rad, unwrap_width, angle_offset
    )

    # Mask out a band near top/bottom radial boundaries (heuristic cleanup)
    unwrapped_bub_data_shape = unwrapped_bub_data.shape
    zero_width = np.max([np.int16(np.around(unwrapped_bub_data_shape[0] / unwrapped_zero_col)),1])
    unwrapped_bub_data_T = unwrapped_bub_data.copy()
    unwrapped_bub_data_T[:zero_width, :] = 0
    unwrapped_bub_data_T[unwrapped_bub_data_shape[0] - zero_width:, :] = 0
    
    # Build connected components on the remaining unwrapped mask
    bub_mask = unwrapped_bub_data_T > 0
    bub_mask_label = measure.label(bub_mask, connectivity=2)
    regions_list = measure.regionprops(bub_mask_label)

    # For each connected region that crosses the midline (unwrap_width), compute skeleton
    for i in range(len(regions_list)):
        mask_coords = regions_list[i].coords
        if unwrap_width in mask_coords[:, 0] and len(mask_coords) > 2*unwrap_width:
            Graph_find_skeleton, Tree = Graph_Infor_Connected_Unwrapped_Bub(unwrapped_bub_data, mask_coords)
            splice_nodes_pair = Get_Splice_Nodes_Pair(unwrapped_bub_data, mask_coords, Tree)
            skeleton_coords_unwrapped_i = Get_Bubble_Skeleton_Weighted_Unwrapped(
                unwrapped_bub_data, Tree, splice_nodes_pair, mask_coords, trim_logic
            )

            skeleton_coords_unwrapped_i, skeleton_radial, skeleton_angles = Get_Mapping_Skeleton_Coords(
                skeleton_coords_unwrapped_i, unwrap_width
            )
            skeleton_coords_unwrapped = np.r_[skeleton_coords_unwrapped, skeleton_coords_unwrapped_i]
            
    skeleton_coords_unwrapped = skeleton_coords_unwrapped[1:]
    
    # Convert skeleton coordinates to (normal_offset, arc_position) in [0,1]
    skeleton_normal = skeleton_coords_unwrapped[:, 0] - unwrap_width
    skeleton_arc = skeleton_coords_unwrapped[:, 1] / skeleton_coords_unwrapped[:, 1].max()

    # Map skeleton back to original image coordinates
    skeleton_x, skeleton_y = Map_Curve_From_Unwrapped_Ellipse(
        skeleton_normal, skeleton_arc, ellipse_x0, ellipse_y0,
        semi_major, semi_minor, ellipse_angle_rad, angle_offset
    )
    skeleton_coords_original = np.c_[skeleton_y, skeleton_x]

    # Fit an ellipse to the mapped skeleton curve (algebraic fit)
    # paras = BFM.Solve_Ellipse(skeleton_coords_original[:, 0], skeleton_coords_original[:, 1])
    # skeleton_ellipse_infor, coords_fit, ellipse_perimeter = BFM.Solve_Ellipse_Infor(paras)
    # skeleton_coords_ellipse = np.c_[coords_fit[0], coords_fit[1]]
    skeleton_ellipse_infor, skeleton_coords_ellipse = BFM.Fit_Ellipse_Geometric(skeleton_coords_original, center=None)

    # Convert back to global coordinates by adding the bubble item start offset
    start_coords_item = bubbleObj.start_coords
    skeleton_ellipse_infor[0] = skeleton_ellipse_infor[0] + start_coords_item[1]
    skeleton_ellipse_infor[1] = skeleton_ellipse_infor[1] + start_coords_item[2]
    skeleton_coords_original = np.c_[skeleton_y, skeleton_x] + start_coords_item[1:]
    skeleton_coords_ellipse = skeleton_coords_ellipse + start_coords_item[1:]

    # Store all products into bubbleObj
    bubbleObj.unwrap_ellipse_points = unwrap_ellipse_points
    bubbleObj.point_mvalue = point_mvalue
    bubbleObj.unwrap_radius = unwrap_radius
    bubbleObj.unwrap_width = unwrap_width
    bubbleObj.angle_offset = angle_offset
    bubbleObj.unwrapped_bub_data = unwrapped_bub_data
    bubbleObj.skeleton_coords_unwrapped = skeleton_coords_unwrapped
    bubbleObj.skeleton_coords_original = skeleton_coords_original
    bubbleObj.skeleton_ellipse_infor = skeleton_ellipse_infor
    bubbleObj.skeleton_coords_ellipse = skeleton_coords_ellipse


def Update_Contour_And_Ellipse_Infor(bubbleObj, add_con):
    """
    Update bubble contour, ellipse information, and radial cut dictionaries
    using skeleton-based ellipse fitting results.

    This function replaces the morphology-based ellipse/contour with the
    skeleton-based ellipse/contour, then recomputes profile cuts along
    ellipse normals.

    Parameters
    ----------
    bubbleObj : object
        Bubble object containing clump, skeleton, and morphology information.
        The following attributes are expected to exist:
            - clumpsObj.origin_data
            - clumpsObj.regions_data
            - clumpsObj.data_wcs
            - clumpsObj.connected_ids_dict
            - clumpsObj.clump_coords_dict
            - skeleton_ellipse_infor
            - skeleton_coords_ellipse
            - skeleton_coords_original
            - start_coords
            - bubble_clump_ids
            - bubble_clump_ids_con
    add_con : bool
        If True, include connected clumps when extracting gas data
        and computing radial cuts.
    """

    # Shortcuts to frequently used data
    origin_data = bubbleObj.clumpsObj.origin_data
    regions_data = bubbleObj.clumpsObj.regions_data
    data_wcs = bubbleObj.clumpsObj.data_wcs
    connected_ids_dict = bubbleObj.clumpsObj.connected_ids_dict
    clump_coords_dict = bubbleObj.clumpsObj.clump_coords_dict

    # Replace ellipse and contour information with skeleton-based results
    bubbleObj.ellipse_infor = bubbleObj.skeleton_ellipse_infor
    bubbleObj.ellipse_coords = bubbleObj.skeleton_coords_ellipse

    # Convert original skeleton contour into local (sub-cube) coordinates
    bubbleObj.bubble_contour = (
        bubbleObj.skeleton_coords_original - bubbleObj.start_coords[1:]
    )

    # Bubble-related clump IDs
    bubble_clump_ids = bubbleObj.bubble_clump_ids
    bubble_clump_ids_con = bubbleObj.bubble_clump_ids_con

    # Container for radial cut results
    dictionary_cuts = defaultdict(list)

    # Update bubble center in local coordinates:
    # keep velocity coordinate unchanged, but replace (b, l) with ellipse centroid
    bubble_com_item = bubbleObj.bubble_com_item
    ellipse_com = np.mean(
        bubbleObj.ellipse_coords - bubbleObj.start_coords[1:], axis=0
    )
    bubbleObj.bubble_com_item = np.array([bubble_com_item[0], ellipse_com[0], ellipse_com[1]])

    # Determine which clumps to include
    clump_ids = list(bubble_clump_ids)
    if add_con:
        clump_ids = list(bubble_clump_ids) + list(bubble_clump_ids_con)

    # Extract a local data cube covering the selected clumps
    bubble_coords,bubble_item,data_wcs_item,regions_data_i,start_coords,bubble_item_mask_2D,lb_area = FCFA.Filament_Coords(
        origin_data,regions_data,data_wcs,clump_coords_dict,clump_ids)

    # Unpack ellipse parameters from skeleton fitting
    ellipse_x0, ellipse_y0, ellipse_angle, a_res, b_res = bubbleObj.skeleton_ellipse_infor

    # Compute derivatives along the ellipse (tangent and normal directions)
    points, fprime, points_b = BFPr.Cal_Derivative(ellipse_x0,ellipse_y0,bubbleObj.skeleton_coords_ellipse,start_coords)

    # Compute radial/profile cuts along ellipse normals
    dictionary_cuts = BFPr.Cal_Dictionary_Cuts(regions_data,clump_ids,connected_ids_dict,clump_coords_dict,
        points,points_b,fprime,bubble_item.sum(0),bubble_item_mask_2D,dictionary_cuts,start_coords)

    # Store updated cut dictionary
    bubbleObj.dictionary_cuts = dictionary_cuts


def Get_Expanded_Boundary_Intersection(angle, center_x, center_y, ellipse_rot, a, b, width, is_outer):
    """
    Compute the intersection point between a ray and the expanded (or contracted)
    boundary of an ellipse.

    The boundary is defined by offsetting the ellipse outward or inward along
    the local normal direction, forming a “shadow / expanded region”.

    Parameters
    ----------
    angle : float
        Ray direction angle in degrees (measured in the global coordinate system).
    center_x, center_y : float
        Center of the ellipse in global coordinates.
    ellipse_rot : float
        Rotation angle of the ellipse (radians).
        This is the rotation from the global frame to the ellipse local frame.
    a : float
        Semi-major axis of the ellipse.
    b : float
        Semi-minor axis of the ellipse.
    width : float
        Expansion (or contraction) width along the normal direction.
    is_outer : bool
        True  -> intersection with the outer expanded boundary.
        False -> intersection with the inner contracted boundary.

    Returns
    -------
    x_int, y_int : float
        Intersection point coordinates in the global frame.
    """

    # Ensure angles are in radians
    ellipse_rot_rad = ellipse_rot
    angle_rad = np.radians(angle)

    # Rotation matrix components (global -> ellipse local frame)
    cos_rot = np.cos(ellipse_rot_rad)
    sin_rot = np.sin(ellipse_rot_rad)

    # Ray direction unit vector in the global frame
    dir_x = np.cos(angle_rad)
    dir_y = np.sin(angle_rad)

    def boundary_eq(r):
        """
        Implicit equation for the expanded/contracted ellipse boundary
        evaluated at a point along the ray.

        Solving boundary_eq(r) = 0 gives the intersection distance r.
        """

        # Point on the ray in global coordinates
        x_g = center_x + r * dir_x
        y_g = center_y + r * dir_y

        # Translate to ellipse-centered coordinates
        dx = x_g - center_x
        dy = y_g - center_y

        # Rotate into ellipse local coordinates
        # (ellipse center at origin, major axis aligned with x)
        x_l = dx * cos_rot + dy * sin_rot
        y_l = -dx * sin_rot + dy * cos_rot

        # Parametric angle on the ellipse corresponding to this point
        t = np.arctan2(y_l, x_l)

        # Tangent vector of the ellipse at parameter t (local frame)
        dx_dt = -a * np.sin(t)
        dy_dt =  b * np.cos(t)
        tangent_len = np.sqrt(dx_dt**2 + dy_dt**2)

        # Outward normal vector (local frame, unit length)
        nx_l = -dy_dt / tangent_len
        ny_l =  dx_dt / tangent_len

        # Offset direction:
        # +width for outer boundary, -width for inner boundary
        offset = width if is_outer else -width

        # Effective semi-axes of the expanded/contracted boundary
        # Projection of the normal onto x/y directions is used
        a_bound = a + offset * np.abs(nx_l)
        b_bound = b + offset * np.abs(ny_l)

        # Implicit ellipse equation:
        # = 0 when the point lies exactly on the expanded boundary
        return (x_l / a_bound)**2 + (y_l / b_bound)**2 - 1

    # Initial guess for the ray parameter r
    # Different guesses for inner/outer boundary improve convergence
    r_guess = (a + width) if is_outer else (a - width)

    # Solve for r such that boundary_eq(r) = 0
    r = fsolve(boundary_eq, r_guess)[0]

    # Convert back to global coordinates
    return center_x + r * dir_x, center_y + r * dir_y






