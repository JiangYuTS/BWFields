import time, os, sys, contextlib
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from skimage import measure


# External clump & filament classes (from DPConCFil package)
from DPConCFil.Clump_Class import *
from DPConCFil.Filament_Class import *

# Internal modules for bubble analysis
from . import Bubble_Funs_Morphology as BFM   # bubble morphology + masks + ellipse
from . import Bubble_Funs_PV as BFPV          # position-velocity related utilities (not used in this snippet)
from . import Bubble_Funs_Table as BFT        # table IO / formatting (not used in this snippet)
from . import Bubble_Funs_Profile as BFPr     # profile extraction along ellipse / derivatives
from . import Bubble_Funs_Skeleton as BFS     # skeleton utilities (not used in this snippet)
from . import Bubble_Funs_Tools as BFTools    # misc WCS/range helpers


class BubbleInfor(object):
    """
    Core class for bubble detection, morphology analysis,
    and gas/kinematic property extraction based on clump information.

    Main data sources
    -----------------
    clumpsObj.origin_data    : 3D data cube (v, b, l)
    clumpsObj.data_wcs       : WCS of the data cube
    clumpsObj.signal_regions : signal regions (SRs)

    Typical workflow
    ----------------
    1. Bubble_Morphology()
       - Detect bubbles and compute morphology
    2. Get_Bubble_Item_Morphology()
       - Extract sub-cubes, fit ellipse, derive gas/velocity properties
    """
    def __init__(self, clumpsObj=None, parameters=None, save_files=None,
                 save=False, pix_scale_arcmin=0.5, vel_resolution=0.167,ClumpNum=3):
        # External clump object produced by DPConCFil (provides cube, WCS, SRs, clump coords...)
        self.clumpsObj = clumpsObj
        # Bubble detection parameters passed to BFM.Bubble_Weight_Data_Detect_By_SR
        self.parameters = parameters
        # Output file paths (e.g. mask fits, tables) used when save=True
        self.save_files = save_files
        self.save = save
        # Spatial/velocity resolution metadata used downstream (e.g. physical conversions)
        self.pix_scale_arcmin = pix_scale_arcmin
        self.vel_resolution = vel_resolution
        self.ClumpNum = ClumpNum

    def Bubble_Morphology(self, srs_ids=None, bubble_weight_data=None, bubble_regions_data=None, 
                          bubbles_coords=None, par_FacetClumps_bub=[False]):
        """
        Detect bubbles and compute morphology properties.
    
        Parameters
        ----------
        srs_ids : list/ndarray, optional
            If provided, only process selected SR indices.
        bubble_weight_data : ndarray, optional
            Precomputed bubble weight cube; if given, skip detection stage.
        bubbles_coords : list, optional
            Precomputed voxel coordinates of bubbles; used with bubble_weight_data.
        par_FacetClumps_bub : list
            [0]==True enables FacetClumps refinement of bubble regions.
    
        Side effects (saved on self)
        ----------------------------
        self.bubble_infor          : dict of morphology results
        self.bubble_regions_data   : labeled cube/mask for bubble regions
        self.bubble_regions        : regionprops-like objects (coords, bbox, etc.)
        plus a collection of convenience fields (coms, radius, contours, ...)
        """
        origin_data = self.clumpsObj.origin_data
        data_wcs = self.clumpsObj.data_wcs
        srs_array = self.clumpsObj.signal_regions_array
        srs_list = self.clumpsObj.signal_regions_list
        parameters = self.parameters
        start_1 = time.time()
    
        # Restrict to selected signal regions (SRs) if specified
        if srs_ids is not None and (len(srs_ids) if isinstance(srs_ids, (list, np.ndarray)) else True):
            srs_arr = np.array(srs_list)[srs_ids]
            srs_list = [srs_arr] if np.isscalar(srs_arr) else srs_arr
    
        # Stage 1: detect bubble weight cube and bubble voxel coords (if not provided)
        if bubble_weight_data is None:
            bubble_weight_data_no_filter, bubble_weight_data, \
            bubbles_coords, morphology_times = \
                            BFM.Bubble_Weight_Data_Detect_By_SR(srs_list, origin_data, parameters)
    
            # Optional refinement: use FacetClumps to adjust bubble regions (improve masks/edges)
            if par_FacetClumps_bub[0]:
                with Suppress_Tqdm():
                    bubble_regions_data, bubbles_coords, srs_array_bub, srs_list_bub = \
                        BFM.Get_Bubble_Regions_By_FacetClumps(
                            srs_array, bubble_weight_data_no_filter,bubble_weight_data, par_FacetClumps_bub)
            else:
                bubble_regions_data = measure.label(bubble_weight_data > 0, connectivity=3)
                bubbles_coords = []
                bubble_weight_regions = measure.regionprops(bubble_regions_data)
                for sreg in bubble_weight_regions:
                    bubbles_coords.append(sreg.coords)
                srs_array_bub = None
                srs_list_bub = None
    
            # Cache intermediate products for later debugging/reuse
            self.bubble_weight_data_no_filter = bubble_weight_data_no_filter
            self.bubble_weight_data = bubble_weight_data
            self.bubbles_coords = bubbles_coords
            self.morphology_times = morphology_times
            self.srs_array_bub = srs_array_bub
            self.srs_list_bub = srs_list_bub
        else:
            if bubbles_coords is None:
                bubbles_coords = []
                bubble_weight_regions = measure.regionprops(bubble_regions_data)
                for sreg in bubble_weight_regions:
                    bubbles_coords.append(sreg.coords)
            self.bubble_weight_data = bubble_weight_data
            self.bubbles_coords = bubbles_coords
            self.morphology_times = 0
            self.srs_array_bub = None
            self.srs_list_bub = None
    
        # Stage 2: compute morphology features per bubble (center, contour, ecc, volume, confidence...)        
        bubble_infor, bubble_regions_data, bubble_regions = \
            BFM.Bubble_Infor_Morphology(bubble_weight_data, bubbles_coords)

        if self.srs_array_bub is not None:
            bubble_peaks = bubble_infor['bubble_peaks']
            rc_dict_bub = Clump_Class_Funs.Build_RC_Dict_Simplified(bubble_peaks, srs_array_bub, srs_list_bub)
            self.rc_dict_bub = rc_dict_bub
    
        # Stage 3: convert pixel-domain morphology results into WCS coordinates
        bubble_infor = BFM.Bubble_Infor_Morphology_WCS(data_wcs, bubble_infor)
    
        # Cache morphology results in self for downstream use
        self.bubble_infor = bubble_infor
        self.bubble_coms = bubble_infor['bubble_coms']                     # (v,b,l) or similar pixel COMs
        self.bubble_coms_wcs = bubble_infor['bubble_coms_wcs']             # COMs in world coords
        self.ranges_v = bubble_infor['ranges_v']                           # velocity ranges in pixels
        self.ranges_v_wcs = bubble_infor['ranges_v_wcs']                   # velocity ranges in WCS
        self.radius_lb = bubble_infor['radius_lb']                         # bubble radius in (l,b) plane
        self.areas_lb = bubble_infor['areas_lb']                           # projected area in (l,b)
        self.contours = bubble_infor['contours']                           # 2D contour points in (l,b)
        self.eccentricities_lb = bubble_infor['eccentricities_lb']         # ellipse eccentricity (lb-plane)
        self.volume = bubble_infor['volume']                               # voxel volume estimate
        self.confidences = bubble_infor['confidences']                     # detection confidence metric(s)
        self.bubble_regions_data = bubble_regions_data
        self.bubble_regions = bubble_regions
    
        # Compute bubble weights (e.g. mean intensity) and store in self via BFM
        BFM.Cal_Bub_Weights(self, type='mean')
    
        # Optional output: save region label cube as FITS
        if self.save:
            fits.writeto(self.save_files[0], bubble_regions_data, overwrite=True)
    
        end_1 = time.time()
        print('Number:', len(self.bubble_coms))
        print('Time:', np.around(end_1 - start_1, 2))
        

    def Get_Bubble_Item_Morphology(self, index, dictionary_cuts, add_con=False,
                                   bubble_infor_provided=None, systemic_v_type=1, 
                                   fixed_ec=True, half_ecoords=False, CalSub=False,
                                   shift=False, ellipse_start_type='max',
                                   skeleton_ellipse_infor=None, skeleton_coords_ellipse=None):
        """
        Analyze a single bubble (by index) and construct its geometry, gas properties,
        and profile sampling structure.
    
        This function is the **core per-bubble analysis pipeline**, bridging:
            morphology → ellipse geometry → gas extraction → profile construction
    
        ------------------------------------------------------------------
        Workflow Overview
        ------------------------------------------------------------------
        (1) Identify associated clumps
            - From bubble region → clump IDs
            - Optionally include connected clumps (graph-based extension)
    
        (2) Extract local data cube
            - Sub-cube centered on relevant clumps
            - Includes: intensity cube, WCS, 2D mask, spatial extent
    
        (3) Determine ellipse geometry
            Three modes:
            - Default: fit ellipse from detected bubble contour
            - Provided: use externally supplied center/radius/ellipse
            - Skeleton: override using skeleton-derived ellipse
    
        (4) Compute gas & kinematic properties
            - Gas center (pixel + WCS)
            - Velocity structure
            - Integrated spectra
    
        (5) Build profile sampling structure
            - Generate points along ellipse
            - Compute normal/tangent directions
            - Construct radial/azimuthal cuts (dictionary_cuts)
    
        ------------------------------------------------------------------
        Parameters
        ------------------------------------------------------------------
        index : int
            Bubble index in self.bubble_regions / self.bubble_coms.
    
        dictionary_cuts : dict
            Accumulator storing radial/azimuthal sampling results.
            Will be updated and returned.
    
        add_con : bool
            If True, include clumps connected to the bubble (graph expansion).
    
        bubble_infor_provided : list or None
            External bubble definition:
                [bubble_clump_ids, bubble_com_pi, bubble_com_wcs_pi, bubble_radius]
            If provided, skips internal morphology & ellipse fitting.
    
        systemic_v_type : int
            Method to define systemic velocity:
                1 → from gas COM (recommended)
                2 → from bubble center
    
        fixed_ec : bool
            If True, fix ellipse center to bubble COM in (b,l) plane.
            Otherwise center is free during fitting.
    
        half_ecoords : bool
            If True, only use half of ellipse points (avoid redundancy).
    
        CalSub : bool
            If True, use background-subtracted cube in Filament_Coords.
    
        shift : bool
            Apply spatial shift when constructing profile cuts.
    
        ellipse_start_type : {'max', 'min', ...}
            Defines how ellipse starting point is reordered based on gas intensity.
    
        skeleton_ellipse_infor : optional
            If provided, overrides ellipse fitting with skeleton-based ellipse.
    
        skeleton_coords_ellipse : optional
            Coordinates corresponding to skeleton-based ellipse.
    
        ------------------------------------------------------------------
        Notes
        ------------------------------------------------------------------
        - Coordinate systems:
            * Global cube coords: (v, b, l)
            * Local item coords: shifted by start_coords
        - Clump IDs are zero-based internally.
        - This function updates many attributes in `self` for downstream use.
    
        ------------------------------------------------------------------
        Outputs (stored in self)
        ------------------------------------------------------------------
        Geometry:
            ellipse_infor, ellipse_coords, bubble_contour
    
        Gas:
            bubble_gas_* (center, spectra, ranges)
    
        Data:
            bubble_item, bubble_coords, mask, WCS, etc.
    
        Profiles:
            dictionary_cuts (radial / azimuthal sampling structure)
        """
        # --- Clump-level data shortcuts (from clumpsObj) ---
        origin_data = self.clumpsObj.origin_data
        regions_data = self.clumpsObj.regions_data
        clump_centers = self.clumpsObj.centers
        clump_centers_wcs = self.clumpsObj.centers_wcs
        data_wcs = self.clumpsObj.data_wcs
        clump_coords_dict = self.clumpsObj.clump_coords_dict
        connected_ids_dict = self.clumpsObj.connected_ids_dict
    
        # --- Bubble-level cached results (from Bubble_Morphology) ---
        bubble_regions_data = self.bubble_regions_data
        bubble_weight_data = self.bubble_weight_data
        bubble_coms = self.bubble_coms
        bubble_coms_wcs = self.bubble_coms_wcs
        bubble_regions = self.bubble_regions
        self.index = index
    
        # If no external bubble info is provided: use region coords to extract bubble inner data
        if bubble_infor_provided is None:
            bubble_region = bubble_regions[index]
            bubble_clump_ids = None
            # Extract cube cutout around bubble coords (intensity weights)
            self.bubble_inner_data_item, self.start_coords_inner = \
                BFM.Get_Bubble_Inner_Item(bubble_weight_data, bubble_region.coords)
            # Extract corresponding region labels cutout (mask/labels)
            self.bubble_inner_region_item, _ = \
                BFM.Get_Bubble_Inner_Item(bubble_regions_data, bubble_region.coords)
        else:
            bubble_region = None
            bubble_clump_ids = bubble_infor_provided[0]  # externally provided clump ids or mapping
    
        # Ellipse fitting: either compute from bubble contour or load from provided info
        if bubble_infor_provided is None:
            bubble_coms_bl = bubble_coms[index][1:] if fixed_ec else None  # ellipse center in (b,l)
            ellipse_infor, ellipse_coords = \
                BFM.Get_Ellipse_Coords(
                    bubble_coms_bl, bubble_region, bubble_weight_data)
    
            # Convert global coords into item-local coords via start_coords offset
            self.bubble_com_item = bubble_coms[index]
            self.bubble_com_item_wcs = bubble_coms_wcs[index]
            self.bubble_contour = self.contours[index]
        else:
            bubble_com_pi, bubble_com_wcs_pi, _, \
            ellipse_coords, ellipse_infor = \
                BFM.Get_Bubble_Ellipse_Infor_By_Provide(
                    self, bubble_infor_provided)
    
            self.bubble_com_item = bubble_com_pi
            self.bubble_com_item_wcs = bubble_com_wcs_pi
            self.bubble_contour = ellipse_coords
    
        self.ellipse_infor = ellipse_infor     # (x0, y0, angle, a, b)
        self.ellipse_coords = ellipse_coords   # ellipse sample points in item coords
        self.ellipse_infor_cav = ellipse_infor   
        self.ellipse_coords_cav = ellipse_coords 
    
        # Find clumps associated with this bubble (and optionally connected clumps)
        bubble_clump_ids, bubble_clump_ids_con = \
            BFM.Get_Bubble_Clump_Ids(
                self, bubble_region, clump_centers, regions_data,
                connected_ids_dict, bubble_clump_ids)
    
        if skeleton_ellipse_infor is not None:
            ellipse_infor = skeleton_ellipse_infor
            ellipse_coords = skeleton_coords_ellipse
    
        self.ellipse_infor = ellipse_infor     # (x0, y0, angle, a, b) 
        self.ellipse_coords = ellipse_coords   # ellipse sample points in item coords
    
        self.bubble_clump_ids = bubble_clump_ids
        self.bubble_clump_ids_con = bubble_clump_ids_con
    
        # If no clumps are related, exit early
        if len(bubble_clump_ids) == 0:
            print('No related clumps!')
            return
    
        # Compose clump id list used for extraction
        clump_ids = list(bubble_clump_ids)
        if add_con:
            clump_ids += list(bubble_clump_ids_con)
    
        # Extract sub-cube around selected clumps; also returns 2D mask and spatial area in lb-plane
        bubble_coords, bubble_item, data_wcs_item, regions_data_i, \
        start_coords, bubble_item_mask_2D, lb_area = \
            FCFA.Filament_Coords(origin_data, regions_data, data_wcs,
                clump_coords_dict, clump_ids, CalSub)
    
        self.bubble_com_item = self.bubble_com_item - start_coords
        self.bubble_contour = self.bubble_contour - start_coords[1:]
    
        # Gas/kinematic info extraction + coordinate ordering cleanup
        BFM.Get_Bubble_Gas_Infor(self, index, systemic_v_type)
        BFM.Resort_Ellipse_Coords(self, add_con, ellipse_start_type)
    
        # Optionally use only half ellipse points
        ellipse_coords = self.ellipse_coords
        if half_ecoords:
            ellipse_coords = ellipse_coords[:len(ellipse_coords)//2]
    
        # Derivative along ellipse for profile extraction (tangent/normal directions)
        ellipse_x0, ellipse_y0, ellipse_angle, a_res, b_res = ellipse_infor
        points, fprime, points_b = \
            BFPr.Cal_Derivative(ellipse_x0, ellipse_y0, ellipse_coords, start_coords)
    
        # Build/update dictionary of cuts used by later profile fitting (radial/azimuthal sampling)
        dictionary_cuts = BFPr.Cal_Dictionary_Cuts(
            regions_data, clump_ids, connected_ids_dict,
            clump_coords_dict, points, points_b, fprime,
            bubble_item.sum(0), bubble_item_mask_2D,
            dictionary_cuts, start_coords, CalSub, shift)
    
        # Compute WCS range/pixel scale for the extracted item cube
        lbv_item_start, lbv_item_end, velocity_axis, pixel_scale = \
            BFTools.Cal_Item_WCS_Range(bubble_item, data_wcs_item)
    
        # Cache item-level products for external access/plotting
        self.dictionary_cuts = dictionary_cuts
        self.bubble_coords = bubble_coords
        self.bubble_item = bubble_item
        self.data_wcs_item = data_wcs_item
        self.regions_data_i = regions_data_i
        self.start_coords = start_coords
        self.bubble_item_mask_2D = bubble_item_mask_2D
        self.lbv_item_start = lbv_item_start
        self.lbv_item_end = lbv_item_end
        self.velocity_axis = velocity_axis
        self.pixel_scale = pixel_scale
                    

    def Bubble_Detect(parameters, save_files, file_path):
        """One-shot bubble detection pipeline from a FITS cube path.

        Notes
        -----
        This looks like a legacy/static helper:
        - It does not use `self` (consider adding @staticmethod or moving out of class).
        - Several variable names appear inconsistent (see FIXME below).

        Returns
        -------
        (bubble_infor, Bubble_Table_Pix, Bubble_Table_WCS) or (None, None, None)
        """
        start_1 = time.time()
        start_2 = time.ctime()

        # Load FITS data cube
        real_data = fits.getdata(file_path)

        # Unpack parameters (order must match caller)
        RMS, Threshold, Sigma, BubbleSize, SlicedVS, Weight = parameters[:6]

        # Output file paths
        mask_name, weight_mask_name, bubble_table_pix_name, \
        bubble_table_wcs_name, bubble_infor_name = save_files[:5]

        # Build weighted mask and derive bubble morphology summary
        bubble_mask, bubble_sliced_mask = Bubble_Weighted_Mask(
            RMS, Threshold, Sigma, SlicedVS, BubbleSize, real_data)
        bubble_infor = Bubble_Infor(Weight, BubbleSize, bubble_mask, bubble_sliced_mask)
        bubble_infor = Bubble_Infor_WCS(file_path, bubble_infor)
        np.savez(bubble_infor_name, bubble_infor=bubble_infor)

        # Bubble centers in pixel coords
        bubble_coms = bubble_infor['bubble_coms']

        # FIXME: `coms_lbv` is undefined; probably meant `bubble_coms`?
        if len(coms_lbv) != 0:
            bubble_weight_mask = bubble_infor['bubble_weight_mask']
            bubble_regions_data = bubble_infor['bubble_regions_data']

            # Build output tables (pixel + WCS)
            Bubble_Table_Pix = Table_Interface_Pix(bubble_infor)
            Bubble_Table_WCS = Table_Interface_WCS(bubble_infor)

            # Save outputs
            fits.writeto(mask_name, bubble_regions_data, overwrite=True)
            fits.writeto(weight_mask_name, bubble_weight_mask, overwrite=True)
            Bubble_Table_Pix.write(bubble_table_pix_name, overwrite=True)
            Bubble_Table_WCS.write(bubble_table_wcs_name, overwrite=True)

            # Record timing
            end_1 = time.time()
            end_2 = time.ctime()
            delta_time = np.around(end_1 - start_1, 2)
            time_record = np.hstack([[start_2, end_2, delta_time]])
            time_record = Table(time_record, names=['Start', 'End', 'DTime'])
            time_record.write(mask_name[:-4] + 'time_record.csv', overwrite=True)

            print('Number:', len(bubble_infor['bubble_coms']))
            print('Time:', delta_time)
            return bubble_infor, Bubble_Table_Pix, Bubble_Table_WCS
        else:
            print('Number:', len(bubble_infor['bubble_coms']))
            return None, None, None


@contextlib.contextmanager
def Suppress_Tqdm():
    """Suppress tqdm progress bar output (avoid clutter in nested/parallel routines)."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr
