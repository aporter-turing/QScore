from nibabel.nifti1 import Nifti1Image
from pathlib import Path
from nipype.interfaces import fsl
import logging
import json 
import numpy as np 
from nipype.interfaces.ants import ResampleImageBySpacing, ApplyTransforms


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

contrast_num = {"objnam":1, "motor":2}

class QScore:
    def __init__(self, base_folder: Path, output_path: Path, original_nifti: Path):
        """
        base_folder: Path to folder to python package
        output_path: Path to output data directory, where FEAT and NIFTI files will be written
        original_nifti: Path to original NIFTI file
        """        

        self.base_folder = base_folder
        # ensure nifti is in 4mm space
        resample(original_nifti, output_path, output_path / 'fake_anat.nii.gz', output_path / 'func_4mm.nii.gz')
        self.task_type = get_series_description(original_nifti)
        if self.task_type == "error":
            raise Exception("Error reading task type from NIFTI file name")
        self.num_contrasts = contrast_num[self.task_type]
        self.design_file_path = self.base_folder / f"design/{self.task_type}/{self.task_type}.fsf"

        self.output_path = output_path
        self.analysis_path = self.output_path / "real"

        # Make sure analysis path directory exists
        if not self.analysis_path.exists():
            self.analysis_path.mkdir(parents=True, exist_ok=True)

        original_image = Nifti1Image.load(output_path / 'func_4mm.nii.gz')
        self.tr_time = original_image.header['pixdim'][4]

        # Truncate the original data by 4 frames
        truncated_image = original_image.slicer[:,:,:,:-4]
        truncated_image_file_path = self.output_path / "truncated.nii.gz"
        truncated_image.to_filename(truncated_image_file_path)
        self.filtered_data = truncated_image_file_path
        self.num_frames = truncated_image.shape[3]
        # Logging setup
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    def run_feat(self):
        """
        Modify a base FEAT design file (.fsf) with subject-specific parameters
        and run the FSL FEAT analysis.

        Args:
            num_frames: Number of volumes in the fMRI dataset.
            tr_time: Repetition time (TR) in seconds.
            filtered: Path to the subject-specific 4D NIfTI file.
            design_file_path: Path to the base .fsf file to modify.
            analysis_path: Directory where FEAT outputs should go.
            design_file_with_confound_path: Optional path to a text file of confound regressors.
        """
        logging.info(f"Preparing to run FEAT for {self.filtered_data}")
        new_design_file_path = self.output_path / "real_new.fsf"

        # Build the dictionary of parameters to insert into the .fsf template
        search_replace_mapping = {
            'set fmri(outputdir) ': f'set fmri(outputdir) "{self.analysis_path}"',
            'set feat_files(1) ': f'set feat_files(1) "{self.filtered_data}"',
            'set fmri(tr) ': f'set fmri(tr) {self.tr_time}',
            'set fmri(npts) ': f'set fmri(npts) {self.num_frames}',
        }


        # Edit the design file line by line, applying replacements
        with open(self.design_file_path, 'r') as old_design_file, open(new_design_file_path, 'w') as new_design_file:
            for line in old_design_file:
                found = False
                for search_string, replacement_text in search_replace_mapping.items():
                    if search_string in line:
                        new_design_file.write(replacement_text)
                        found = True
                if not found:
                    new_design_file.write(line)

        logging.info(f"Running FEAT with design file: {new_design_file_path}")

        feat = fsl.FEAT()
        feat.inputs.fsf_file = str(new_design_file_path)
        feat.run()

    def do_registration(self):
        """
        Register zstat images to standard space using FSL tools:
        1. Extract example_func from filtered_func_data
        2. Prepare standard brain image
        3. Compute transform (FLIRT) between example_func and standard
        4. Invert the transform (ConvertXFM)
        5. Apply transform to zstat maps

        Args:
            num_contrasts: Number of contrasts (e.g., 1 or 2)
        """

        output_path = self.output_path
        standard_brain = Path(self.base_folder) / "design/standard/MNI152_T1_4mm_brain.nii.gz"

        # Step 1: Extract example_func image (volume 78)
        example_func_path = output_path / "example_func.nii.gz"
        fslroi = fsl.ExtractROI(
            in_file=str(output_path / "real.feat" / "filtered_func_data.nii.gz"),
            roi_file=str(example_func_path),
            t_min=78,
            t_size=1
        )
        fslroi.run()

        # Step 2: Prepare standard brain image (copy to local output)
        standard_copy_path = output_path / "standard.nii.gz"
        fslmaths = fsl.ImageMaths(
            in_file=str(standard_brain),
            out_file=str(standard_copy_path)
        )
        fslmaths.run()

        # Step 3: Run FLIRT registration of example_func to standard
        flirt_mat_path = output_path / "example_func2standard.mat"
        flirt_out_path = output_path / "example_func2standard.nii.gz"
        flirt = fsl.FLIRT(
            in_file=str(example_func_path),
            reference=str(standard_copy_path),
            out_file=str(flirt_out_path),
            out_matrix_file=str(flirt_mat_path),
            cost="corratio",
            dof=12,
            searchr_x=[-90, 90],
            searchr_y=[-90, 90],
            searchr_z=[-90, 90],
            interp="trilinear"
        )
        flirt.run()

        # Step 4: Invert the transformation matrix
        inv_mat_path = output_path / "standard2example_func.mat"
        convert_xfm = fsl.ConvertXFM(
            in_file=str(flirt_mat_path),
            out_file=str(inv_mat_path),
            invert_xfm=True
        )
        convert_xfm.run()

        # Step 5: Apply registration to each zstat image
        for contrast in range(1, self.num_contrasts + 1):
            zstat_input = output_path / "real.feat" / f"stats/zstat{contrast}.nii.gz"
            zstat_output = output_path / f"zstat{contrast}_registered.nii.gz"

            zstat_flirt = fsl.FLIRT(
                in_file=str(zstat_input),
                reference=str(standard_brain),
                apply_xfm=True,
                in_matrix_file=str(flirt_mat_path),
                out_file=str(zstat_output)
            )
            zstat_flirt.run()

    def get_q_score(self):
        """
        Calculates a quality (Q) score by comparing registered z-stat images to 
        task-specific template maps using multiple Dice coefficients at different 
        statistical thresholds.

        Returns:
            float: Scaled Q score derived from Dice overlap and task-specific scaling.
        """
        #Step 1: Create task design file 
        self.run_feat()

        # Step 2: Register zstat images to standard space
        self.do_registration()

        q_scores = []

        # Step 2: Loop through each contrast to compute Dice overlap
        for contrast in range(1, self.num_contrasts + 1):
            
            # Get path to the template mask
            if self.task_type == "objnam":
                template_map_path = self.base_folder / "design" / self.task_type / "template_mask.nii.gz"
            else:
                template_map_path = self.base_folder / "design" / self.task_type / f"template_contrast{contrast}_mask.nii.gz"
            
            # Load registered z-stat image
            zstat_path = self.output_path / f"zstat{contrast}_registered.nii.gz"
            zstat_data = Nifti1Image.load(zstat_path).get_fdata()
            zstat_data = np.nan_to_num(zstat_data)
    
            # Apply masking to exclude ventricles and non-brain voxels
            brain_mask = Nifti1Image.load(self.base_folder / "design/standard/MNI152_T1_4mm_brain.nii.gz").get_fdata()
            ventricle_mask = Nifti1Image.load(self.base_folder / "design/standard/MNI152_T1_ventricles_4mm.nii.gz").get_fdata()
            zstat_data = np.where((brain_mask != 0) & (ventricle_mask == 0), zstat_data, 0)

            # Load template mask
            template_data = Nifti1Image.load(template_map_path).get_fdata()
            template_data = np.nan_to_num(template_data)
            # apply mask to template as well 
            template_data = np.where((brain_mask != 0) & (ventricle_mask == 0), template_data, 0)
            # Step 3: Compute Dice coefficients at multiple thresholds
            thresholds = [99.5, 99, 97, 95, 90]
            dice_coeffs = []

            for thresh in thresholds:
                threshold_val = np.percentile(zstat_data, thresh)
                logging.info(f"Contrast {contrast} - {thresh}th percentile threshold: {threshold_val}")

                binarized_zstat = (zstat_data > threshold_val).astype(int)
                
                binarized_template = (template_data > 0).astype(int)
                dice = get_dice_coefficient(binarized_zstat, binarized_template)
                logging.info(f"Contrast {contrast} - Dice coefficient at {thresh}%: {dice:.4f}")
                dice_coeffs.append(dice)

            # Step 4: Select best Dice coefficient and compute capped Q contribution
            best_dice = max(dice_coeffs)
            logging.info(f"Contrast {contrast} - Best Dice coefficient: {best_dice:.4f}")
            
            capped_score = min((best_dice * 100 / 0.8), 100)  # normalize to a scale capped at 100
            q_scores.append(capped_score)
            #logging.info(f"Best q_score for contrast {contrast} of task type {self.task_type} is {capped_score}")

        # Step 5: Average unscaled Q scores and apply task-specific scaling
        unscaled_q_score = np.mean(q_scores)
        scaling_params_path = self.base_folder / "design/scales" / f"{self.task_type}_scaling_params.json"
        scaling_params = load_scaling_params(path_to_params=scaling_params_path)
        q_score_value = get_scaled_score(unscaled_q_score, scaling_params)
   
        logging.info(f"Overall q_score is {q_score_value}")
        return q_score_value

def get_series_description(nifti_path: Path):
    """
    nifti_path: Path to nifti file
    
    Returns: task_name: str
    """

    nifti_file_name = nifti_path.name
    series_description = nifti_file_name.split("_")
    task_name = series_description[1].lower()
    # Make sure that task name is either objnam or motor
    if task_name not in ["objnam", "motor"]:
        logging.error("Task name not objnam or motor")
        return "error"

    return task_name

def load_scaling_params(path_to_params: Path):
    """
    path_to_params: Path to json file that contains scaling parameters
    
    Returns: params dict: A dictionary with keys 'data_min', 'data_max', 'scaled_mean', and 'scaled_std', 
        e.g., {"data_min": 2.8, "data_max": 11.2, "scaled_mean": 56.39, "scaled_std": 39.03}.
    """

    with open(path_to_params, 'r') as f:
        params = json.load(f)
    return params

def get_scaled_score(unscaled_score, params):
    """
    unscaled_score: QScore integer value without any scale applied to it
    params: dictionary containing relevant parameters for scale for that task

    Returns: scaled_score: integer representing QScore, with range of 1-100
    """
    scaled_score = ((unscaled_score - params['data_min']) / (params['data_max'] - params['data_min'])) * 100
    scaled_score = int(round(np.clip(scaled_score, 1, 100),0))
    return scaled_score

def get_dice_coefficient(thresholded_image, template_map_data):
    """
    Calculate the dice coefficient between the thresholded image and the template map

    Args:
        thresholded_image: The binarized thresholded image in numpy array format
        template_map_data: The binarized template map in numpy array format
    """
        
    # Get overlap between thresholded image and template map
    overlap = np.sum(np.logical_and(thresholded_image > 0, template_map_data > 0))

    area_thresholded_image = np.sum(thresholded_image>0)
    area_template_map = np.sum(template_map_data>0)
    
    # Calculate dice coefficient
    dice_coefficient = (2 * overlap) / (area_thresholded_image + area_template_map)

    return round(dice_coefficient,2)

def resample(input_path, output_path, iter_path, transform_path):
    original_image = Nifti1Image.load(input_path)

    # Extract the first frame (index 0)
    first_frame_data = original_image.get_fdata()[..., 0]

    # Use the original affine and header
    affine = original_image.affine
    header = original_image.header.copy()
    header.set_data_shape(first_frame_data.shape)

    # Create a new 3D NIfTI image
    first_frame_img = Nifti1Image(first_frame_data, affine=affine, header=header)

    # Save the 3D image
    first_frame_img.to_filename(iter_path)

    resample = ResampleImageBySpacing()
    resample.inputs.input_image = iter_path
    resample.inputs.output_image = output_path
    resample.inputs.out_spacing = (4, 4, 4)
    resample.run()
    
    apply_tf = ApplyTransforms()
    apply_tf.inputs.fixed_image = output_path
    apply_tf.inputs.moving_image = input_path
    apply_tf.inputs.output_image = transform_path
    apply_tf.inputs.transforms = ["identity"]
    apply_tf.inputs.invert_transform_flags = [False]
    apply_tf.inputs.input_image_type = 3
    apply_tf.inputs.interpolation = "LanczosWindowedSinc"
    apply_tf.inputs.interpolation_parameters = (5,)  # Only needed for some methods

    apply_tf.run()
    # Ants apply messes with nifti header
    # reopening and saving fixes this issue 
    nifti = Nifti1Image.load(transform_path)
    nifti.to_filename(transform_path)