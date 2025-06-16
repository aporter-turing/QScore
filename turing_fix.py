from tm_common.utilities.ants import ANTSResampleImageBySpacing, ANTSApplyTransforms
from tm_common.utilities import Path
import nibabel as nib 

def resample(input_path, output_path, iter_path, transform_path):
    original_image = nib.load(input_path)

    # Extract the first frame (index 0)
    first_frame_data = original_image.get_fdata()[..., 0]

    # Use the original affine and header
    affine = original_image.affine
    header = original_image.header.copy()
    header.set_data_shape(first_frame_data.shape)

    # Create a new 3D NIfTI image
    first_frame_img = nib.Nifti1Image(first_frame_data, affine=affine, header=header)

    # Save the 3D image
    nib.save(first_frame_img, iter_path)

    ANTSResampleImageBySpacing(
            input_image=iter_path,
            output_image=output_path, 
            out_spacing=(4,4,4),
        ).run()
    
    ANTSApplyTransforms(
        fixed_image=output_path,
        moving_image=input_path,
        output_image=transform_path,
        transforms=["identity"],
        interpolation="LanczosWindowedSinc",
        invert_transform_flags=[False],
        input_image_type = 3,
        interpolation_parameters=(5,)
    ).run()

task_runs = [
'QTASK_objnam_withbio_16bit_maton_20250417134348_43.nii',
'QTASK_objnam_nobio_16bit_maton_20250417134348_23.nii',
'QTASK_objnam_withbio_16bit_maton_20250417134348_41.nii',
'QTASK_motor_withbio_16bit_maton_20250417134348_17.nii',
'QTASK_objnam_withbio_16bit_maton_20250417134348_29.nii',
'QTASK_motor_withbio_16bit_maton_20250417134348_15.nii',
'QTASK_objnam_withbio_16bit_maton_20250417134348_27.nii',
'QTASK_motor_nobio_16bit_maton_20250417134348_13.nii',
'QTASK_objnam_nobio_16bit_maton_20250417134348_25.nii',
'QTASK_motor_nobio_16bit_maton_20250417134348_11.nii'
]
main_dir = Path('/home/sagemaker-user/user-default-efs/monk/BIO-20018-01_vc57909_20250417/')
# for t in task_runs:
#     try:
#         resample(main_dir / 'test' / t, main_dir / 'resampled' / f'{t}.gz', main_dir / 'anat' / f'{t}.gz', main_dir / 'transform' / f'{t}.gz')
#     except:
#         print(f"Failed to process {t}")
from nibabel.nifti1 import Nifti1Image
success_list = [
    'QTASK_objnam_withbio_16bit_maton_20250417134348_43.nii.gz',
    'QTASK_objnam_withbio_16bit_maton_20250417134348_29.nii.gz',
    'QTASK_objnam_nobio_16bit_maton_20250417134348_25.nii.gz',
    'QTASK_motor_withbio_16bit_maton_20250417134348_17.nii.gz',
    'QTASK_motor_nobio_16bit_maton_20250417134348_13.nii.gz',
]
proc_dir = Path('/home/sagemaker-user/user-default-efs/monk/BIO-20018-01_vc57909_20250417/transform')
save_dir = Path('/home/sagemaker-user/user-default-efs/monk/BIO-20018-01_vc57909_20250417/resave')

for s in success_list:
    nifti = Nifti1Image.load(proc_dir / s)
    nifti.to_filename(save_dir / s)
