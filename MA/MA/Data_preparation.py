import os
from nilearn import datasets, masking, image
from nilearn.plotting import plot_epi, plot_roi, show
import nibabel as nib

# Define the data directory and subject range
data_dir = 'G:\\Data\\Haxby_2001'
subjects = range(1, 7)

# Loop through each subject
for sub in subjects:
    # Fetch the dataset for the current subject
    haxby_dataset = datasets.fetch_haxby(data_dir=data_dir, subjects=(sub,), fetch_stimuli=True, url=None, resume=True,
                                         verbose=1)

    # Print basic information about the dataset
    print(f"Subject {sub}")
    print(f"Anatomical image: {haxby_dataset.anat[0]}")
    print(f"Functional image: {haxby_dataset.func[0]}")

    # Load the functional image
    func_filename = haxby_dataset.func[0]
    func_img = nib.load(func_filename)

    # Compute the mean EPI for the functional image (mean across time dimension)
    mean_func_img = image.mean_img(func_img)

    # Compute the mask for the functional image
    mask_img = masking.compute_epi_mask(func_img)

    # Visualize the mean functional image and the mask
    plot_epi(mean_func_img, title=f"Mean Functional Image - Subject {sub}", colorbar=True, cbar_tick_format="%i")
    plot_roi(mask_img, mean_func_img, title=f"Computed Mask - Subject {sub}")
    show()

    # Save the mask image
    mask_filename = os.path.join(data_dir, f'sub{sub}_mask.nii.gz')
    mask_img.to_filename(mask_filename)
    print(f"Mask saved to {mask_filename}")

print("Processing completed.")
