import os
import numpy as np
from PIL import Image
import re
def get_files(path):
    grid_files = []
    fem_files = []
    tib_files = []
    for file_name in os.listdir(path):
        # Check if file is a grid file
        if re.match(r'^grid.*\.tif$', file_name):
            grid_files.append(os.path.join(path, file_name))
        # Check if file is a fem file
        elif re.match(r'^fem_label.*\.tif$', file_name):
            fem_files.append(os.path.join(path, file_name))
        elif re.match(r'^tib_label.*\.tif$', file_name):
            tib_files.append(os.path.join(path, file_name))

    # Sort the grid and fem files by number
    grid_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    fem_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    tib_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    # Return the requested number of files
    return grid_files, fem_files, tib_files

def create_multiclass_segmentation_mask(femur_path, tibia_path):
    # Open the two binary segmentation mask images and convert them to NumPy arrays
    femur = np.array(Image.open(femur_path))
    tibia = np.array(Image.open(tibia_path))
    
    # Create a new multiclass segmentation mask as an array of zeros
    multiclass = np.zeros_like(femur)
    
    # Set the pixel values based on the input images
    multiclass[femur == 255] = 1
    multiclass[tibia == 255] = 2
    
    # Convert the multiclass segmentation mask back to an image
    multiclass_image = Image.fromarray(multiclass)
    
    return multiclass_image
def create_multiclass_segmentation_masks(femur_paths, tibia_paths, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for femur_path, tibia_path in zip(femur_paths, tibia_paths):
        # Get the image number from the file path
        image_num = os.path.splitext(os.path.basename(femur_path))[0].split('_')[-1]
        
        # Create the output file path for the multiclass segmentation mask
        output_path = os.path.join(output_dir, f"tibia_and_femur_{image_num}.tif")
        
        # Call the create_multiclass_segmentation_mask() function to create the multiclass segmentation mask
        multiclass_mask = create_multiclass_segmentation_mask(femur_path, tibia_path)
        
        # Save the multiclass segmentation mask as an image
        multiclass_mask.save(output_path)
#script will take a few minutes to run
#consider using more cpu cores
grid_files, fem_files, tib_files = get_files('TPLO_Ten_Dogs_grids')
#load files from wherever you files are stored. you can use absolute path if not in the same directory
create_multiclass_segmentation_masks(fem_files, tib_files, 'multiclass_mask_directory')
#outputs tiff files that have masks for the femur and tibia
