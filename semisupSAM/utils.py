
import torch
import os
import numpy as np
import glob

join = os.path.join
SMOOTH = 1e-6
MRI_SCAN = {
        'T1w'   : 't1',   
        'T1Gd'  : 't1ce', # or NON-ENHANCING tumor CORE - RED
        'T2'    : 't2',  # Green
        'FLAIR' : 'flair' # original 4 -> converted into 3 later, Yellow
    }

# %%Helper Functions
def get_path_files(root_dir, mri_scan):

    img_path_files = sorted(
            glob.glob(join(root_dir, f"**/*{MRI_SCAN[mri_scan]}.nii.gz"), recursive=True)
    )

    img_path_files = [
        file
        for file in img_path_files
        if os.path.isfile(file)
    ]

    return img_path_files

# %%Training
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# %%Evaluation
def convert_outputs_to_masks(outputs, threshold=0.5):
    # Move the outputs to CPU if on a GPU
    outputs = outputs.detach().cpu()

    # Normalize the outputs to [0, 1] using min-max scaling
    outputs_min = torch.min(outputs)
    outputs_max = torch.max(outputs)
    normalized_outputs = (outputs - outputs_min) / (outputs_max - outputs_min)

    # Apply thresholding to convert to binary mask
    binary_masks = (normalized_outputs > threshold).type(torch.uint8)
    
    return binary_masks

def compute_precision_recall_f1(pred_mask, true_mask):
    true_positive = (pred_mask & true_mask).sum().float()
    false_positive = (pred_mask & ~true_mask).sum().float()
    false_negative = (~pred_mask & true_mask).sum().float()

    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision, recall, f1_score

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = np.logical_and(outputs, labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    
    union = np.logical_or(outputs, labels).float().sum((1, 2))   # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded.mean()  # thresholded Or thresholded.mean() if you are interested in average across the batch

def compute_pixel_accuracy(preds, true):
    tot_accuracy = 0
    for pred_mask, true_mask in zip(preds, true):
        # Find pixels that were correctly identified
        correct_pixels = (pred_mask == true_mask).sum().item()
        
        total_pixels = pred_mask.numel()
        tot_accuracy += correct_pixels / total_pixels
    return tot_accuracy / preds.shape[0]

# %% Working with torch datasets
def save_test_data(dataset, directory, output_filename):
    try:
        # Create a new folder if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Folder '{directory}' created successfully.")
        else:
            print(f"Folder '{directory}' already exists.")

        # Write the file paths to a text file
        output_file_path = os.path.join(directory, output_filename)
        torch.save(dataset, output_file_path)
        
        print(f"Dataset listed in '{output_filename}'.")
    except Exception as e:
        print(f"Error: {str(e)}")

def load_test_data(file_path):
    try:
        # Load the dataset from the file
        dataset = torch.load(file_path)
        print(f"Testing dataset loaded from '{file_path}'.")
        return dataset
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
