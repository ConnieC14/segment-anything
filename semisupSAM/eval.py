
from torch.utils.data import DataLoader
from semisupSAM.utils import load_test_data

def main():
    file_path = ".pth"
    # Load the testing dataset from the file
    loaded_testing_dataset = load_test_data(file_path)

    # You can now use 'loaded_testing_dataset' as a regular PyTorch dataset
    # For example, you can create a data loader and iterate over the dataset.
    data_loader = DataLoader(loaded_testing_dataset, batch_size=6, shuffle=True)
    