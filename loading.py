import os                             
from PIL import Image                 
from torchvision import transforms    
from torch.utils.data import Dataset, DataLoader  

class ImageTextDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir                                                                    
        self.image_paths = [f"{data_dir}/{f}" for f in os.listdir(data_dir) if f.endswith(".jpeg")] 
        self.text_paths = [f.replace(".jpeg", ".txt") for f in self.image_paths]                    

    def __len__(self):                                             # Returns the number of images & text files               
        return len(self.image_paths)                               
        return len(self.text_paths)                                 


    def __getitem__(self, idx):                                     # Get image and text paths

        image_path = self.image_paths[idx]                          
        text_path = self.text_paths[idx]                            

        image = Image.open(image_path)                              # Preprocess Image
        transform = transforms.Compose([transforms.Resize((580, 770)),transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  # Normalize for better training
        image = transform(image)                                    # Transforms image (Grayscale = 1 channel, RGB = 3 channels)

                                                            
        with open(text_path, 'r') as f:                             # Preprocess Text
            text = f.read().strip()

        return image, text


data_dir = "/Users/sarthakkapila/Desktop/Dataset-espanol/train"

dataset = ImageTextDataset(data_dir)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)         # Load data in batches of 1 (this is reflect in the train phase)

# Testing data loading
test_root = '/Users/sarthakkapila/Desktop/Dataset-espanol/test'

test_dataset = torchvision.datasets.ImageFolder(root=test_root, transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=75, shuffle=False)



