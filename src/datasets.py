from PIL import Image
import glob

class CatDataset():

    def __init__(self, path, transform=None, all_in_memory=False):

        self.image_paths = glob.glob(path + 'train/cat*')
        self.transform = transform
        self.all_in_memory = all_in_memory
        if self.all_in_memory:
            self.images = [self.load_image(image_path) for image_path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def load_image(self,path):
        
        image = Image.open(path)
        return image

    def __getitem__(self,index):
        
        if self.all_in_memory:
            image = self.images[index]
        else:
            image_path = self.image_paths[index]
            image = self.load_image(image_path)

        if self.transform is not None:
            image = self.transform(image)
        return image