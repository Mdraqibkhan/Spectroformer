from os import listdir
from os.path import join
import random
from PIL import Image ,ImageOps
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file

def rgb2gray(rgb):
  r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray

def rgb_2_lab(im):
   srgb_p = ImageCms.createProfile("sRGB")
   lab_p  = ImageCms.createProfile("LAB")
   rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
   Lab = ImageCms.applyTransform(im, rgb2lab)
   return color.rgb2lab(im)


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        tar = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
  


        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        transform = transforms.Compose(transform_list)
  

        a1 = a.resize((512,512), Image.BICUBIC)
        tar = tar.resize((512, 512), Image.BICUBIC)

        a = transform(a)
        tar = transform(tar)
        
        return a, tar, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)