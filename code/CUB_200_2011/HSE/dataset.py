import torch.utils.data as data
from os.path import join
from PIL import Image

class CubDataset(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform = None):
        super(CubDataset, self).__init__()
        print list_path

        name_list = []
        order_label_list = []
        family_label_list = []
        genus_label_list = []
        class_label_list = []
        with open(list_path, 'r') as f:
            for l in f.readlines():
                imagename, class_label, genus_label, family_label, order_label  = l.strip().split(' ')
                name_list.append(imagename)
                order_label_list.append(int(order_label))
                family_label_list.append(int(family_label))
                genus_label_list.append(int(genus_label))
                class_label_list.append(int(class_label))

        self.image_filenames = [join(image_dir, x) for x in name_list]
        self.input_transform = input_transform

        self.order_label_list = order_label_list
        self.family_label_list = family_label_list
        self.genus_label_list = genus_label_list
        self.class_label_list = class_label_list

    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        input = Image.open(self.image_filenames[index]).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input) 

        order_label = self.order_label_list[index] - 1
        family_label = self.family_label_list[index] - 1
        genus_label = self.genus_label_list[index] - 1
        class_label = self.class_label_list[index] - 1
        
        return input, order_label, family_label, genus_label, class_label

    def __len__(self):
        return len(self.image_filenames)