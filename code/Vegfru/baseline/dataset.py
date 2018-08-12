import torch.utils.data as data
from os.path import join
from PIL import Image

class VegfruDataset(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform = None, level='sub'):
        super(VegfruDataset, self).__init__()
        print list_path

        self.level = level

        name_list = []
        sup_label_list = []
        sub_label_list = []

        with open(list_path, 'r') as f:
            for l in f.readlines():
                imagename, sub_label, sup_label  = l.strip().split(' ')
                name_list.append(imagename)
                sup_label_list.append(int(sup_label))
                sub_label_list.append(int(sub_label))

        self.image_filenames = [join(image_dir, x) for x in name_list]
        self.input_transform = input_transform

        self.sup_label_list = sup_label_list
        self.sub_label_list = sub_label_list

    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        input = Image.open(self.image_filenames[index]).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)

        sup_label = self.sup_label_list[index]
        sub_label = self.sub_label_list[index]
        
        if self.level == 'sup':
            target = sup_label
        if self.level == 'sub':
            target = sub_label

        return input, target

    def __len__(self):
        return len(self.image_filenames)