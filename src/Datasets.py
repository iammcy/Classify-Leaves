import pandas as pd
import numpy as np
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms

class LeavesData(Dataset):
    def __init__(self, data_path, mode='train', valid_ratio=0.2, resize_height=224, resize_width=224) -> None:
        super(LeavesData, self).__init__()

        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = data_path
        self.mode = mode

        self.class_to_num = Class_To_Num(data_path+"train.csv")

        # 读取csv文件
        if mode == 'test':
            csv_file = data_path + 'test.csv'
        else:
            csv_file = data_path + 'train.csv'
        self.data_info = pd.read_csv(csv_file)
        self.data_len = len(self.data_info.index)
        self.train_len = int(self.data_len*(1-valid_ratio))

        if mode == 'train':
            self.image_arr = np.asarray(self.data_info.iloc[:self.train_len, 0])
            self.label_arr = np.asarray(self.data_info.iloc[:self.train_len, 1])
        elif mode == 'valid':
            self.image_arr = np.asarray(self.data_info.iloc[self.train_len:, 0])
            self.label_arr = np.asarray(self.data_info.iloc[self.train_len:, 1])
        elif mode == 'test':
            self.image_arr = np.asarray(self.data_info.iloc[:, 0])

        self.real_len = len(self.image_arr)
        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))
    
    def __getitem__(self, index: int):
        image_name = self.image_arr[index]
        image = Image.open(self.file_path+image_name)

        if image.mode != 'L':
            image = image.convert('L')
            image = np.array(image)
            image = np.expand_dims(image, axis=2)
            image = np.concatenate((image, image, image), axis=-1)
            image = Image.fromarray(image.astype('uint8')).convert('RGB')

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.resize_height, self.resize_width)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.resize_height, self.resize_width)),
                transforms.ToTensor()
            ])
        
        image = transform(image)

        if self.mode == 'test':
            return image
        
        else:
            label = self.label_arr[index]
            num = self.class_to_num[label]
            return image, num

    def __len__(self) -> int:
        return self.real_len


def Class_To_Num(file_path):
    dataframe = pd.read_csv(file_path)
    labels = sorted(list(set(dataframe['label'])))
    n_classes = len(labels)
    map = dict(zip(labels, range(n_classes)))
    return map