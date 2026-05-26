import os
import torch
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class FTaobaoTBFMTemplateDataset(Dataset):
    def __init__(
            self,
            root_path: str,
            temp_path: str,
            transforms_: list = None,
            mode: str = "train",
            img_size: int = 64,
            ) -> None:
        
        super(FTaobaoTBFMTemplateDataset, self).__init__()

        self.transform = transforms.Compose(transforms_)
        self.mode = mode
        self.img_size = img_size
        if not self.mode == 'CIR':
            self.temp_path = os.path.join(temp_path, mode)
        else:
            self.temp_path = os.path.join(temp_path, 'test')

        if mode == 'train':
            f = os.path.join(root_path, os.path.join('files', 'train.csv'))
            self.data = pd.read_csv(f)

        elif mode == 'valid':
            f = os.path.join(root_path, os.path.join('files', 'valid.csv'))
            self.data = pd.read_csv(f)
            
        elif mode == 'test' or mode == 'CIR':
            f = os.path.join(root_path, os.path.join('files', 'test.csv'))
            self.data = pd.read_csv(f)
        else:
            raise ValueError("mode should be 'train', 'valid' or 'test'")

        self.img_dir = os.path.join(root_path, 'img')
        self.all_pants = pd.read_csv(os.path.join(root_path, os.path.join('files', 'all_pants_with_captions.csv')))

    def __getitem__(
            self,
            index: int
            ) -> dict:
        
        data = self.data.loc[index]

        tshirt_image = self.transform(Image.open(os.path.join(self.img_dir, str(data['tshirt']) + '.jpg')).convert('RGB'))
        gt_pant = self.transform(Image.open(os.path.join(self.img_dir, str(data['positive_pant']) + '.jpg')).convert('RGB'))
        temp_pant = self.transform(Image.open(os.path.join(self.temp_path, f"{str(data['tshirt'])}_{str(data['positive_pant'])}_template.jpg")).convert('RGB'))

        if not self.mode == 'CIR':
            neg_imgs = torch.empty(3, 3, self.img_size, self.img_size)
            neg_captions = []

            for i in range(len(neg_imgs)):
                neg_imgs[i] = self.transform(Image.open(os.path.join(self.img_dir, str(data[f'neg_{i+1}']) + '.jpg')).convert('RGB'))
                neg_captions.append(self.all_pants[self.all_pants['positive_pant'] == data[f"neg_{i+1}"]]['bottom_description'].values[0])
            
            dic = {
                'tshirt_image': tshirt_image,
                'gt_image': gt_pant,
                'template_image': temp_pant,
                'gt_caption': self.all_pants[self.all_pants['positive_pant'] == data['positive_pant']]['bottom_description'].values[0],
                'neg_images': neg_imgs,
                'neg_captions': neg_captions,
                'tshirt_id': data['tshirt'],
                'pant_id': data['positive_pant'],
                'neg_1_id': data['neg_1'],
                'neg_2_id': data['neg_2'],
                'neg_3_id': data['neg_3'],
                }
            
        else:
            neg_imgs = torch.empty(9, 3, self.img_size, self.img_size)
            neg_captions = []

            for i in range(len(neg_imgs)):
                neg_imgs[i] = self.transform(Image.open(os.path.join(self.img_dir, str(data[f'neg_{i+1}']) + '.jpg')).convert('RGB'))
                neg_captions.append(self.all_pants[self.all_pants['positive_pant'] == data[f"neg_{i+1}"]]['bottom_description'].values[0])
            
            dic = {
                'tshirt_image': tshirt_image,
                'gt_image': gt_pant,
                'template_image': temp_pant,
                'gt_caption': self.all_pants[self.all_pants['positive_pant'] == data['positive_pant']]['bottom_description'].values[0],
                'neg_images': neg_imgs,
                'neg_captions': neg_captions,
                'tshirt_id': data['tshirt'],
                'pant_id': data['positive_pant'],
                'neg_1_id': data['neg_1'],
                'neg_2_id': data['neg_2'],
                'neg_3_id': data['neg_3'],
                'neg_4_id': data['neg_4'],
                'neg_5_id': data['neg_5'],
                'neg_6_id': data['neg_6'],
                'neg_7_id': data['neg_7'],
                'neg_8_id': data['neg_8'],
                'neg_9_id': data['neg_9'],
                }

        return dic

    def __len__(self) -> int:
        return len(self.data)


class FashionFMTemplateDataset(Dataset):
    def __init__(
            self,
            root_path: str,
            dataset_name: str,
            temp_path: str,
            transforms_: list = None,
            mode: str = "train",
            img_size: int = 64,
            ) -> None:
        
        super(FashionFMTemplateDataset, self).__init__()

        self.transform = transforms.Compose(transforms_)
        self.mode = mode
        self.img_size = img_size

        if dataset_name == 'fashionvc':
            train_file = 'train.csv'
            valid_file = 'valid.csv'
            test_file = 'test.csv'
        elif dataset_name == 'fashiontaobao-tb':
            train_file = 'train.csv'
            valid_file = 'valid.csv'
            test_file = 'test.csv'
        elif dataset_name == 'expreduced':
            train_file = 'train.csv'
            valid_file = 'valid.csv'
            test_file = 'test.csv'
        else:
            raise AssertionError("dataset_name must be ['fashionvc', 'expreduced', 'fashiontaobao-tb']")

        if not self.mode == 'CIR':
            self.temp_path = os.path.join(temp_path, mode)
        else:
            self.temp_path = os.path.join(temp_path, 'test')

        if mode == 'train':
            f = os.path.join(root_path, os.path.join('files', train_file))
            self.data = pd.read_csv(f)

        elif mode == 'valid':
            f = os.path.join(root_path, os.path.join('files', valid_file))
            self.data = pd.read_csv(f)
            
        elif mode == 'test' or mode == 'CIR':
            f = os.path.join(root_path, os.path.join('files', test_file))
            self.data = pd.read_csv(f)
        else:
            raise ValueError("mode should be 'train', 'valid' or 'test'")

        self.img_dir = os.path.join(root_path, 'img')
        self.all_pants = pd.read_csv(os.path.join(root_path, os.path.join('files', 'all_pants_with_captions.csv')))

    def __getitem__(
            self,
            index: int
            ) -> dict:
        
        data = self.data.loc[index]

        tshirt_image = self.transform(Image.open(os.path.join(self.img_dir, str(data['tshirt']) + '.jpg')).convert('RGB'))
        gt_pant = self.transform(Image.open(os.path.join(self.img_dir, str(data['positive_pant']) + '.jpg')).convert('RGB'))
        temp_pant = self.transform(Image.open(os.path.join(self.temp_path, f"{str(data['tshirt'])}_{str(data['positive_pant'])}_template.jpg")).convert('RGB'))

        if self.mode == 'train':
            dic = {
                'tshirt_image': tshirt_image,
                'gt_image': gt_pant,
                'template_image': temp_pant,
                'gt_caption': self.all_pants[self.all_pants['positive_pant'] == data['positive_pant']]['bottom_description'].values[0],
                'tshirt_id': data['tshirt'],
                'pant_id': data['positive_pant'],
            }

        elif self.mode == 'CIR':
            neg_imgs = torch.empty(9, 3, self.img_size, self.img_size)

            for i in range(len(neg_imgs)):
                neg_imgs[i] = self.transform(Image.open(os.path.join(self.img_dir, str(data[f'neg_{i+1}']) + '.jpg')).convert('RGB'))
            
            dic = {
                'tshirt_image': tshirt_image,
                'gt_image': gt_pant,
                'template_image': temp_pant,
                'gt_caption': self.all_pants[self.all_pants['positive_pant'] == data['positive_pant']]['bottom_description'].values[0],
                'neg_images': neg_imgs,
                'tshirt_id': data['tshirt'],
                'pant_id': data['positive_pant'],
                'neg_1_id': data['neg_1'],
                'neg_2_id': data['neg_2'],
                'neg_3_id': data['neg_3'],
                'neg_4_id': data['neg_4'],
                'neg_5_id': data['neg_5'],
                'neg_6_id': data['neg_6'],
                'neg_7_id': data['neg_7'],
                'neg_8_id': data['neg_8'],
                'neg_9_id': data['neg_9'],
                }
            
        else:
            neg_imgs = torch.empty(3, 3, self.img_size, self.img_size)

            for i in range(len(neg_imgs)):
                neg_imgs[i] = self.transform(Image.open(os.path.join(self.img_dir, str(data[f'neg_{i+1}']) + '.jpg')).convert('RGB'))
            
            dic = {
                'tshirt_image': tshirt_image,
                'gt_image': gt_pant,
                'template_image': temp_pant,
                'gt_caption': self.all_pants[self.all_pants['positive_pant'] == data['positive_pant']]['bottom_description'].values[0],
                'neg_images': neg_imgs,
                'tshirt_id': data['tshirt'],
                'pant_id': data['positive_pant'],
                'neg_1_id': data['neg_1'],
                'neg_2_id': data['neg_2'],
                'neg_3_id': data['neg_3'],
                }

        return dic

    def __len__(self) -> int:
        return len(self.data)


class FashionTaobaoTBDataset(Dataset):
    def __init__(
            self,
            root_path: str,
            transforms_: list = None,
            mode: str = "train",
            img_size: int = 64,
            ) -> None:
        
        super(FashionTaobaoTBDataset, self).__init__()

        self.transform = transforms.Compose(transforms_)
        self.mode = mode
        self.img_size = img_size

        if mode == 'train':
            f = os.path.join(root_path, os.path.join('files', 'train.csv'))
            self.data = pd.read_csv(f)

        elif mode == 'valid':
            f = os.path.join(root_path, os.path.join('files', 'valid.csv'))
            self.data = pd.read_csv(f)
            
        elif mode == 'test' or mode == 'CIR':
            f = os.path.join(root_path, os.path.join('files', 'test.csv'))
            self.data = pd.read_csv(f)

        else:
            raise ValueError("mode should be 'train', 'valid' or 'test'")

        self.img_dir = os.path.join(root_path, 'img')

    def __getitem__(
            self,
            index: int
            ) -> dict:
        
        data = self.data.loc[index]

        tshirt_image = self.transform(Image.open(os.path.join(self.img_dir, str(data['tshirt']) + '.jpg')).convert('RGB'))
        gt_pant = self.transform(Image.open(os.path.join(self.img_dir, str(data['positive_pant']) + '.jpg')).convert('RGB'))

        if not self.mode == 'CIR':
            neg_imgs = torch.empty(3, 3, self.img_size, self.img_size)

            for i in range(len(neg_imgs)):
                neg_imgs[i] = self.transform(Image.open(os.path.join(self.img_dir, str(data[f'neg_{i+1}']) + '.jpg')).convert('RGB'))
            
            dic = {
                'tshirt_image': tshirt_image,
                'gt_image': gt_pant,
                'neg_images': neg_imgs,
                'tshirt_id': data['tshirt'],
                'pant_id': data['positive_pant'],
                'neg_1_id': data['neg_1'],
                'neg_2_id': data['neg_2'],
                'neg_3_id': data['neg_3'],
                }
            
        else:
            neg_imgs = torch.empty(9, 3, self.img_size, self.img_size)

            for i in range(len(neg_imgs)):
                neg_imgs[i] = self.transform(Image.open(os.path.join(self.img_dir, str(data[f'neg_{i+1}']) + '.jpg')).convert('RGB'))
            
            dic = {
                'tshirt_image': tshirt_image,
                'gt_image': gt_pant,
                'neg_images': neg_imgs,
                'tshirt_id': data['tshirt'],
                'pant_id': data['positive_pant'],
                'neg_1_id': data['neg_1'],
                'neg_2_id': data['neg_2'],
                'neg_3_id': data['neg_3'],
                'neg_4_id': data['neg_4'],
                'neg_5_id': data['neg_5'],
                'neg_6_id': data['neg_6'],
                'neg_7_id': data['neg_7'],
                'neg_8_id': data['neg_8'],
                'neg_9_id': data['neg_9'],
                }

        return dic

    def __len__(self) -> int:
        return len(self.data)


class FashionDataset(Dataset):
    def __init__(
            self,
            root_path: str,
            transforms_: list = None,
            mode: str = "train",
            img_size: int = 64,
            name: str = None
        ) -> None:

        super(FashionDataset, self).__init__()

        self.transform = transforms.Compose(transforms_)
        self.mode = mode
        self.img_size = img_size

        if not name == 'expreduced':
            train_file = os.path.join('files', 'train.csv')
            valid_file = os.path.join('files', 'valid.csv')
            test_file = os.path.join('files', 'test.csv')
        else:
            train_file = os.path.join('files', 'train.csv')
            valid_file = os.path.join('files', 'valid.csv')
            test_file = os.path.join('files', 'test.csv')

        if mode == 'train':
            f = os.path.join(root_path, train_file)
            self.data = pd.read_csv(f)

        elif mode == 'valid':
            f = os.path.join(root_path, valid_file)
            self.data = pd.read_csv(f)

        elif mode == 'test' or mode == 'CIR':
            f = os.path.join(root_path, test_file)
            self.data = pd.read_csv(f)

        else:
            raise ValueError("mode should be 'train', 'valid' or 'test'")

        self.img_dir = os.path.join(root_path, 'img')

    def __getitem__(
            self,
            index: int
    ) -> dict:

        data = self.data.loc[index]

        tshirt_image = self.transform(
            Image.open(os.path.join(self.img_dir, str(data['tshirt']) + '.jpg')).convert('RGB'))
        gt_pant = self.transform(
            Image.open(os.path.join(self.img_dir, str(data['positive_pant']) + '.jpg')).convert('RGB'))

        if self.mode == 'train':
            dic = {
                'tshirt_image': tshirt_image,
                'gt_image': gt_pant,
                'tshirt_id': data['tshirt'],
                'pant_id': data['positive_pant'],
            }

        elif not self.mode == 'CIR':
            neg_imgs = torch.empty(3, 3, self.img_size, self.img_size)

            for i in range(len(neg_imgs)):
                neg_imgs[i] = self.transform(
                    Image.open(os.path.join(self.img_dir, str(data[f'neg_{i + 1}']) + '.jpg')).convert('RGB'))

            dic = {
                'tshirt_image': tshirt_image,
                'gt_image': gt_pant,
                'neg_images': neg_imgs,
                'tshirt_id': data['tshirt'],
                'pant_id': data['positive_pant'],
                'neg_1_id': data['neg_1'],
                'neg_2_id': data['neg_2'],
                'neg_3_id': data['neg_3'],
            }

        else:
            neg_imgs = torch.empty(9, 3, self.img_size, self.img_size)

            for i in range(len(neg_imgs)):
                neg_imgs[i] = self.transform(
                    Image.open(os.path.join(self.img_dir, str(data[f'neg_{i + 1}']) + '.jpg')).convert('RGB'))

            dic = {
                'tshirt_image': tshirt_image,
                'gt_image': gt_pant,
                'neg_images': neg_imgs,
                'tshirt_id': data['tshirt'],
                'pant_id': data['positive_pant'],
                'neg_1_id': data['neg_1'],
                'neg_2_id': data['neg_2'],
                'neg_3_id': data['neg_3'],
                'neg_4_id': data['neg_4'],
                'neg_5_id': data['neg_5'],
                'neg_6_id': data['neg_6'],
                'neg_7_id': data['neg_7'],
                'neg_8_id': data['neg_8'],
                'neg_9_id': data['neg_9'],
            }

        return dic

    def __len__(self) -> int:
        return len(self.data)

    