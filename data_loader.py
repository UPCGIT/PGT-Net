import torch.utils.data
import scipy.io as sio
import torchvision.transforms as transforms
import numpy as np

class TrainData(torch.utils.data.Dataset):
    def __init__(self, img, target, transform=None, target_transform=None):
        self.img = img.float()
        self.target = target.float()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.img[index], self.target[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.img)

class Data:
    def __init__(self, dataset):
        super(Data, self).__init__()

        if dataset == 'samson':
            self.P, self.L, self.col = 3, 156, 95
            data_path = "./data/samson_data.mat"
            bundle_path = './data/MyEndmemberBundles_samson.mat'
        elif dataset == 'jasper':
            self.P, self.L, self.col = 4, 198, 100
            data_path = './data/jasper_data.mat'
            bundle_path = './data/MyEndmemberBundles_jasper.mat'
        elif dataset == 'urban':
            self.P, self.L, self.col = 5, 162, 307
            data_path = './data/urban_data.mat'
            bundle_path = './data/MyEndmemberBundles_urban.mat' 

        data = sio.loadmat(data_path)
        print(f"原始 Mat 中 Y 的形状: {data['Y'].shape}")
        self.Y = torch.from_numpy(data['Y'].T)
        self.Y=self.Y.float()
        self.A = torch.from_numpy(data['A'].T)
        self.M = torch.from_numpy(data['M'])

        bundle_data = sio.loadmat(bundle_path)
        bundle_cells = bundle_data['MM'][0]
        self.bundle_list = [torch.from_numpy(bundle.astype(np.float32)).contiguous() for bundle in bundle_cells]
        mean_list = [b.mean(dim=1) for b in self.bundle_list]
        self.bundle_mean = torch.stack(mean_list, dim=1)
        self.bundle_sizes = [bundle.shape[1] for bundle in self.bundle_list]
        
        print("="*30)
        print(f"数据集: {dataset}")
        print(f"输入图像 Y (self.Y) - Max: {self.Y.max().item():.4f}, Min: {self.Y.min().item():.4f}")
        
        for i, b in enumerate(self.bundle_list):
            print(f"端元束 {i} (包含{b.shape[1]}条曲线) - Max: {b.max().item():.4f}, Min: {b.min().item():.4f}, Mean: {b.mean().item():.4f}")
        print("="*30)
        print(f"已计算端元束均值，形状为: {self.bundle_mean.shape}")
        

    def get(self, typ):
        if typ == "hs_img":
            return self.Y.float()
        elif typ == "abd_map":
            return self.A.float()
        elif typ == "end_mem":
            return self.M
        elif typ == "bundle_list":
            return self.bundle_list
        elif typ == "bundle_sizes":
            return self.bundle_sizes
        elif typ == "bundle_mean":
            return self.bundle_mean.float()
        
    def get_loader(self, batch_size=1):
        train_dataset = TrainData(img=self.Y, target=self.A, transform=transforms.Compose([]))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)

        return train_loader

