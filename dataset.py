from torch.utils.data import Dataset, DataLoader
import os
import cv2


class dataset(Dataset):
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.readFiles = []
        self.Encoder = []
        my_dir = os.listdir(dataPath)
        
        for d in my_dir:
            # different augmentation on same image, don't read again
            if d[:-4].split('_')[0] in self.readFiles:
                continue
            
            # error file format
            if d[-4:]!='.png':
                continue

            self.readFiles.append(d[:-4].split('_')[0])

            # loading images
            img1 = cv2.imread(dataPath + '/' +  d[:-4].split('_')[0] + '_aug1.png')
            img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
            img1 = cv2.resize(img1,(150,140),interpolation=cv2.INTER_CUBIC)
            img1 = img1.reshape((3,img1.shape[0],img1.shape[1]))
            img2 = cv2.imread(dataPath + '/' + d[:-4].split('_')[0] + '_aug2.png')
            img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
            img2 = cv2.resize(img2,(150,140),interpolation=cv2.INTER_CUBIC)
            img2 = img2.reshape((3,img2.shape[0],img2.shape[1]))
            
            self.Encoder.append((img1, img2))
            self.Encoder.append((img2, img1))



    def __len__(self):
        return len(self.Encoder)

    def __getitem__(self,idx):
        return self.Encoder[idx][0], self.Encoder[idx][1] # positive pairs
    
