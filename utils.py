import torch, torchvision
import skimage.io as io
import os, numpy

class utils:
    def Data_Loader(root:str, data_flag:str):
        download=False if os.path.exists(root) else True
        train_loader = torchvision.datasets.MNIST(root=root, train=True, download=download)
        test_loader = torchvision.datasets.MNIST(root=root, train=False, download=False)
        if data_flag == 'Train':
            return train_loader.data.tolist(), train_loader.targets.tolist()
        elif data_flag == 'Test':
            return test_loader.data.tolist(), test_loader.targets.tolist()
        return None, None
    def Data_to_Image(data, targets, root='./picture'):
        if not os.path.exists(root): os.makedirs(root)
        file = open(root+'/label.txt', 'w')
        for i, (img, label) in enumerate(zip(data, targets)):
            img_path=root+'/'+str(i)+'.jpg'
            io.imsave(img_path, numpy.array(img))
            file.write(img_path+'    '+str(label)+'\n')
        file.close()
