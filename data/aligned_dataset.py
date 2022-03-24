import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        self.norm_A = 0 #Normalization value
        for img in self.A_paths:
            img_arr = np.load(img)
            val = np.max(np.abs(img_arr[img_arr.files[0]]))
            self.norm_A = max(self.norm_A, val)
        #print('Path:', self.A_paths[0])
        #raise Exception(self.A_paths[0])
        ### input B (real images)
        if True: #opt.isTrain or opt.use_encoded_image: #True: #Get real images
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

            #self.norm_B = 0
            #for img in self.B_paths:
            #    img_arr = np.load(img)
            #    val = np.max(np.abs(img_arr[img_arr.files[0]]))
            #    #print('val:', val) ###
            #    self.norm_B = max(self.norm_B, val)
            #    print(img)
        #print(self.B_paths[0])
        #raise Exception(self.B_paths[0])
        #print('self.norm_A:', self.norm_A)
        #print('self.norm_B:', self.norm_B)
        


        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = np.load(A_path) #Image.open(A_path)
        A = A[A.files[0]][:, :, np.newaxis] #np.repeat(A[A.files[0]][256:512,0:256, np.newaxis], 3, axis=2) #To extract the array from the .npz
        #A = A / self.norm_A
        params = get_params(self.opt, (A.shape[1], A.shape[0]))
        if self.opt.label_nc == 0: #This branch will execute.
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A) #.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=True, norm_val = self.norm_A)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if True: #self.opt.isTrain or self.opt.use_encoded_image: #True: #Always have real images
            B_path = self.B_paths[index]   
            B = np.load(B_path) #Image.open(B_path).convert('RGB')
            B = B[B.files[0]][:, :, np.newaxis] * 20 #np.repeat(B[B.files[0]][256:512,0:256, np.newaxis] * 20, 3, axis=2) #To extract the array from the .npz
            #B = B / self.norm_B
            transform_B = get_transform(self.opt, params, normalize=True, norm_val = self.norm_A)      
            B_tensor = transform_B(B)
            #print('real images:', B_tensor)
            #print('self.norm_A:', self.norm_A)
            #print('self.norm_B:', self.norm_B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
