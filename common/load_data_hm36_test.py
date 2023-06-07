
import torch.utils.data as data
import numpy as np

from common.utils import deterministic_random
from common.camera import world_to_camera, normalize_screen_coordinates
from common.generator_tds import ChunkedGenerator

import os.path as path

class Fusion(data.Dataset):
    def __init__(self, opt, dataset, root_path, train=True, MAE=False, tds=1):
        self.data_type = opt.dataset
        self.train = train
        self.keypoints_name = opt.keypoints
        self.root_path = root_path

        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad
        self.MAE=MAE
        tds = opt.t_downsample
        #print(tds)
        #exit()
        if self.train:
            self.keypoints = self.prepare_data(dataset, self.train_list)
            self.cameras_train, self.poses_train, self.poses_train_2d = self.fetch(dataset, self.train_list,
                                                                                   subset=self.subset)
            self.generator = ChunkedGenerator(opt.batchSize // opt.stride, self.cameras_train, self.poses_train,
                                              self.poses_train_2d, self.stride, pad=self.pad,
                                              augment=opt.data_augmentation, reverse_aug=opt.reverse_augmentation,
                                              kps_left=self.kps_left, kps_right=self.kps_right,
                                              joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all, MAE=MAE, tds=tds)
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.keypoints = self.prepare_data(dataset, self.test_list)
            self.cameras_test, self.poses_test, self.poses_test_2d = self.fetch(dataset, self.test_list,
                                                                                subset=self.subset)
            self.generator = ChunkedGenerator(opt.batchSize // opt.stride, self.cameras_test, self.poses_test,
                                              self.poses_test_2d,
                                              pad=self.pad, augment=False, kps_left=self.kps_left,
                                              kps_right=self.kps_right, joints_left=self.joints_left,
                                              joints_right=self.joints_right, MAE=MAE, tds=tds)
            self.key_index = self.generator.saved_index
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))

    def prepare_data(self, dataset, folder_list):
        HR_PATH= self.root_path+'HRN'
        camara_path=path.join(self.root_path, 'data_2d_h36m_gt' + '.npz')
        
        keypoints_train3d_path=path.join(HR_PATH, 'threeDPose_train' + '.npy')
        keypoints_test3d_path=path.join(HR_PATH, 'threeDPose_test' + '.npy') 
        hrn_train3d = np.load(keypoints_train3d_path, allow_pickle=True)
        hrn_test3d = np.load(keypoints_test3d_path, allow_pickle=True)
        
        re_order  = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
        for key in hrn_train3d.item().keys():
#             print(key)
            subject = 'S'+str(key[0])
            action  = key[2].split('.')[0]
            hrn_train_key=hrn_train3d.item()[key].reshape(-1,32,3)
            hrn_train_key=hrn_train_key[:,re_order,:]
            anim = dataset[subject][action]
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(hrn_train_key.astype(cam['orientation'].dtype)/1000, R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1] 
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d
            
        for key in hrn_test3d.item().keys():
            subject = 'S'+str(key[0])
            action  = key[2].split('.')[0]
            hrn_test_key=hrn_test3d.item()[key].reshape(-1,32,3)
            hrn_test_key=hrn_test_key[:,re_order,:]
            if subject=='S11' and action == 'Directions':
                continue
            anim = dataset[subject][action]
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(hrn_test_key.astype(cam['orientation'].dtype)/1000, R=cam['orientation'], t=cam['translation'])
                #pos_3d = hrn_test_key
                pos_3d[:, 1:] -= pos_3d[:, :1]
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d  
        


        keypoints_train=path.join(HR_PATH, 'twoDPose_HRN_train' + '.npy')
        keypoints_test=path.join(HR_PATH, 'twoDPose_HRN_test' + '.npy')
        keypoints_symmetry = [[4,5,6,11,12,13],[1,2,3,14,15,16]]

        keypoints = self.create_2d_data(keypoints_train,keypoints_test,camara_path,dataset)
        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        self.joints_left, self.joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
        

        return keypoints
    
    

    def create_2d_data(self,train_path,test_path,camera_path,dataset):
        keypoints = np.load(camera_path, allow_pickle=True)
#         print(keypoints.keys)
#         exit()
        keypoints = keypoints['positions_2d'].item()


        re_order  = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
        hrn_train = np.load(train_path, allow_pickle=True)
        hrn_test = np.load(test_path, allow_pickle=True)
        for key in hrn_train.item().keys():
            subject = 'S'+str(key[0])
#             print(key)
#             exit()
            action  = key[2].split('.')[0]
            hr_cam  = key[2].split('.')[1]
            hrn_train_key=hrn_train.item()[key].reshape(-1,32,2)
            hrn_train_key=hrn_train_key[:,re_order,:]
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                cameras_name = cam['id']
                if cameras_name==hr_cam:
                    hrn_train_key[..., :2]=normalize_screen_coordinates(hrn_train_key[..., :2], w=cam['res_w'], h=cam['res_h'])
                    keypoints[subject][action][cam_idx] = hrn_train_key  
        for key in hrn_test.item().keys():
            subject = 'S'+str(key[0])
            action  = key[2].split('.')[0]
            hr_cam  = key[2].split('.')[1]
            hrn_test_key=hrn_test.item()[key].reshape(-1,32,2)
            hrn_test_key=hrn_test_key[:,re_order,:]
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                cameras_name = cam['id']
#                 print(key,cam_idx,cameras_name,hr_cam)
                if cameras_name==hr_cam:
                    hrn_test_key[..., :2]=normalize_screen_coordinates(hrn_test_key[..., :2], w=cam['res_w'], h=cam['res_h'])
                    keypoints[subject][action][cam_idx] = hrn_test_key
#         print(subject,action,cam_idx)
#         exit()

        return keypoints 

    def fetch(self, dataset, subjects, subset=1, parse_3d_poses=True):
        out_poses_3d = {}
        out_poses_2d = {}
        out_camera_params = {}
#         print(dataset['S9']['Directions']['positions_3d'][0][199,:,:])
#         exit()
        for subject in subjects:
            for action in self.keypoints[subject].keys():
                if self.action_filter is not None:
                    found = False
                    for a in self.action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_2d = self.keypoints[subject][action]
                #print(action)

                for i in range(len(poses_2d)):
                    out_poses_2d[(subject, action, i)] = poses_2d[i]

                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for i, cam in enumerate(cams):
                        if 'intrinsic' in cam:
                            out_camera_params[(subject, action, i)] = cam['intrinsic']

                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)): 
                        out_poses_3d[(subject, action, i)] = poses_3d[i]

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        stride = self.downsample
        if subset < 1:
            for key in out_poses_2d.keys():
                n_frames = int(round(len(out_poses_2d[key]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[key]) - n_frames + 1, str(len(out_poses_2d[key])))
                out_poses_2d[key] = out_poses_2d[key][start:start + n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][start:start + n_frames:stride]
        elif stride > 1:
            for key in out_poses_2d.keys():
                out_poses_2d[key] = out_poses_2d[key][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d

    def __len__(self):
        return len(self.generator.pairs)
        #return 200

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]

        if self.MAE:
            cam, input_2D, action, subject, cam_ind = self.generator.get_batch(seq_name, start_3d, end_3d, flip,
                                                                                      reverse)
            if self.train == False and self.test_aug:
                _, input_2D_aug, _, _,_ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
                input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
        else:
            cam, gt_3D, input_2D, action, subject, cam_ind = self.generator.get_batch(seq_name, start_3d, end_3d, flip, reverse)
        
            if self.train == False and self.test_aug:
                _, _, input_2D_aug, _, _,_ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
                input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
            
        bb_box = np.array([0, 0, 1, 1])
        input_2D_update = input_2D

        scale = np.float(1.0)

        if self.MAE:
            return cam, input_2D_update, action, subject, scale, bb_box, cam_ind
        else:
            return cam, gt_3D, input_2D_update, action, subject, scale, bb_box, cam_ind



