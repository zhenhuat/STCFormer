import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns


dataFile_gt_3d = "F:\pose_data\data_3d_h36m.npz"
pose_3d = np.load(dataFile_gt_3d,allow_pickle=True)

actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

keypoints = pose_3d['positions_3d'].item()['S9']['SittingDown']
# vis_3d(keypoints[93,:,:])
vis_joint= [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
pose_3d_gt = keypoints[:,vis_joint]


n,j,c = pose_3d_gt.shape

cof_avg = torch.zeros([j,j])
length = 243
for i in range(n-length):
    reindex=[0,7,8,9,10,1,2,3,4,5,6,14,15,16,11,12,13]
    motion_sequence = pose_3d_gt[i:i+length]
    motion_sequence = motion_sequence[:,reindex,:]
    # print(motion_sequence.shape)
    # exit()
    motion_sequence = motion_sequence.transpose(1,0,2).reshape(j,-1)
    cof = np.corrcoef(motion_sequence)
    cof_avg = cof_avg + cof

cof_avg = cof_avg/(n-length)


sns.set_style('whitegrid')
sns.heatmap(cof_avg,cmap="bone",vmin=0, vmax=1.0)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

