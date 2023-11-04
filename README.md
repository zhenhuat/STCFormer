# STCFormer: 3D Human Pose Estimation with Spatio-Temporal Criss-cross Attention [CVPR 2023]
This is the readme file for the code release of "3D Human Pose Estimation with Spatio-Temporal Criss-cross Attention" on PyTorch platform.

Thank you for your interest, the code and checkpoints are being updated.
> [**3D Human Pose Estimation with Spatio-Temporal Criss-cross Attention**](https://openaccess.thecvf.com/content/CVPR2023/papers/Tang_3D_Human_Pose_Estimation_With_Spatio-Temporal_Criss-Cross_Attention_CVPR_2023_paper.pdf),         
> Zhenhua Tang, Zhaofan Qiu, Yanbin Hao, Richang Hong, And Ting Yao,        
> *In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023*

## Poster:
<p align="center"><img src="poster_9M.png" width="100%" alt="" /></p>

## Demo:
![Alt Text](demo.gif)

## The released codes include:
    checkpoint/:                        the folder for model weights of STCFormer.
    dataset/:                           the folder for data loader.
    common/:                            the folder for basic functions.
    model/:                             the folder for STCFormer network.
    run_stc.py:                         the python code for STCFormer networks training.


## Dependencies
Make sure you have the following dependencies installed:
* PyTorch >= 0.4.0
* NumPy
* Matplotlib=3.1.0

## Dataset

Our model is evaluated on [Human3.6M](http://vision.imar.ro/human3.6m) and [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) datasets. 

### Human3.6M
We set up the Human3.6M dataset in the same way as [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md). 
### MPI-INF-3DHP
We set up the MPI-INF-3DHP dataset in the same way as [P-STMO](https://github.com/paTRICK-swk/P-STMO). 


## Training from scratch
### Human 3.6M
For the training stage, please run:
```bash
python run_stc.py -f 27 -b 128  --train 1 --layers 6 -s 3
```
For the testing stage, please run:
```bash
python run_stc.py -f 27 -b 128  --train 0 --layers 6 -s 1 --reload 1 --previous_dir ./checkpoint/your_best_model.pth
```


## Evaluating our models

You can download our pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1waaQ1Yj-HfbNahnCN8AWCjMCGzyhZJF7?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1axVQNHxdZFH4Eiqiy2EvYQ) (extraction code：STC1). Put them in the ./checkpoint directory.

### Human 3.6M

To evaluate our STCFormer model on the 2D keypoints obtained by CPN, please run:
```bash
python run_stc.py -f 27 -b 128  --train 0 --layers 6 -s 1 -k 'cpn_ft_h36m_dbb' --reload 1 --previous_dir ./checkpoint/model_27_STCFormer/no_refine_6_4406.pth
```
```bash
python run_stc.py -f 81 -b 128  --train 0 --layers 6 -s 1 -k 'cpn_ft_h36m_dbb' --reload 1 --previous_dir ./checkpoint/model_81_STCFormer/no_refine_6_4172.pth
```
Different models use different configurations as follows.

| Frames | P1 (mm) | P2 (mm) | 
| ------------- | ------------- | ------------- |
| 27  | 44.08  | 34.76  |
| 81  | 41.72 | 32.94  |

Since the model with 243-frames input is proprietary and stored exclusively on the company server, it is unavailable due to copyright restrictions. If you require results based on that specific model, I recommend training a similar model internally to achieve the desired outcome.

### MPI-INF-3DHP
The pre-trained models and codes for STCFormer are currently undergoing updates. In the meantime, you can run this code, which is based on an earlier version and may lack organization, to observe the results for 81 frames.

```bash
 python run_3dhp_stc.py --train 0 --frames 81  -b 128  -s 1  --reload 1 --previous_dir ./checkpoint/model_81_STMO/no_refine_8_2310.pth
```


### In the Wild Video
Accroding MHFormer, make sure to download the YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA) and put it in the './demo/lib/checkpoint' directory firstly. Then, you need to put your in-the-wild videos in the './demo/video' directory.

You can modify the 'get_pose3D' function in the 'vis.py' script according to your needs, including the checkpoint and model parameters, and then execute the following command:

```bash
 python demo/vis.py --video sample_video.mp4
```




## Citation

If you find this repo useful, please consider citing our paper:

@inproceedings{tang20233d,\
  title={3D Human Pose Estimation With Spatio-Temporal Criss-Cross Attention},\
  author={Tang, Zhenhua and Qiu, Zhaofan and Hao, Yanbin and Hong, Richang and Yao, Ting},\
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},\
  pages={4790--4799},\
  year={2023}
}

## Acknowledgement
Our code refers to the following repositories.

[VideoPose3D](https://github.com/facebookresearch/VideoPose3D) \
[StridedTransformer-Pose3D](https://github.com/Vegetebird/StridedTransformer-Pose3D) \
[P-STMO](https://github.com/paTRICK-swk/P-STMO/tree/main) \
[MHFormer](https://github.com/Vegetebird/MHFormer) \
[MixSTE](https://github.com/JinluZhang1126/MixSTE) \
[FTCM](https://github.com/zhenhuat/FTCM)

We thank the authors for releasing their codes.
