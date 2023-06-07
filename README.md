# STCFormer
This is the readme file for the code release of "3D Human Pose Estimation with Spatio-Temporal Criss-cross Attention" on PyTorch platform.

Thank you for your interest, the code and checkpoints are being updated.



##The released codes include:
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
python run_stc.py -f 27 -b 128  --train 1 --layers 6 
```
For the testing stage, please run:
```bash
python run_stc.py -f 27 -b 128  --train 0 --layers 6 --reload 1 --previous_dir ./checkpoint/your_best_model.pth
```

The pre-trained models and the codes for STCFormer will be released after the review process.
