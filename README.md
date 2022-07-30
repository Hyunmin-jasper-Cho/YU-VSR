# YU-VSR

> This is the project based on RealSR,  
> where train model using YUV, not RGB format.  

> It trains 3 components using xxxY.png, xxxU.png and xxxV.png.  
> And it makes total 3 different models which is contains  
> each components characteristic.  

## Training 
1. Prepare training dataset
- Specify datasets path(source & target to create path) in './preprocess/path.yml' and create bicubic dataset:  
`python3 ./preprocess/create_bicubic_dataset.py --dataset df2k --artifacts tdsr`  
for current drive version, to make dataset using the file YUV, use  
`python3 ./preprocess/create_bicubic_dataset_1c.py --dataset yuv_y --artifacts onec`

-  Run the below command to collect high frequency noise from Source:  
`python3 ./preprocess/collect_noise.py --dataset df2k --artifact tdsr`  
Or...  
`python3 ./preprocess/collect_noise.py --dataset yuv_y --artifact onec`

2. Train model
- Modify the configuration file options/df2k/train_bicubic_noise.yml  

- Run command:  
`CUDA_VISIBLE_DEVICES=0 python3 train.py -opt options/yuv_y/train_bicubic_noise.yml`  

- Checkpoints dir is in './experiments'  
