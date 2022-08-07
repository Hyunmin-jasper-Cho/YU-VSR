# YU-VSR

> This is the project based on **[RealSR, Tencent](https://github.com/Tencent/Real-SR)**, but train the model using YUV, not RGB format.  

> It trains 3 components using xxxY.png, xxxU.png and xxxV.png.  
> And it makes total 3 different models which is contains  
> each components characteristic.  

## MP4TOYUV / YUV2PNG
> If you have any **raw yuv file**, which you want to use when training,  
> you can simply use **[YUV2PNG](https://github.com/Hyunmin-jasper-Cho/YUV2PNG)** to generate .png gray file for train the model.   

> Or, if you have any **.mp4 file** which you want to use, you can also  
> simply use **[MP4TOYUV](https://github.com/Hyunmin-jasper-Cho/MP4TOYUV)** then **[YUV2PNG](https://github.com/Hyunmin-jasper-Cho/YUV2PNG)**, to generate dataset whatever you want.  

## Training 
**1. Prepare training dataset**
- Specify datasets path(source & target to create path) in './preprocess/path.yml' and create bicubic dataset:  
`python3 ./preprocess/create_bicubic_dataset.py --dataset df2k --artifacts tdsr`  
for current drive version, to make dataset using the file YUV, goto codes_YUV then use  
`python3 ./preprocess/create_bicubic_dataset.py --dataset yuv_y --artifacts onec`  

- `create_bicubic_dataset.py` > Use utils, torchvision.transforms.functional and tqdm  
utils: standard util function related to image processing. [imresize, bicubic, gaussian_noise ]

-  Run the below command to collect high frequency noise from Source:  
`python3 ./preprocess/collect_noise.py --dataset df2k --artifact tdsr`  
Or...  
`python3 ./preprocess/collect_noise.py --dataset yuv_y --artifact onec`

**2. Train model**
- Modify the configuration file options/df2k/train_bicubic_noise.yml  

- Run command:  
`CUDA_VISIBLE_DEVICES=0 python3 train.py -opt options/yuv_y/train_bicubic_noise.yml`  

- Checkpoints dir is in './experiments'  
