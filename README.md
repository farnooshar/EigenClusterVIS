This repository contains the source code for Improving Weakly-supervised Video Instance Segmentation Using Keypoints
Consistency


<img src="pull_figure1.png">
<img src="Dtemporal_pipeline_v2.png">


To prepare and install and train for steps 1 and 2, you can use this [link](https://github.com/SysCV/MaskFreeVIS).
All the weights are in this [link](https://drive.google.com/drive/folders/1opId5FM1MrVI3uRYJABf-uDxOlbev6ya?usp=drive_link).

## Inference

```Shell
#R50
!CUDA_VISIBLE_DEVICES=0 python test.py --num-gpus 1 --resume --dist-url tcp://0.0.0.0:12349\
	--config-file configs/Step3/youtubevis_2019/video_maskformer2_R50_bs16_8ep.yaml\
        --eval-only MODEL.WEIGHTS /home/user01/MaskFreeVIS/KeyVIS_r50_47.6.pth
```

```Shell
#R101

!CUDA_VISIBLE_DEVICES=0 python test.py --num-gpus 1 --resume --dist-url tcp://0.0.0.0:12349\
	--config-file configs/Step3/youtubevis_2019/video_maskformer2_R101_bs16_8ep.yaml\
        --eval-only MODEL.WEIGHTS /home/user01/MaskFreeVIS/KeyVIS_r101_49.9.pth
```
```Shell
#SwinL

!CUDA_VISIBLE_DEVICES=0 python test.py --num-gpus 1 --resume --dist-url tcp://0.0.0.0:12349\
	--config-file configs/Step3/youtubevis_2019/swin/video_maskformer2_swin_large_IN21k_384_bs16_8ep.yaml\
        --eval-only MODEL.WEIGHTS /home/user01/MaskFreeVIS/KeyVIS_SwinL_57.0.pth
```

## Train
```Shell
#R50
#2400< iter is good.
! cp alpha_beta_r50.txt alpha_beta.txt && CUDA_VISIBLE_DEVICES=0 python3 train_net_video_r50_s3.py --num-gpus 1 --resume --dist-url tcp://0.0.0.0:12349\
	--config-file configs/Step3/youtubevis_2019/video_maskformer2_R50_bs16_8ep.yaml
```

```Shell
#R101
#2700< iter is good.

!cp alpha_beta_r101.txt alpha_beta.txt && CUDA_VISIBLE_DEVICES=0 python3 train_net_video_r101_s3.py --num-gpus 1 --resume --dist-url tcp://0.0.0.0:12349\
	--config-file configs/Step3/youtubevis_2019/video_maskformer2_R101_bs16_8ep.yaml 
```
```Shell
#SwinL
#2500< iter is good.

!cp alpha_beta_SwinL.txt alpha_beta.txt && CUDA_VISIBLE_DEVICES=0 python3 train_net_video_SwinL_s3.py --num-gpus 1 --resume --dist-url tcp://0.0.0.0:12349\
	--config-file configs/Step3/youtubevis_2019/swin/video_maskformer2_swin_large_IN21k_384_bs16_8ep.yaml 
```
            
### Acknowledgement
This codebase is heavily borrowed from [Mask-Free Video Instance Segmentation](https://github.com/SysCV/MaskFreeVIS). Thanks for their excellent work.
