# MogaCC-Net
## Testing
Testing on COCO val2017 dataset using model zoo's models
python tools/test.py \
    --cfg experiments/coco/hrnet/simdr/nmt_w48_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE _PATH_TO_CHECKPOINT_ \
    TEST.USE_GT_BBOX False
python tools/test.py \
    --cfg experiments/coco/hrnet/sa_simdr/w48_256x192_adam_lr1e-3_split2_sigma4.yaml \
    TEST.MODEL_FILE _PATH_TO_CHECKPOINT_ \
    TEST.USE_GT_BBOX False
Testing on MPII dataset using model zoo's models
python tools/test.py \
    --cfg experiments/mpii/hrnet/simdr/norm_w32_256x256_adam_lr1e-3_ls2e1.yaml \
    TEST.MODEL_FILE _PATH_TO_CHECKPOINT_ TEST.PCKH_THRE 0.5
