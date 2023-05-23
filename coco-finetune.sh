python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py \
       --config-file configs/pretrain/glip_A_Swin_T_O365.yaml \
       --skip-test \
       MODEL.WEIGHT MODEL/glip_a_tiny_o365.pth \
       DATASETS.TRAIN '("coco_grounding_train", )' \
       MODEL.BACKBONE.FREEZE_CONV_BODY_AT -1 SOLVER.IMS_PER_BATCH 4 SOLVER.USE_AMP True SOLVER.MAX_EPOCH 24 TEST.DURING_TRAINING False TEST.IMS_PER_BATCH 4 SOLVER.FIND_UNUSED_PARAMETERS False SOLVER.BASE_LR 0.00001 SOLVER.LANG_LR 0.00001 SOLVER.STEPS \(0.67,0.89\) DATASETS.DISABLE_SHUFFLE True MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection