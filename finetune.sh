config_file='configs/pretrain/glip_A_Swin_T_O365.yaml'
custom_shot_and_epoch_and_general_copy='0_200_1'
model_checkpoint='MODEL/glip_a_tiny_o365.pth'
python -m torch.distributed.launch --nproc_per_node=6 tools/finetune.py \
      --config-file $config_file --skip-test --ft-tasks /data1/yenanfei/git/GLIP/configs/custom_dataset.yaml \
      --custom_shot_and_epoch_and_general_copy $custom_shot_and_epoch_and_general_copy \
      --evaluate_only_best_on_test --push_both_val_and_test \
      MODEL.WEIGHT $model_checkpoint \
      SOLVER.USE_AMP True TEST.DURING_TRAINING True TEST.IMS_PER_BATCH 6 SOLVER.IMS_PER_BATCH 6 \
      SOLVER.WEIGHT_DECAY 0.05 TEST.EVAL_TASK detection DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
      MODEL.BACKBONE.FREEZE_CONV_BODY_AT -1 MODEL.DYHEAD.USE_CHECKPOINT True SOLVER.FIND_UNUSED_PARAMETERS False \
      SOLVER.TEST_WITH_INFERENCE True SOLVER.USE_AUTOSTEP True DATASETS.USE_OVERRIDE_CATEGORY True SOLVER.SEED 10 \
      DATASETS.SHUFFLE_SEED 3 DATASETS.USE_CAPTION_PROMPT True DATASETS.DISABLE_SHUFFLE True \
      SOLVER.STEP_PATIENCE 2 SOLVER.CHECKPOINT_PER_EPOCH 1.0 SOLVER.AUTO_TERMINATE_PATIENCE 4 SOLVER.BASE_LR 0.0001 \
      SOLVER.MODEL_EMA 0.0 SOLVER.TUNING_HIGHLEVEL_OVERRIDE full DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE False