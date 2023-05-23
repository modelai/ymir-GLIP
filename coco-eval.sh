python tools/test_grounding_net.py --config-file configs/pretrain/glip_A_Swin_T_O365.yaml --weight MODEL/glip_a_tiny_o365.pth \
        TEST.IMS_PER_BATCH 1 \
        MODEL.DYHEAD.SCORE_AGG "MEAN" \
        TEST.EVAL_TASK detection \
        MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
        OUTPUT_DIR output_dir