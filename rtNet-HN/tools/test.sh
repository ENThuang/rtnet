for RUN in 0
do
    for FOLD in 0
    do
        EXP_FOLDER="RUN${RUN}"
        EXP_DIR=/xxx/final_results/RUN${RUN}/fold${FOLD}
        SETTING="mobilenetv3_2d.mobilenetv3_large_25d_fudan_hn_ln_bce_loss_dual_maxpool_ENE"

        CUDA_VISIBLE_DEVICES=0 python tools/test.py \
        -n $SETTING -d 1 -b 4 \
        -c $EXP_DIR/last_epoch_ckpt.pth \
        current_fold $FOLD \
        backbone mobilenet_v3_large \
        output_dir $EXP_DIR/last_epoch_submit_check drop_top_and_bottom false \
        num_25d_group 3 test_num_25d_group 3 group_25d_overlap 0
    done
done