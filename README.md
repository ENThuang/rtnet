**Accurate Pretreatment Identification of Extra-nodal Extension in Laryngeal and Hypopharyngeal Cancers using Deep Learning: A Multicenter Retrospective Study**

This repository contains code and inference scripts.

---

## Repository Layout

```
rtnet/
├─ rtNet-HN/
│  ├─ tools/
│  │  ├─ test.py          # Inference entrypoint
│  │  └─ test.sh          # Example inference shell script
│  └─ ...
├─ requirements/          # Environment spec(s)
└─ README.md
```

---

## Environment

> Linux recommended. Single GPU is sufficient for inference.

```bash
# Create & activate an environment (example with Conda)
conda create -n rtnet python=3.10 -y
conda activate rtnet

# Install dependencies
pip install -r requirements/requirements.txt
```

> Ensure your PyTorch/CUDA versions match your system before installing the remaining dependencies.

---

## Pretrained Weights

- Download (multiple **Run × Fold** checkpoints included):  
  https://drive.google.com/file/d/1ppJgRtTe2NwhAcfqimV4m0CwaPeYI3xD/view?usp=drive_link

- Place them to match the expected layout in scripts:
  ```
  /xxx/
    ├─ RUN0/
    │   └─ fold0/
    │      └─ last_epoch_ckpt.pth
    ├─ RUN1/
    │   └─ fold0/
    │      └─ last_epoch_ckpt.pth
    └─ ...
  ```

---

## Inference

### Option A: Use the provided script (single Run/Fold example)

`rtNet-HN/tools/test.sh` (as provided):

```bash
for RUN in 0
do
    for FOLD in 0
    do
        EXP_FOLDER="RUN${RUN}"
        EXP_DIR=/xxx/final_results/RUN${RUN}/fold${FOLD}
        SETTING="mobilenetv3_2d.mobilenetv3_large_25d_fudan_hn_ln_bce_loss_dual_maxpool_ENE"

        CUDA_VISIBLE_DEVICES=0 python tools/test.py         -n $SETTING -d 1 -b 4         -c $EXP_DIR/last_epoch_ckpt.pth         current_fold $FOLD         backbone mobilenet_v3_large         output_dir $EXP_DIR/last_epoch_submit_check drop_top_and_bottom false         num_25d_group 3 test_num_25d_group 3 group_25d_overlap 0
    done
done
```

Run it:

```bash
cd rtNet-HN
bash tools/test.sh
```

Notes:
- Update `EXP_DIR` root (`/xxx`) to your actual path.
- Adjust `-b` (batch size) to fit GPU memory.
- `CUDA_VISIBLE_DEVICES=0` selects GPU 0; change as needed.

### Option B: Run all **5 Runs × 5 Folds**

```bash
cd rtNet-HN

ROOT_EXP_DIR=/xxx/final_results
SETTING="mobilenetv3_2d.mobilenetv3_large_25d_fudan_hn_ln_bce_loss_dual_maxpool_ENE"

for RUN in 0 1 2 3 4
do
  for FOLD in 0 1 2 3 4
  do
    EXP_DIR=${ROOT_EXP_DIR}/RUN${RUN}/fold${FOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/test.py       -n ${SETTING} -d 1 -b 4       -c ${EXP_DIR}/last_epoch_ckpt.pth       current_fold ${FOLD}       backbone mobilenet_v3_large       output_dir ${EXP_DIR}/last_epoch_submit_check drop_top_and_bottom false       num_25d_group 3 test_num_25d_group 3 group_25d_overlap 0
  done
done
```

---

## Important Arguments

- `-n/--name $SETTING`: experiment/config name.
- `-d 1`: number of devices (1 for inference).
- `-b 4`: batch size.
- `-c $CKPT`: checkpoint path (`.pth`).
- `current_fold $FOLD`: current fold index.
- `backbone mobilenet_v3_large`: backbone network.
- `output_dir $DIR`: output directory.
- `num_25d_group`, `test_num_25d_group`, `group_25d_overlap`: 2.5D grouping strategy settings.

---



---

## License

See `LICENSE` (e.g., MIT/Apache-2.0).

---

## Contact

- Please open a GitHub Issue for questions and suggestions.
