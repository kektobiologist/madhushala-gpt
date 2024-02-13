
# Madhushala GPT (cloned from nanoGPT) 

## Install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3



## Training

### Pretraining

Pretraining on ~50k lines of AI4Bharat's [IndicCORPv2](https://github.com/AI4Bharat/IndicBERT/tree/main#indiccorp-v2) hindi dataset. Prepare the dataset:
```
python data/indic_hi_100k/prepare_hf.py
```
This will generate train.bin, val.bin, meta.pkl. Start pretraining:
```
python train.py config/train_hi_100k.py
```
Check results. The checkpoint is saved in `out-hi-100k/`
```
python sample.py --out_dir=out-hi-100k
```

### Finetuning

Finetune on Madhushala.txt. Prepare the dataset:
```
python data/madhushala/prepare.py
```
This will create train/val split. Start finetuning:
```
python train.py config/finetune_madhushala
```
Check results. The checkpoint is the same in `out-hi-100k/`
```
python sample.py --out_dir=out-hi-100k
```

## Demo notebook
See `demo.ipynb`. Will need file `out-hi-100k/ckpt.pt` (not part of repo). Download from [here](https://drive.google.com/file/d/1yQHsuxbwXSN5Bm0NWYtlWKeqhL19ytU-/view?usp=drive_link).

## Notes
- Pretraining with hi_100k gives better validation loss than just pretraining with madhushala.txt (best 1.86 vs best 1.51, lost wandb logs unfortunately). Also improves grammar and makes it use hindi words not part of madhushala.txt
- Increasing batch size also improves validation loss. Didn't experiment much with model sizes. Reducing finetuning LR also helps
- Using eot token to separate stanzas didn't give much improvement in terms of stanza size consistency generation results like I hoped, not sure why. Probably need to deep dive more.
- Need to do more experiments:
    - Use full size hi.txt dataset (50mil rows) for pt
    - Larger model (more layers, heads)
