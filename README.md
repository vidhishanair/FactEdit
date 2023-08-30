
This repository contains code and data for EMNLP 2022 paper [Correcting Diverse Factual Errors in Abstractive Summarization via Post-Editing and Language Model Infilling](https://arxiv.org/abs/2210.12378) work with Hannaneh Hajishirzi, William Cohen and Yulia Tsvetkov.

## Data and Models

All training data and pretrained models can be found here: https://drive.google.com/drive/folders/1VeALcCBLIx0H3VQF2_pEJJ5ieQWtEpo9?usp=sharing

Check out [scripts/](scripts/) for various training, inference and evaluations scripts.

## Training Infilling Model

Edit data and output paths in [scripts/cnndm_run_infill.sh](scripts/cnndm_run_infill.sh)

```
bash scripts/cnndm_run_infill.sh
```

## Generate Training Data for correction Model

Edit data and output paths in [scripts/cnndm_predict_infill.sh](scripts/cnndm_predict_infill.sh)

```
bash scripts/cnndm_predict_infill.sh
```

## Training and Evaluating Fact Correction Model
Edit data and output paths in [scripts/cnndm_run_corr.sh](scripts/cnndm_run_corr.sh), [scripts/cnndm_predict_corr.sh](scripts/cnndm_predict_corr.sh)

```
bash scripts/cnndm_run_corr.sh
bash scripts/cnndm_predict_corr.sh
bash eval.sh
```
