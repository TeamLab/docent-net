#!/usr/bin/env bash
mkdir mrc_data
aws s3 sync s3://docent-information/mrc_data ./mrc_data --no-sign-request

python3 main.py \
        --do_train \
        --do_eval \

sed -i 's/dataset_name: klue/dataset_name: squad_kor_v1/g' config.yaml
python3 main.py \
        --do_train \
        --do_eval \
        --load_checkpoint

sed -i 's/dataset_name: squad_kor_v1/dataset_name: docent/g' config.yaml
python3 main.py \
        --do_train \
        --do_eval \
        --load_checkpoint