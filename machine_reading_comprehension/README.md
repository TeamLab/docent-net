# Machine Reading Comprehension with PLM

## Data

- [KLUE](https://github.com/KLUE-benchmark/KLUE)
- [KorQuAD_v1](https://korquad.github.io/)
- docent dataset

## 실행 방법
최초 실행 시 aws S3에서 docent 데이터를 다운받으므로 시간이 걸립니다.
데이터 다운로드 이후 klue, korquad, docent 순으로 학습을 진행합니다.

### Docker

```
docker load -i mrc_docent.tar
docker run -it --name mrc \
--gpus all \
-v $PWD:/app \
-w /app \
mrc_docent
```
명령어로 도커 이미지를 띄운 다음, 실행시켜줍니다. 
본 명령어를 실행하면 데이터 다운로드부터, 데이터 학습, 평가가 순차적으로 이루어집니다.


## Char-F1 확인

최종 prediction_file은 `./model/{dataset_name}_predictions.json`로 저장됩니다.
저장된 파일이 확인되면 아래의 명령어를 실행하여 char-f1을 확인할 수 있습니다.

```bash
python3 prediction.py --prediction_file {prediction_file}
```