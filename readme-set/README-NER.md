# Model Description

# Usage
## Use Saved Model in Python
```python
from transformers import pipeline, AutoTokenizer

model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = BertForTokenClassification.from_pretrained({your_model_path})
p = pipeline("token-classification", model=model, tokenizer=tokenizer)

test_text = "경찰은 또 성매매 알선 자금을 관리한 박씨의 딸(32)과 성매매 여성 김모(33)씨 등 16명을 같은 혐의로 불구속 입건했다"

p(test_text)
```

# Dataset
## Tag
Same as KLUE NER dataset format, except “Duration”.
- **Tag (entity): Beginning Number / Inside Number**
- DT (Date): 0 / 1 
- LC (Location): 2 / 3
- OG (Organization): 4 / 5
- PS (Person): 6 / 7
- QT (Quantity): 8 / 9
- TI (Time): 10 / 11
- O (Unknown): 12
- DUR (Duration): 13 / 14

## Data
All datasets is in the json file format, Such as Klue, 한국해양대학교 NER dataset and Korean Docent dataset have been combined.

Each data should be like: 
This example and format is from KLUE, but same as ours.
```json
{"ner_tags": [12, 12, 12, 2, 3, 3, 3, 3, 3, 12, 2, 3, 12, 12, 12, 12, 2, 3, 3, 3, 3, 12, 12, 12, 2, 3, 3, 3, 3, 12, 12, 12, 8, 9, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
"tokens": ['특', '히', ' ', '영', '동', '고', '속', '도', '로', ' ', '강', '릉', ' ', '방', '향', ' ', '문', '막', '휴', '게', '소', '에', '서', ' ', '만', '종', '분', '기', '점', '까', '지', ' ', '5', '㎞', ' ', '구', '간', '에', '는', ' ', '승', '용', '차', ' ', '전', '용', ' ', '임', '시', ' ', '갓', '길', '차', '로', '제', '를', ' ', '운', '영', '하', '기', '로', ' ', '했', '다', '.']}
```


# Training
Docent data has long sentence which is made of Perfomance or Exhibition explains. But due to lack of token length, We use kss sentence spliter only for Docent data. 
Argumented data which is splitted sentence is used when finetuning a model. 

## Docker
if you need a container, please build with this command.
```bash
docker build -t {image_name} -f Dockerfile .
docker run -it --gpus all -p 8888:8888 -v {host_dir}:{container_dir} {image_name}
```
When you don’t have any saved model, please fix `run.sh` to train a model before build a docker container.

## Bash
If you don’t want to build a container then please install requirements.txt first, and execute `run.sh`

### Run respectively
```python
python3 main.py --do_train --do_eval --dset_name {dataset_name}
```

`—load_checkpoint`: Training with saved checkpoint.
`—do_train`: Training Flag
`—do_eval`: Validation Flag
`—dset_name`: Dataset Name

## Training dataset
klue and kmo are 100 % used for training ( 40,179 files )

80% of docent dataset used for training ( 983 files )
10 % used for validation ( 123 files )
10% left for testing ( 123 files )

# Evaluate
entity-based F1 score for test set is about 0.91

# Lincese
본 데이터셋 및 모델은 기존 AIHub의 라이센스 정책을 따르며, 모델의 경우 Hugging Face를 기반으로 작성하여 `MIT License` 를 따른다.

## AIHub Licence
1. 저작자인 한국지능정보사회진흥원 명시할 경우 자유로운 이용 및 변경이 가능하며 2차적 저작물에는 원 저작물에 적용된 것과 동일한 라이선스를 적용하여야 한다.
2. 제공받은 AI데이터를 대하여 승인을 얻은 연구자가 아닌 제 3자에게 열람하게 하거나 제공, 양도, 대여, 판매하지 아니한다.
3. AI데이터에 기반한 제품 제작 또는 기술연구에 활용한 논문, 제품 등 결과물에 데이터의 출처가 한국지능정보사회진흥원임을 반드시 명시 하여야 한다.
4. 자료의 이용 및 그에 따른 연구로 인하여 발생하는 모든 책임은 해당 기관의 기관장에게 있다.
5. 향후 한국지능정보사회진흥원에서 활용사례 등 실태조사를 수행할 경우 성실하게 임하여야 한다.

## MIT License

Copyright ©2022 한국지능정보사회진흥원 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.