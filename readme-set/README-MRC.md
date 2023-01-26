# Model Description
# Usage
## Use SavedModel in Python
```python
from transformers import pipeline, AutoTokenizer, AutoModel

model_name = {your_model_path}
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

p = pipeline("question-answering", model=model, tokenizer=tokenizer)

context = "경찰은 또 성매매 알선 자금을 관리한 박씨의 딸(32)과 성매매 여성 김모(33)씨 등 16명을 같은 혐의로 불구속 입건했다"

question = "불구속 입건된 사람은 몇 명인가요?"

p(context=context, question=question)
```

# Dataset
## Question Categorization 
**Docent**
- What: 0
- Where: 1
- When: 2
- Who: 3
- How: 4

## Data
All datasets is in the json file format, Such as Klue, Korquad-v1 dataset and Korean Docent dataset have been combined.

Each data should be like: 
```json
{"title": "title",
"context": "something context",
"id": "AW-00001",
"question_type": 1,
"question": "one question",
"answers": "answer",
}
```

Docent data which has multiple questions in a file contructed like this:
```json
{"id": "file id", 
"title": "title", 
"explain": "something context",
"q&a":
	[{"QnAID": "question id", 
	"Questions": "one question", 
	"Answer": "one answer", 
	"Type": 1, 
	"StartPoint": 0}, ... ]}
```

# Training
A model learns datasets step by step. After the first training,  a checkpoint have to call for continue the next training.
Put values into `config.yaml` if you want to setup hyperparameters

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
klue and korquad-v1 are 100 % used for training ( @@ files )

80% of docent dataset used for training ( 984 files )
10 % used for validation ( 123 files )
10% left for testing ( 123 files )

# Evaluate
Character-based F1 score for test set is about 0.88

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

