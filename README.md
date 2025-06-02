# 中文抽取式問答系統 (Chinese Extractive QA)
本項目實現了一個兩階段的中文抽取式問答系統：

段落選擇：從候選段落中選出最相關的段落。

答案抽取：從選定段落中提取答案。

## 項目文件概覽
原始數據: context.json, train.json, valid.json, test.json

預處理後數據: mc_train.json, mc_valid.json (用於段落選擇), qa_train.json, qa_valid.json (用於答案抽取)

提交範例: sample_submission.csv

run_swag_no_trainer.py: 用於訓練第一階段的段落選擇模型。

run_qa_no_trainer.py: 用於訓練第二階段的答案抽取模型。

inference_pipeline.py: 用於執行完整的兩階段推理流程並生成 submission.csv。

utils_qa.py: 問答任務常用的輔助函數 (由 Hugging Face 提供)。

dataCorrect.py: 您的數據預處理腳本，用於生成多選任務數據 (例如 mc_*.json)。

QAdata.py: 您的數據預處理腳本，用於生成問答任務數據 (例如 qa_*.json)。

requirements.txt: 項目所需的 Python 庫。

README.md: 本說明文件。

## 環境設置
克隆倉庫:

git clone [https://github.com/YuTsyh/Chinese-Extractive-Question-Answering-QA-.git](https://github.com/YuTsyh/Chinese-Extractive-Question-Answering-QA-.git)

cd Chinese-Extractive-Question-Answering-QA-

創建並激活 Python 虛擬環境 (推薦):

python -m venv .venv

安裝依賴:

pip install -r requirements.txt

## 數據準備
為段落選擇模型 (生成 mc_*.json):

python dataCorrect.py 

為答案抽取模型 (生成 qa_*.json):

python QAdata.py

## 模型訓練
1. 訓練段落選擇模型

python run_swag_no_trainer.py ^
     --model_name_or_path bert-base-chinese ^
     --train_file ./data/mc_train.json ^
     --validation_file ./data/mc_valid.json ^
     --output_dir ./paragraph_selector_output_local ^
     --max_seq_length 512 ^
     --per_device_train_batch_size 1 ^
     --gradient_accumulation_steps 2 ^
     --learning_rate 3e-5 ^
     --num_train_epochs 1 ^
     --pad_to_max_length 

2. 訓練答案抽取模型

python run_qa_no_trainer.py ^
    --model_name_or_path bert-base-chinese ^
    --train_file ./data/qa_train.json ^
    --validation_file ./data/qa_valid.json ^
    --output_dir ./qa_model_output ^
    --max_seq_length 512 ^
    --doc_stride 128 ^
    --per_device_train_batch_size 1 ^
    --gradient_accumulation_steps 2 ^
    --learning_rate 3e-5 ^
    --num_train_epochs 1 ^
    --pad_to_max_length ^
    --preprocessing_num_workers 2

## 執行推理
使用 inference_pipeline.py 腳本，利用已訓練的兩個模型生成對 test.json 的預測。

python inference_pipeline.py \
    --paragraph_selector_model_path ./paragraph_selector_output_local \
    --qa_model_path ./qa_model_output \
    --tokenizer_name_or_path bert-base-chinese \
    --context_file ./data/context.json \
    --test_file ./data/test.json \
    --output_csv submission.csv \
    --max_seq_length_mc 512 \
    --max_seq_length_qa 384 \
    --doc_stride_qa 128 \
    --qa_batch_size 8 \
    --n_best_size 20 \
    --max_answer_length 100

預測結果將保存在 submission.csv 文件中。

## 已訓練模型 (Hugging Face Hub)
段落選擇模型: [段落選擇模型](https://huggingface.co/TheWeeeed/chinese-paragraph-selector)

答案抽取模型: [答案抽取模型](https://huggingface.co/TheWeeeed/chinese-extractive-qa)

結果測試: [https://huggingface.co/spaces/TheWeeeed/chinese-qa-demo](https://huggingface.co/spaces/TheWeeeed/chinese-qa-demo)
