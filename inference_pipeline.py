import argparse
import json
import logging
import os
import collections

import torch
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader

# --- ADD THIS LINE ---
import numpy as np
# ---------------------

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    # DataCollatorWithPadding, # You are using default_data_collator
    default_data_collator # Ensure this is imported if you use it directly
)

# 假設 utils_qa.py 在同一目錄下
from utils_qa import postprocess_qa_predictions

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def parse_cli_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="兩階段問答系統的推理腳本。")
    parser.add_argument(
        "--paragraph_selector_model_path",
        type=str,
        required=True,
        help="已訓練的段落選擇模型的路徑。",
    )
    parser.add_argument(
        "--qa_model_path",
        type=str,
        required=True,
        help="已訓練的答案抽取模型的路徑。",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="bert-base-chinese",
        help="用於兩個模型的分詞器名稱或路徑。",
    )
    parser.add_argument(
        "--context_file",
        type=str,
        default="context.json",
        help="包含所有段落文本的 JSON 文件路徑。",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="test.json",
        help="包含測試問題和候選段落ID的 JSON 文件路徑。",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="submission.csv",
        help="輸出的 CSV 文件名。",
    )
    parser.add_argument(
        "--max_seq_length_mc",
        type=int,
        default=512,
        help="多選模型的最大序列長度。",
    )
    parser.add_argument(
        "--max_seq_length_qa",
        type=int,
        default=384,
        help="問答模型的最大序列長度。",
    )
    parser.add_argument(
        "--doc_stride_qa",
        type=int,
        default=128,
        help="問答模型處理長文檔時的步長。",
    )
    parser.add_argument(
        "--qa_batch_size",
        type=int,
        default=8,
        help="答案抽取模型的推理批次大小。",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="為 QA 後處理生成的 n-best 預測數量。",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help="QA 後處理允許的最大答案長度。",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="即使CUDA可用也不使用。"
    )
    return parser.parse_args()

def load_data(context_file_path, test_file_path):
    """加載上下文和測試數據"""
    logger.info(f"從 {context_file_path} 加載上下文...")
    with open(context_file_path, "r", encoding="utf-8") as f:
        contexts = json.load(f) # 假設是段落字符串的列表
    logger.info(f"從 {test_file_path} 加載測試問題...")
    with open(test_file_path, "r", encoding="utf-8") as f:
        test_data = json.load(f) # 假設是問題對象的列表
    return contexts, test_data

def select_relevant_paragraph(question_text, candidate_paragraph_texts, model, tokenizer, device, max_seq_len):
    """
    使用多選模型選擇最相關的段落。
    """
    model.eval()
    inputs_mc = []
    for p_text in candidate_paragraph_texts:
        inputs_mc.append(
            tokenizer(
                question_text,
                p_text,
                add_special_tokens=True,
                max_length=max_seq_len,
                padding="max_length", # 多選模型通常期望固定長度輸入
                truncation=True,
                return_tensors="pt",
            )
        )

    # 將多個選項的輸入堆疊起來
    # input_ids: (num_choices, seq_len) -> (1, num_choices, seq_len)
    input_ids = torch.stack([inp["input_ids"].squeeze(0) for inp in inputs_mc]).unsqueeze(0).to(device)
    attention_mask = torch.stack([inp["attention_mask"].squeeze(0) for inp in inputs_mc]).unsqueeze(0).to(device)
    
    token_type_ids = None
    if "token_type_ids" in inputs_mc[0]:
        token_type_ids = torch.stack([inp["token_type_ids"].squeeze(0) for inp in inputs_mc]).unsqueeze(0).to(device)

    with torch.no_grad():
        if token_type_ids is not None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    predicted_index = torch.argmax(outputs.logits, dim=1).item()
    return candidate_paragraph_texts[predicted_index], predicted_index


def prepare_features_for_qa_inference(examples, tokenizer, pad_on_right, max_seq_len, doc_stride):
    examples["question"] = [q.lstrip() if isinstance(q, str) else "" for q in examples["question"]]
    questions = examples["question" if pad_on_right else "context"]
    contexts = examples["context" if pad_on_right else "question"]

    # Ensure questions and contexts are lists of strings, handle None by converting to empty string
    questions = [q if isinstance(q, str) else "" for q in questions]
    contexts = [c if isinstance(c, str) else "" for c in contexts]

    tokenized_output = tokenizer(
        questions,
        contexts,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_len,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length", # This ensures all primary outputs are lists of numbers of fixed length
    )

    # The tokenizer with padding="max_length" should already produce lists of integers
    # for input_ids, attention_mask, token_type_ids.
    # The main risk of 'None' would be if the input strings were so problematic
    # that the tokenizer failed internally in a way not producing standard padded output.
    # However, standard tokenizers are quite robust with empty strings when padding is enabled.

    # Let's directly create the structure we need for the output Dataset.
    # `tokenized_output` is a BatchEncoding (dict-like).
    # If `return_overflowing_tokens=True` and N features are generated from one example,
    # then `tokenized_output['input_ids']` is a list of N lists.

    processed_features = []
    num_generated_features = len(tokenized_output["input_ids"]) # Number of features due to overflow

    # `sample_mapping` maps each generated feature back to its original example index in the input `examples`
    sample_mapping = tokenized_output.pop("overflow_to_sample_mapping", list(range(len(examples["id"]))))


    for i in range(num_generated_features):
        feature = {}
        original_example_index = sample_mapping[i] # Index of the original example this feature came from

        # These should always be lists of integers due to padding="max_length"
        feature["input_ids"] = tokenized_output["input_ids"][i]
        if "attention_mask" in tokenized_output:
            feature["attention_mask"] = tokenized_output["attention_mask"][i]
        if "token_type_ids" in tokenized_output:
            feature["token_type_ids"] = tokenized_output["token_type_ids"][i]
        
        # These might not be strictly needed by the model's forward pass but are used by postprocessing
        feature["example_id"] = examples["id"][original_example_index]
        
        current_offset_mapping = tokenized_output["offset_mapping"][i]
        sequence_ids = tokenized_output.sequence_ids(i) # Pass the index of the feature
        context_idx_in_pair = 1 if pad_on_right else 0
        
        feature["offset_mapping"] = [
            offset if sequence_ids[k] == context_idx_in_pair else None
            for k, offset in enumerate(current_offset_mapping)
        ]
        processed_features.append(feature)

    # The .map function expects a dictionary where keys are column names
    # and values are lists of features for those columns.
    # Since we are processing one original example at a time (batched=True on a Dataset of 1 row),
    # and this one example can produce multiple features, `processed_features` is a list of dicts.
    # We need to return a dictionary of lists.
    if not processed_features: # Should not happen if tokenizer works, but as a safeguard
        # Return structure with empty lists to match expected features by .map()
        # This case indicates an issue with tokenizing the input example.
        logger.error(f"No features generated for example ID {examples['id'][0]}. Input q: {examples['question'][0]}, c: {examples['context'][0]}")
        return {
            "input_ids": [], "token_type_ids": [], "attention_mask": [],
            "offset_mapping": [], "example_id": []
        }

    # Transpose the list of feature dictionaries into a dictionary of feature lists
    # This is what the .map(batched=True) function expects as a return value
    final_batch = {}
    for key in processed_features[0].keys():
        final_batch[key] = [feature[key] for feature in processed_features]
    
    for key_to_check in ["input_ids", "attention_mask", "token_type_ids"]:
        if key_to_check in final_batch:
            for i, lst in enumerate(final_batch[key_to_check]):
                if lst is None:
                    raise ValueError(f"在 prepare_features_for_qa_inference 中，{key_to_check} 的第 {i} 個特徵列表為 None！")
                if any(x is None for x in lst):
                    raise ValueError(f"在 prepare_features_for_qa_inference 中，{key_to_check} 的第 {i} 個特徵列表內部包含 None！內容: {lst[:20]}")
                    
    return final_batch


def main():
    args = parse_cli_args()

    # 設置設備
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用的設備: {device}")

    # 加載分詞器
    logger.info(f"從 {args.tokenizer_name_or_path} 加載分詞器...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)
    pad_on_right = tokenizer.padding_side == "right"

    # 加載段落選擇模型
    logger.info(f"從 {args.paragraph_selector_model_path} 加載段落選擇模型...")
    selector_config = AutoConfig.from_pretrained(args.paragraph_selector_model_path)
    selector_model = AutoModelForMultipleChoice.from_pretrained(
        args.paragraph_selector_model_path,
        config=selector_config,
    )
    selector_model.to(device)
    selector_model.eval()

    # 加載答案抽取模型
    logger.info(f"從 {args.qa_model_path} 加載答案抽取模型...")
    qa_config = AutoConfig.from_pretrained(args.qa_model_path)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        args.qa_model_path,
        config=qa_config,
    )
    qa_model.to(device)
    qa_model.eval()

    # 加載數據
    contexts, test_data = load_data(args.context_file, args.test_file)

    all_predictions = collections.OrderedDict()
    results_for_submission = []

    logger.info(f"開始對 {len(test_data)} 個測試樣本進行預測...")
    for test_item in tqdm(test_data, desc="測試問題處理中"):
        question_id = test_item["id"]
        question_text = test_item["question"]
        candidate_paragraph_indices = test_item["paragraphs"]
        
        candidate_paragraph_texts = []
        for p_idx in candidate_paragraph_indices:
            try:
                candidate_paragraph_texts.append(contexts[p_idx])
            except IndexError:
                logger.warning(f"段落索引 {p_idx} 超出範圍 (上下文總數: {len(contexts)})，問題 ID: {question_id}。跳過此段落。")
                candidate_paragraph_texts.append("") # 添加空字符串作為占位符

        if not any(candidate_paragraph_texts): # 如果所有候選段落都為空
            logger.warning(f"問題 {question_id} 的所有候選段落均為空或無效。預測空答案。")
            results_for_submission.append({"id": question_id, "answer": ""})
            all_predictions[question_id] = ""
            continue

        # 階段一：選擇相關段落
        selected_paragraph_text, _ = select_relevant_paragraph(
            question_text,
            candidate_paragraph_texts,
            selector_model,
            tokenizer,
            device,
            args.max_seq_length_mc
        )
        if not question_text or not isinstance(question_text, str) or \
           not selected_paragraph_text or not isinstance(selected_paragraph_text, str) or \
           len(question_text.strip()) == 0 or len(selected_paragraph_text.strip()) == 0:
            logger.warning(
                f"問題 ID {question_id} 的問題或選定段落為空、無效或僅包含空格。"
                f"Q: '{question_text}', P: '{selected_paragraph_text}'. 預測空答案。"
            )
            results_for_submission.append({"id": question_id, "answer": ""})
            all_predictions[question_id] = ""
            continue # 跳到下一個測試樣本
        # 準備QA模型的輸入
        # 創建一個臨時的 "example" 結構給 prepare_features_for_qa_inference
        qa_example_for_processing = {
            "id": [question_id], # 需要是列表
            "question": [question_text],
            "context": [selected_paragraph_text]
        }
        
        # 使用 Dataset.from_dict 創建一個 Dataset 對象
        # `prepare_features_for_qa_inference` 是為 datasets.map 設計的
        # 我們需要將單個樣本包裝成 Dataset
        temp_dataset = Dataset.from_dict(qa_example_for_processing)

        # 為選定的段落生成特徵
        qa_features = temp_dataset.map(
            lambda examples: prepare_features_for_qa_inference(
                examples, tokenizer, pad_on_right, args.max_seq_length_qa, args.doc_stride_qa
            ),
            batched=True,
            remove_columns=temp_dataset.column_names # 移除原始列
        )
        logger.info(f"QA features for question {question_id}: {qa_features}")
        if len(qa_features) == 0: # 檢查 qa_features 是否為空
            logger.warning(f"No QA features generated for question {question_id}. Skipping.")
            results_for_submission.append({"id": question_id, "answer": ""})
            all_predictions[question_id] = ""
            continue

        # 為了 DataLoader，只選擇模型需要的列
        model_input_columns = ["input_ids", "attention_mask"]
        if "token_type_ids" in qa_features.features: # Bert 等模型需要 token_type_ids
            model_input_columns.append("token_type_ids")

        # 創建一個只包含模型輸入列的 Dataset view (或者重新創建)
        # to_dict() 然後 from_dict() 是一種確保結構的方式
        try:
            features_for_dataloader_dict = {col: qa_features[col] for col in model_input_columns}
            features_for_dataloader = Dataset.from_dict(features_for_dataloader_dict)
        except KeyError as e:
            logger.error(f"在 qa_features 中準備模型輸入時缺少關鍵列: {e}。Features: {qa_features.features}")
            results_for_submission.append({"id": question_id, "answer": ""})
            all_predictions[question_id] = ""
            continue

        from transformers import default_data_collator # 確保已導入
        data_collator = default_data_collator
        logger.info("使用 default_data_collator。")

        qa_dataloader = DataLoader(
            features_for_dataloader, # <--- 使用只包含模型輸入列的數據集
            collate_fn=data_collator,
            batch_size=args.qa_batch_size,
            shuffle=False
        )

        all_start_logits_item = []
        all_end_logits_item = []

        for batch in qa_dataloader: # <--- 錯誤發生點
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = qa_model(**batch)
            all_start_logits_item.append(outputs.start_logits.cpu().numpy())
            all_end_logits_item.append(outputs.end_logits.cpu().numpy())
        
        if not all_start_logits_item: # 如果沒有生成任何 logits (例如，如果 qa_features 為空)
            logger.warning(f"問題 {question_id} 沒有生成 QA logits。預測空答案。")
            results_for_submission.append({"id": question_id, "answer": ""})
            all_predictions[question_id] = ""
            continue

        start_logits = np.concatenate(all_start_logits_item, axis=0)
        end_logits = np.concatenate(all_end_logits_item, axis=0)
        
        def add_empty_answers(example):
            example["answers"] = {"text": [], "answer_start": []}
            return example
        temp_dataset_with_answers = temp_dataset.map(add_empty_answers)


        predictions_item = postprocess_qa_predictions(
            examples=temp_dataset_with_answers, # 包含原始 context, question, id, answers
            features=qa_features,        # 包含 offset_mapping, example_id
            predictions=(start_logits, end_logits),
            version_2_with_negative=False, # 假設您的任務不是 SQuAD 2.0
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=0.0, # 如果不是 SQuAD 2.0，這個通常設為0
            output_dir=None, # 不需要保存中間文件
            prefix="predict",
        )
        
        # predictions_item 是一個字典 {example_id: predicted_text}
        predicted_answer_text = predictions_item.get(question_id, "") # 如果找不到ID，默認為空字符串
        all_predictions[question_id] = predicted_answer_text
        results_for_submission.append({"id": question_id, "answer": predicted_answer_text})

    # 保存結果到 CSV
    submission_df = pd.DataFrame(results_for_submission)
    submission_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig") # utf-8-sig 確保中文正確顯示
    logger.info(f"預測結果已保存到 {args.output_csv}")

    # (可選) 打印一些預測結果
    logger.info("部分預測結果示例:")
    for i, res in enumerate(results_for_submission):
        if i < 5: # 只打印前5個
            logger.info(f"ID: {res['id']}, Answer: {res['answer']}")
        else:
            break

if __name__ == "__main__":
    main()
