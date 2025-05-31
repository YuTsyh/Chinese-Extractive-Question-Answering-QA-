import json

def preprocess_for_qa(original_data_path, context_list_path, output_path):
    with open(original_data_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f) # 您的原始 train.json 或 valid.json
    with open(context_list_path, 'r', encoding='utf-8') as f:
        contexts_list = json.load(f) # 您的 context.json (段落字符串列表)

    processed_qa_data_list = []
    for item in original_data:
        relevant_paragraph_id = item["relevant"]
        # 假設 relevant_paragraph_id 是 contexts_list 的索引
        # 如果 context.json 是 {id: text} 字典，則用 contexts_list[str(relevant_paragraph_id)]
        context_text = contexts_list[relevant_paragraph_id]

        # 確保 answer.start 是相對於 context_text 的字符索引
        # SQuAD 格式期望的 answers 結構
        formatted_answers = {
            "text": [item["answer"]["text"]],
            "answer_start": [item["answer"]["start"]]
        }

        # SQuAD 格式通常每個 context 可以有多個 qas
        # 為了簡化，這裡我們為每個原始 item 創建一個 qas 條目
        # 並且 title 可以用 id 或 question 的一部分
        processed_qa_data_list.append({
            "title": item["id"], # 或者問題的簡短版本
            "paragraphs": [{
                "context": context_text,
                "qas": [{
                    "id": item["id"],
                    "question": item["question"],
                    "answers": formatted_answers,
                    "is_impossible": False # 假設您的數據集總是有答案
                }]
            }]
        })

    # SQuAD 格式通常有一個頂層的 "data" 鍵
    final_output = {"data": processed_qa_data_list, "version": "1.1"} # 或 2.0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)

# 使用示例 (請修改路徑):
print("Preprocessing training data for QA...")
preprocess_for_qa("train.json", "context.json", "qa_train.json")
print("Preprocessing validation data for QA...")
preprocess_for_qa("valid.json", "context.json", "qa_valid.json")
print("QA data preprocessing complete.")