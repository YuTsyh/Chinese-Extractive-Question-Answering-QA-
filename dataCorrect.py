import json

def preprocess_for_multiple_choice(original_data_path, context_list_path, output_path):
    with open(original_data_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    with open(context_list_path, 'r', encoding='utf-8') as f:
        contexts_list = json.load(f)

    processed_mc_data = []
    for item in original_data:
        # Ensure 'question' key exists, especially if processing test data later
        if "question" not in item:
            print(f"Warning: 'question' key missing in item: {item.get('id', 'Unknown ID')}")
            continue

        question_text = item["question"]
        paragraph_ids = item["paragraphs"]

        # 'relevant' might not exist in test data, handle gracefully
        relevant_paragraph_id = item.get("relevant") # Use .get() for safety
        correct_label = -1

        choices_texts = []
        for i, p_id in enumerate(paragraph_ids):
            try:
                # Assuming p_id is an integer index for contexts_list
                choices_texts.append(contexts_list[p_id])
            except IndexError:
                print(f"Error: Paragraph ID {p_id} is out of bounds for context_list for question {item.get('id', 'Unknown ID')}")
                # Add placeholder or skip, depending on how you want to handle missing contexts
                choices_texts.append("") # Add an empty string as a placeholder
                continue # Or skip this choice / item
            except TypeError:
                print(f"Error: contexts_list seems to be a dictionary, but p_id ({p_id}) is not a string key or other issue. Question: {item.get('id', 'Unknown ID')}")
                choices_texts.append("")
                continue


            if relevant_paragraph_id is not None and p_id == relevant_paragraph_id:
                correct_label = i

        # If it's training/validation data, a correct label must be found
        if relevant_paragraph_id is not None and correct_label == -1:
            print(f"Warning: Relevant paragraph ID {relevant_paragraph_id} not found in paragraphs list {paragraph_ids} for question {item.get('id', 'Unknown ID')}")
            continue

        if len(choices_texts) != 4:
            print(f"Warning: Expected 4 choices, but gathered {len(choices_texts)} for question {item.get('id', 'Unknown ID')}. Padding with empty strings if necessary.")
            while len(choices_texts) < 4:
                choices_texts.append("") # Pad with empty strings if fewer than 4 choices were processed due to errors

        mc_item = {
            "id": item.get("id", f"item_{len(processed_mc_data)}"), # Ensure ID exists
            "question_context": question_text,
            "choice_0": choices_texts[0],
            "choice_1": choices_texts[1],
            "choice_2": choices_texts[2],
            "choice_3": choices_texts[3],
        }
        # Only add label if it's not test data (where relevant_paragraph_id would be None)
        if relevant_paragraph_id is not None:
            mc_item["label"] = correct_label

        processed_mc_data.append(mc_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_mc_data, f, ensure_ascii=False, indent=4)

# Corrected usage:
# Assuming your validation file is named "valid.json" and is in /content/
# Assuming your context file is named "context.json" and is in /content/

# For training data:
preprocess_for_multiple_choice("train.json", "context.json", "mc_train.json")

# For validation data (replace "valid.json" if your file has a different name):
preprocess_for_multiple_choice("valid.json", "context.json", "mc_valid.json")