import json
import os

import torch
from pdf2image import convert_from_path
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import re

import pandas as pd
from io import StringIO


def main():
    input_path = "data/seijishikin/2023/0035200032.pdf"
    out_image_dir = "Analysis/seijishikin/2023/images"

    promt_hyoshi = 'この文書をOCRして、以下の項目を抜き出して以下に指定するJSON形式で出力してください。\n1. 政治団体の名称\n2. 年次\n3. 団体コード\n4. 前年繰越額\n JSONフォーマット:最終応答は、"{"で始まり"}"で終わる。または"["で始まり"]"で終わるJSONのみを出力し、JSON以外の文字は一切応答に含めないでください。'
    promt_table = 'この文書をOCRして、以下に指定するJSON形式で出力してください。\n JSONフォーマット:1. 最終応答は、"{"で始まり"}"で終わる。または"["で始まり"]"で終わるJSONのみを出力し、JSON以外の文字は一切応答に含めないでください。\n2. 空白のセルに対応する値には"NA"と出力してください'
    promt_table = 'この文書をOCRして、以下に指定するJSON形式で出力してください。この際2列目の値が空白である行は出力しなくて良いです。\n JSONフォーマット: 最終応答は、"{"で始まり"}"で終わる。または"["で始まり"]"で終わるJSONのみを出力し、JSON以外の文字は一切応答に含めないでください。'
    # promt_table = 'OCR this document and output it in the JSON format specified below. \n JSON format:1. The final response should begin with “{” and end with “}”. or output only JSON starting with “[” and ending with “]”, and do not include any characters other than JSON in the response. \n2. Do not traslate Japanese to English.\n3. Do not output rows whose second column is empty. \n 4. key of the response should be like {"name": "name of table", "data": [{"column1": "value1", "column2": "value2"}, ...]}'
    # promt_table = 'OCR this document and output it in the JSON format specified below. \n JSON format:1. The final response should begin with “{” and end with “}”. or output only JSON starting with “[” and ending with “]”, and do not include any characters other than JSON in the response. \n2. Do not traslate Japanese to English.\n3. Do not output rows whose second column is empty. \n 4. key of the response should be like {"name": "name of table", "data": [{"column1": ["value1_1", "value1_2", ...], "column2": ["value2_1", "value2_2", ...], ...]}'
    promt_table = 'OCR this document and output it in the JSON format specified below. \n JSON format:1. The final response should begin with “{” and end with “}”. or output only JSON starting with “[” and ending with “]”, and do not include any characters other than JSON in the response. \n2. Do not traslate Japanese to English.\n3. Do not output rows whose second column is empty. \n 4. key of the response should be like {"column1": ["value1_1", "value1_2", ...], "column2": ["value2_1", "value2_2", ...], ...]}'
    # promt_table = 'OCR this document and output it in the JSON format specified below. Note that you should not output rows whose second column is empty.\n JSON format:1. The final response should begin with “{” and end with “}”. or output only JSON starting with “[” and ending with “]”, and do not include any characters other than JSON in the response. \n2. Do not traslate Japanese to English.\n'
    promt_table = 'OCR this document and output it in the JSON format specified below. Note that you should not output rows whose second column is empty.\n JSON format:1. The final response should begin with “{” and end with “}”. or output only JSON starting with “[” and ending with “]”, and do not include any characters other than JSON in the response. \n2. Do not translate Japanese to English.\n3. Do not output rows whose second column is empty'
    prompt_table2 = 'この表の名称をJSON形式で出力してください。この際、表の名称以外の文字は一切応答に含めないでください。また、最終応答は、"{"で始まり"}"で終わるようにしてください。'
    prompt_table2 = 'Please output the name of this table in Japanese in JSON format. Do not include any characters other than the name of the table in the response. The final response should begin with “{” and end with “}”. Moreover, key of the response should be “name”.'

    prefix = os.path.basename(input_path).split(".pdf")[0]
    result_list = []

    model, processor = build_model_and_processor()

    images = convert_from_path(input_path)

    for i, image in enumerate(images):
        file_path = f"{out_image_dir}/{prefix}_page{i}.jpg"
        file_path_json = f"{out_image_dir}/{prefix}_page{i}.json"
        image.save(file_path, "JPEG")
        # i==0のときのみpromptはprompt_hyoshiになる。それ以外がprompt_tableを利用する
        if i == 0:
            prompt = promt_hyoshi
            max_new_tokens = 256
        else:
            prompt = promt_table
            max_new_tokens = 2048

        result = process_documents(model, processor, file_path, prompt, max_new_tokens=max_new_tokens)

        text_use = result[0].replace("\n", "").replace("```", "").replace("json", "")

        match = re.search(r'\{.*?\}$', result[0], re.DOTALL)

        if match:
            extracted_text = text_use
            json_extracted_text = json.loads(extracted_text)

            result_table_name = process_documents(model, processor, file_path, prompt_table2, max_new_tokens=max_new_tokens)
            match2 = re.search(r'\{.*?\}', result_table_name[0], re.DOTALL)
            extracted_text = match2.group(0).replace("\n", "")
            try:
                json_extracted_tname = json.loads(extracted_text)
            except json.JSONDecodeError:
                continue
            json_extracted_text["table_name"] = json_extracted_tname["name"]

            result_list.append(json_extracted_text)

            with open(file_path_json, "w") as f:
                json.dump(json_extracted_text, f)


def main2():
    input_path = "data/seijishikin/2023/0035200032.pdf"
    out_image_dir = "Analysis/seijishikin/2023/images"

    promt_table = 'OCR this document and output it in the TSV (tab-separated-value) format specified below. Note that you should not output rows whose second column is empty. Moreover, you should not putput duplicated row. \n TSV format:2. Do not translate Japanese to English.\n3. Do not output rows whose second column is empty'
    prompt_table2 = 'Please output the name of this table in Japanese in JSON format. Do not include any characters other than the name of the table in the response. The final response should begin with “{” and end with “}”. Moreover, key of the response should be “name”.'

    prefix = os.path.basename(input_path).split(".pdf")[0]
    result_list = []

    model, processor = build_model_and_processor()

    images = convert_from_path(input_path)

    i = 9
    i = 4
    file_path = f"{out_image_dir}/{prefix}_page{i}.jpg"
    file_path_json = f"{out_image_dir}/{prefix}_page{i}.json"
    # i==0のときのみpromptはprompt_hyoshiになる。それ以外がprompt_tableを利用する
    prompt = promt_table
    max_new_tokens = 2048

    result = process_documents(model, processor, file_path, prompt, max_new_tokens=max_new_tokens)

    # StringIOを使ってデータを読み込む
    data_io = StringIO(result[0])

    # データフレームに変換
    df = pd.read_csv(data_io, sep='|', skipinitialspace=True)

    # 不要な列を削除（最初と最後の空の列）
    df = df.loc[:, df.columns.notnull()]




def build_model_and_processor():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        temperature=10**(-5)
    )

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8")

    return model, processor


def process_documents(model, processor, image_path: str, text_prompt: str, max_new_tokens=2048):
    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text


def convert_table_to_dataframe(text: str) -> pd.DataFrame:
    match = re.search(r'```csv\n(.*?)```', text, re.DOTALL)

    if match:
        extracted_text = match.group(1)

        # StringIOを使って文字列をpandasのDataFrameに変換
        data = StringIO(extracted_text)
        df = pd.read_csv(data)

        # データフレームを表示
        return df
    else:
        print("該当する部分が見つかりませんでした。")

