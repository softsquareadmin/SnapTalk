import os
import json
import tempfile
import re
from openai import OpenAI, OpenAIError, RateLimitError
from pinecone import Pinecone

def sectionwise_chunk_json(input_file, output_file=None):
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_chunks.json"

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunked_docs = []

    def process_section(key, value, parent_keys=[]):
        section_id = "#".join([os.path.basename(input_file)] + parent_keys + [key])
        heading = " > ".join(parent_keys + [key]).replace("_", " ").title()
        if isinstance(value, dict):
            for subkey, subval in value.items():
                process_section(subkey, subval, parent_keys + [key])
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                process_section(f"{key}_{idx}", item, parent_keys)
        else:
            text = f"{heading}: {value}"
            chunked_docs.append({
                "id": section_id,
                "text": text
            })

    for k, v in data.items():
        process_section(k, v, [])

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunked_docs, f, indent=2, ensure_ascii=False)

    print(f"✅ Section-wise chunked OCR JSON saved as {output_file} ({len(chunked_docs)} chunks)")
    return len(chunked_docs)

def extract_text_from_image(image_path, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    with open(image_path, "rb") as image_file:
        import base64
        base64_img = f"data:image/png;base64,{base64.b64encode(image_file.read()).decode()}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Return JSON Document with data. Only return JSON not other text"},
                    {
                        "type": "image_url",
                        "image_url": {"url": base64_img}
                    }
                ],
            }
        ],
        max_tokens=2048
    )
    return response.choices[0].message.content

def extract_json_from_response(text):
    # Remove code block markers and extract JSON
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Fallback: try to remove any ``` and whitespace
        json_str = text.strip().strip('`')
    return json_str

def get_embedding(text, openai_api_key, model="text-embedding-3-small", max_retries=5):
    client = OpenAI(api_key=openai_api_key)
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except RateLimitError:
            import time
            print("⚠️ OpenAI rate limit hit. Retrying in 5 seconds...")
            time.sleep(5)
        except OpenAIError as e:
            print(f"❌ OpenAI error: {e}. Retrying in 5 seconds...")
            import time
            time.sleep(5)
    raise RuntimeError("Failed to get embedding after retries.")

def process_and_upload(image_path, openai_api_key, pinecone_api_key, pinecone_index_name):
    # Ensure json folder exists
    os.makedirs("json", exist_ok=True)
    # Use image filename as base
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    ocr_json_path = os.path.join("json", f"{base_filename}.json")
    chunked_json_path = os.path.join("json", f"{base_filename}_chunks.json")
    # 1. OCR
    ocr_json_str = extract_text_from_image(image_path, openai_api_key)
    # 2. Extract JSON from response
    ocr_json_str = extract_json_from_response(ocr_json_str)
    # 3. Save OCR JSON
    with open(ocr_json_path, "w", encoding="utf-8") as tmp_json:
        tmp_json.write(ocr_json_str)
    # 4. Chunk
    sectionwise_chunk_json(ocr_json_path, chunked_json_path)
    # 5. Embed & upload
    with open(chunked_json_path, "r", encoding="utf-8") as f:
        chunked_docs = json.load(f)
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    for doc in chunked_docs:
        embedding = get_embedding(doc["text"], openai_api_key)
        index.upsert([(doc["id"], embedding, {"text": doc["text"][:500]})])
    return True 