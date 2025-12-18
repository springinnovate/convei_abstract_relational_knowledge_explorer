import os

from dotenv import load_dotenv
from openai import OpenAI

import os

import json
from uuid import uuid4
from openai import OpenAI

import os
import time
import json
from openai import OpenAI

load_dotenv()


def setup_batch(prompt, abstracts):
    client = OpenAI(api_key=os.environ["openai_key"])

    batch_input_path = "batch_requests.jsonl"

    with open(batch_input_path, "w") as f:
        for idx, abstract in enumerate(abstracts):
            body = {
                "model": "gpt-4.1-mini",
                "input": f"{prompt}\n\nAbstract:\n{abstract}",
            }
            row = {
                "custom_id": f"abstract-{idx}",
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            f.write(json.dumps(row) + "\n")

    input_file = client.files.create(
        file=open(batch_input_path, "rb"), purpose="batch"
    )

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )

    print(batch.id)


def monitor_batch():
    client = OpenAI(api_key=os.environ["openai_key"])

    batch_id = "YOUR_BATCH_ID"

    while True:
        batch = client.batches.retrieve(batch_id)
        print(batch.status)
        if batch.status in ("completed", "failed", "cancelled", "expired"):
            break
        time.sleep(10)

    if batch.status == "completed":
        output_file_id = batch.output_file_id
        output_file = client.files.content(output_file_id)
        lines = output_file.text.splitlines()
        results = {}
        for line in lines:
            obj = json.loads(line)
            custom_id = obj["custom_id"]
            response = obj["response"]
            text = response["output"][0]["content"][0]["text"]
            results[custom_id] = text


if __name__ == "__main__":
    pass
