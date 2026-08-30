import base64
import json
import mimetypes
import os
import urllib.request


base_url = os.environ.get("BASE_URL", "http://127.0.0.1:8010/v1")
api_key = os.environ.get("API_KEY")
image_path = os.environ["IMAGE_PATH"]
mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"

with open(image_path, "rb") as handle:
    data_url = (
        f"data:{mime_type};base64," + base64.b64encode(handle.read()).decode()
    )

payload = {
    "model": "glm-5.3-flash",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the main object in this image."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ],
    "max_tokens": 512,
    "temperature": 0,
}
headers = {"Content-Type": "application/json"}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"

request = urllib.request.Request(
    f"{base_url.rstrip('/')}/chat/completions",
    data=json.dumps(payload).encode(),
    headers=headers,
    method="POST",
)
with urllib.request.urlopen(request, timeout=900) as response:
    print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
