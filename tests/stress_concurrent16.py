import concurrent.futures
import json
import time
import urllib.request


def run_request(index: int) -> dict:
    payload = {
        "model": "glm-5.3-flash",
        "messages": [
            {
                "role": "user",
                "content": (f"request-{index} context " * 300)
                + "Write numbered items until the output limit.",
            }
        ],
        "max_tokens": 64,
        "min_tokens": 64,
        "ignore_eos": True,
        "temperature": 0,
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8010/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    usage = result.get("usage") or {}
    return {
        "index": index,
        "finish_reason": (result.get("choices") or [{}])[0].get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


started = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    results = list(executor.map(run_request, range(1, 17)))
print(json.dumps({"wall_seconds": round(time.monotonic() - started, 2), "results": results}))
