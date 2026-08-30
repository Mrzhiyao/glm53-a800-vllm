import json
import time
import urllib.request


payload = {
    "model": "glm-5.3-flash",
    "messages": [
        {
            "role": "user",
            "content": ("context " * 1500)
            + "Continue by writing short numbered items until the output limit.",
        }
    ],
    "max_tokens": 512,
    "min_tokens": 512,
    "ignore_eos": True,
    "temperature": 0,
}
request = urllib.request.Request(
    "http://127.0.0.1:8010/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
started = time.monotonic()
with urllib.request.urlopen(request, timeout=900) as response:
    status = response.status
    result = json.load(response)
elapsed = time.monotonic() - started
choices = result.get("choices") or []
message = choices[0].get("message", {}) if choices else {}
usage = result.get("usage") or {}
print(
    {
        "status": status,
        "model": result.get("model"),
        "finish_reason": choices[0].get("finish_reason") if choices else None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        ),
        "content_chars": len(message.get("content") or ""),
        "reasoning_chars": len(message.get("reasoning") or ""),
        "elapsed_seconds": round(elapsed, 2),
        "error": result.get("error"),
    }
)
