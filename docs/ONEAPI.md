# OneAPI configuration

Create an OpenAI-compatible channel:

```text
Type: OpenAI
Base URL: http://127.0.0.1:8010
Models: GLM-5.3-Flash,glm-5.3-flash
```

The vLLM endpoint itself does not require a key by default. Authentication and quotas can be enforced by OneAPI.

## Images

OneAPI forwards standard OpenAI chat image messages. Client applications must mark the custom model as Vision/image-capable so that attachments are sent as `messages[].content[].image_url` rather than plain text.

## Timeouts and 502

Long-running requests can sit in the vLLM capacity queue before the first token. Configure the outermost proxy and client for a timeout of several minutes. A gateway-side `Broken pipe` usually means the downstream client/proxy disconnected first.

## Model list

Standard `/v1/models` responses do not advertise a universal `vision=true` field. GUI clients may therefore require a one-time manual Vision capability toggle for custom model names.
