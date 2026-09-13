# Multimodal input integrity

Image generation resolves effective model/profile/adapter capabilities before transport,
including per-call model changes and multiple output requests. Explicit profile facts
are required for uncataloged models; provider names do not establish model support.
Configured capability overrides and catalog keys are preserved by the image factory.

At this implementation stage, OpenAI and Azure image-edit transports are not implemented:
nonempty `input_images` raise `LLMUnsupportedFeatureError` rather than becoming a
prompt-only generation request. An override cannot bypass the adapter restriction.
Gemini retains its implemented reference-image transport; common options the adapter
cannot project are rejected, including non-default output count, size and quality.
There is no automatic endpoint/model switch or text-only retry.

Legacy `chat` and `chat_stream` validate content shapes even without a managed image
policy. Canonical text/images are supported according to the selected profile.
Unknown/malformed content, PDF/document/audio/video and unimplemented file-reference
forms fail with message/part indexes; they are never silently omitted by conversion.
Provider-specific availability does not imply that the AG endpoint contract admits it.
Ordinary canonical Tool continuations retain their existing dedicated contracts.

Model facts belong in the existing model catalog, adapter implementation limits in
the existing endpoint registry, and effective support in the capability resolver.
Full document/audio/video catalog and transport coverage remains pending; callers
must not infer it from successful text or image calls. Transport spies and local
contract tests establish projection/rejection behavior, not live provider acceptance.

## Model facts and effective input support

The existing catalog now owns an `input_media` domain. Initial exact model records cover GPT-5.2, Claude Sonnet 4.6 and Gemini 2.5 Flash; records carry provider documentation URLs, verification dates and revision 8. The existing effective Chat resolver consumes `image_input` from these records and still applies adapter restrictions and explicit overrides. Unknown model IDs do not inherit support from similar names or per-call model changes.

Audio/video/document entries describe model facts only; they do not advertise an implemented AG transport. The current normalized Chat boundary accepts text/images and rejects unsupported media parts explicitly. In particular, Gemini's documented audio/video support does not make the current AG serializer support these payloads. No provider substitution or text-only retry is performed.

Evidence: [GPT-5.2 modalities](https://developers.openai.com/api/docs/models/gpt-5.2), [Claude Sonnet 4.6 inputs](https://platform.claude.com/docs/en/models/sonnet-4-6/overview), [Gemini 2.5 Flash inputs](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash). Provider facts are documentation-verified; live endpoint acceptance has not been certified by these regression tests.
