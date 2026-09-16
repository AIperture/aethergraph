# Multimodal input integrity

Image generation resolves effective model/profile/adapter capabilities before transport,
including per-call model changes and multiple output requests. Explicit profile facts
are required for uncataloged models; provider names do not establish model support.
Configured capability overrides and catalog keys are preserved by the image factory.

OpenAI and Azure image profiles support reference-image edits through their existing
image adapter and `/images/edits` multipart transport. Pass canonical `ImagePart`
objects (the public alias for `ImageInput`) or inline data URLs to
`context.image_model(profile=...).generate_image(..., input_images=[...])`.
The selected profile's `input_policy` bounds count, decoded bytes, dimensions and
normalization through AG's existing media preparer. Image profiles enable this
policy by default; Chat profiles retain their explicit opt-in. Remote URLs and
provider file references are not admitted by the image-generation API.

OpenAI/Azure edits reject options that cannot be projected, including `style` and
`response_format`; unknown extra options fail explicitly. Gemini uses the same
prepared canonical image inputs, with its existing inline image transport; its
unsupported count, size and quality options remain explicit errors. There is no
automatic endpoint/model substitution or text-only retry. Unknown model capability
facts still require an explicit profile declaration.

`LLMUnsupportedFeatureError` exposes `provider`, `model`, `feature` and `detail`.
Malformed/oversized raster inputs raise `MediaPreparationError` before HTTP I/O.
A supported catalog fact is necessary but does not override adapter or input-policy
restrictions. Image outputs continue through the existing canonical artifact owner.

Transport specifications: [OpenAI image edits](https://developers.openai.com/api/reference/resources/images/methods/edit)
and [Azure image generation and editing](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/dall-e).
Regression tests inspect real multipart HTTP requests with ordered PNG bytes and
exact endpoints. They do not certify live credentials, deployment availability or
visual reference fidelity.

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


## GPT Image 2.5 catalog revision 9

Verified on 2026-09-13 against the official [Sunburst model page](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst), [Flare model page](https://developers.openai.com/api/docs/models/gpt-image-2.5-flare) and [image guide](https://developers.openai.com/api/docs/guides/image-generation).

The OpenAI `image_generation` / `openai_images` catalog entries match only
`gpt-image-2.5-sunburst`, `gpt-image-2.5-flare`, and each model's documented
`-2026-09-08` snapshot. Both support text-to-image and image editing. Multiple
outputs are independently supported by the current guide's Image API `n`
parameter documentation. No Chat, audio, video, streaming, mask or other capability
is inferred from the GPT Image 2 entry. Unknown variants and unverified dated
snapshots remain unknown. Azure deployment support is not inferred from OpenAI
model names.

The existing canonical image facade and OpenAI adapter carry prepared reference
bytes to `/images/edits`; no model-specific adapter or fallback is added. Resolver
and mock HTTP transport regressions cover aliases and snapshots without capability
overrides. These checks do not certify account access or a live provider call.

Distribution starts with AG `0.1.0a22`. Update the actual calling interpreter,
restart long-lived workers so their cached catalog reloads, and verify the catalog
digest from that interpreter. Updating a Studio host alone does not update an
existing project interpreter. Explicit `model="gpt-image-2"` calls still select that
model until their author changes them.

## Canonical inline image projection

`prepare_model_request` projects admitted bytes/base64 into the existing shared
`image_url` data-URL representation. Responses, Chat Completions, Azure, Anthropic and
Gemini consume that one preparation contract through their existing adapters. No
caller serializer or remote-URL conversion is required. Active preparation decodes
inline data URLs too; they are not a bypass. Declared image MIME must match decoded
content, whether resizing is enabled or disabled. Preparation and unsupported
capability failures expose stable local stage/code metadata.

Streaming/native-tool combinations remain governed by the existing adapter contract.
Azure Responses is Tool-only and does not implement streaming; canonical streaming
currently does not accept a Tool catalog. These requests fail before transport.
