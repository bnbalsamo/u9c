"""
Wrapper classes for API requests and responses + their components.

These classes provide input validation as well as some added
functionality for certain messages.
"""

from __future__ import annotations

from base64 import b64decode
from datetime import datetime
from enum import Enum, unique
from io import BufferedReader
from pathlib import Path
from typing import Sequence

import aiofiles  # type: ignore
from attrs import define, field
from attrs.validators import ge, le, max_len
from cattrs import Converter

# TODO: Fine-tunes
# TODO: expected_response_type classvar to facilitate easier
# handling by a dispatch style method in the client?
# dict, string, or ResponseT?

#: Converts message classes <-> API payloads
MESSAGE_CONVERTER = Converter(omit_if_default=True, forbid_extra_keys=True)


# List Models


@define
class ListModelsRequest:
    pass


@define
class ListModelsResponse:
    data: list[ModelData]
    object: str

    @property
    def model_names(self):
        return [model_desc.id for model_desc in self.data]


@define
class ModelData:
    id: str
    object: str
    created: int
    owned_by: str
    permission: list[ModelPermission]
    root: str
    parent: str | None  # TODO: Confirm correctness


@define
class ModelPermission:
    id: str
    object: str
    created: int
    allow_create_engine: bool
    allow_sampling: bool
    allow_logprobs: bool
    allow_search_indices: bool
    allow_view: bool
    allow_fine_tuning: bool
    organization: str
    group: str | None  # TODO: Confirm correctness
    is_blocking: bool


# Completions
@define
class CreateCompletionRequest:
    model: str
    prompt: str | list[str] | list[float] | list[list[float]] = "<|endoftext|>"
    suffix: str | None = None
    max_tokens: int = field(
        default=16, validator=[ge(1)]
    )  # Max is model dependent, either 2048 or 4096
    temperature: float = field(default=1, validator=[ge(0), le(2)])
    top_p: float = field(default=1, validator=[ge(0), le(1)])
    n: int = 1
    stream: bool = False
    logprobs: int | None = field(
        default=None,
        validator=lambda instance, attribute, value: value is None or 0 < value <= 5,
    )
    echo: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float = field(default=0, validator=[ge(-2), le(2)])
    frequency_penality: float = field(default=0, validator=[ge(-2), le(2)])
    best_of: int = 1
    logit_bias: dict[str, float] | None = None
    user: str | None = None


@define
class CompletionOption:
    text: str
    index: int
    logprobs: int | None  # TODO: confirm
    finish_reason: str


@define
class CreateCompletionResponse:
    id: str
    object: str
    created: int
    model: str
    choices: list[CompletionOption]
    usage: UsageInfo


# Chat Completions


@define
class ChatMessage:
    role: str
    content: str

    @classmethod
    def from_str(cls, content: str, role: str = "user") -> ChatMessage:
        """
        Generate a `ChatMessage` from a message string.

        :param content: The contents of the message.
        :type content: str
        :param role: The role associated with the sender of the message.
        :type role: str | None

        :returns: A chat message.
        """
        return cls(role, content)


@define
class CreateChatCompletionRequest:
    messages: Sequence[ChatMessage]
    model: str
    temperature: int | float = field(default=1, validator=[ge(0), le(2)])
    top_p: int | float = field(default=1, validator=[ge(0), le(1)])
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | float = field(
        default=float("Infinity"),
        validator=lambda instance, attribute, value: isinstance(value, int)
        or value == float("Infinity"),
    )
    presence_penalty: int | float = field(default=0, validator=[ge(-2), le(2)])
    frequency_penalty: int | float = field(default=0, validator=[ge(-2), le(2)])
    logit_bias: dict[str, float] | None = None
    user: str | None = None


@define
class ChatChoice:
    index: int
    message: ChatMessage
    finish_reason: str


@define
class UsageInfo:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@define
class CreateChatCompletionResponse:
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo


# Edits


@define
class CreateEditRequest:
    instruction: str
    model: str
    input: str = ""
    n: int = 1
    temperature: int | float = field(default=1, validator=[ge(0), le(2)])
    top_p: int | float = field(default=1, validator=[ge(0), le(1)])
    user: str | None = None


@define
class EditOption:
    text: str
    index: int


@define
class CreateEditResponse:
    object: str
    created: int
    choices: list[EditOption]
    usage: UsageInfo


# Image Generation/Variations/Edits


@unique
class ImageSizes(Enum):
    """Enum describing valid sizes that can be passed to image generation endpoints"""

    SMALL: str = "256x256"
    MEDIUM: str = "512x512"
    LARGE: str = "1024x1024"


@unique
class ImageResponseFormat(Enum):
    """Enum describing valid response formats that can be passed to image generation endpoints"""

    URL: str = "url"
    B64_JSON: str = "b64_json"


@define
class CreateImageResponse:
    created: int
    data: list

    def generate_base_filename(self):
        return str(int(datetime.timestamp(datetime.utcnow()) * 1000000))

    async def save_images(self, output_dir: Path):
        # TODO: Better base filename
        # make sure to retain sort order
        base_filename = self.generate_base_filename()
        # TODO: Support URL
        for i, image_entry in enumerate(self.data):
            if "b64_json" not in image_entry:
                raise NotImplementedError(
                    "Only responses generated via B64_JSON requests are supported."
                )
            async with aiofiles.open(
                output_dir / f"{base_filename}_{str(i)}.png", "wb"
            ) as f:
                await f.write(b64decode(image_entry["b64_json"]))


@define
class CreateImageRequest:
    prompt: str = field(validator=max_len(1000))
    n: int = field(default=1, validator=[le(10), ge(1)])
    size: ImageSizes = ImageSizes.MEDIUM
    response_format: ImageResponseFormat = ImageResponseFormat.URL
    user: str | None = None


@define
class CreateImageVariationRequest:
    image: BufferedReader
    n: int = field(default=1, validator=[le(10), ge(1)])
    size: ImageSizes = ImageSizes.MEDIUM
    response_format: ImageResponseFormat = ImageResponseFormat.URL
    user: str | None = None


@define
class CreateImageEditRequest:
    image: BufferedReader
    prompt: str = field(validator=max_len(1000))
    mask: BufferedReader | None = None
    n: int = 1
    size: ImageSizes = ImageSizes.MEDIUM
    response_format: ImageResponseFormat = ImageResponseFormat.URL
    user: str | None = None


# Moderation Request


@define
class CreateModerationRequest:
    input: str
    model: str = "text-moderation-latest"


# TODO: TypedDict of categories? Are they stable?


@define
class ModerationReport:
    categories: dict
    category_scores: dict
    flagged: bool


@define
class CreateModerationResponse:
    id: str
    model: str
    results: list[ModerationReport]


@define
class CreateEmbeddingsRequest:
    model: str
    input: str
    user: str | None = None


@define
class EmbeddingInfo:
    object: str
    embedding: list[float]
    index: int


@define
class EmbeddingsUsageInfo:
    prompt_tokens: int
    total_tokens: int


@define
class CreateEmbeddingsResponse:
    object: str
    data: list[EmbeddingInfo]
    model: str
    usage: EmbeddingsUsageInfo


@unique
class AudioResponseFormat(Enum):
    JSON = "json"
    TEXT = "text"
    SRT = "srt"
    VERBOSE_JSON = "verbose_json"
    VTT = "vtt"


@define
class CreateTranscriptionRequest:
    file: BufferedReader
    model: str
    prompt: str | None = None
    response_format: AudioResponseFormat = AudioResponseFormat.JSON
    temperature: int | float = field(default=0, validator=[ge(0), le(1)])
    language: str | None = None


@define
class CreateTranslationRequest:
    file: BufferedReader
    model: str
    prompt: str | None = None
    response_format: AudioResponseFormat = AudioResponseFormat.JSON
    temperature: int | float = field(default=0, validator=[ge(0), le(1)])


@define
class ListFilesRequest:
    pass


@define
class RetrieveFileRequest:
    file_id: str


@define
class RetrieveFileResponse:
    id: str
    object: str
    bytes: int
    created_at: int
    filename: str
    purpose: str


@define
class ListFilesResponse:
    object: str
    data: list[RetrieveFileResponse]


@define
class UploadFileRequest:  # Returns RetrieveFileResponse
    filepath: Path
    purpose: str


@define
class DeleteFileRequest:
    file_id: str


@define
class DeleteFileResponse:
    id: str
    object: str
    deleted: bool


@define
class RetrieveFileContentRequest:
    file_id: str
