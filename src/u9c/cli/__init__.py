"""
The u9c CLI.

These CLIs are meant to be examples - they aren't feature complete.
"""

import asyncio
import readline  # noqa: F401 - more functionality for conversation input
from contextlib import AsyncExitStack, ExitStack
from functools import wraps
from hashlib import md5
from pathlib import Path
from typing import Callable, Dict, TypeVar, cast

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.status import Status

from u9c.api.client import OpenAiApiClient
from u9c.api.messages import (
    ChatMessage,
    CreateChatCompletionRequest,
    CreateCompletionRequest,
    CreateEditRequest,
    CreateImageEditRequest,
    CreateImageRequest,
    CreateImageVariationRequest,
    CreateTranscriptionRequest,
    CreateTranslationRequest,
    ImageResponseFormat,
    ImageSizes,
)
from u9c.conversations import (
    Conversation,
    ConversationHistory,
    JsonFileConversationHistory,
)
from u9c.misc import hash_username as _hash_username


def hash_username(username: str) -> str:
    return _hash_username(md5, username, b"u9c")


_T = TypeVar("_T", bound=Callable)


def _syncify(async_fn: _T) -> _T:
    """
    Decorator that turns an async function into a sync function.

    Used to make click play nice with async defs.
    """

    @wraps(async_fn)
    def deco(*args, **kwargs):
        return asyncio.run(async_fn(*args, **kwargs))

    return cast(_T, deco)


# TODO: Embeddings CLI?
# TODO: Files CLI?


# Reused arguments
KEY_ARG = click.option("-k", "--key", required=True, help="Your OpenAI API key.")
USERNAME_ARG = click.option("-u", "--username", required=True, help="Your username.")


@click.group()
def u9c():
    """The root parser for the u9c CLI. Select from subcommands below."""


##########
# Images #
##########

SIZE_ARG = click.option(
    "-s",
    "--size",
    type=click.Choice(["small", "medium", "large"], case_sensitive=False),
    default="small",
    help="The size of image(s) to generate. Small=256x26, Medium=512x512, Large=1024x1024",
)

IMG_NUM_ARG = click.option(
    "-n",
    "--number",
    default=1,
    type=click.IntRange(min=0, max=10),
    help="The number of images to generate.",
)

DL_LOC_ARG = click.option(
    "-d",
    "--download-location",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
        readable=True,
        writable=True,
    ),
    default=Path("."),
)

# Normalized str -> Enum member
IMAGE_SIZE_ARG_MAP: Dict[str, ImageSizes] = {
    "small": ImageSizes.SMALL,
    "medium": ImageSizes.MEDIUM,
    "large": ImageSizes.LARGE,
}


@u9c.group()
def images():
    """Subparser for image based functionalities. Select from subcommands below."""


@images.command()
@KEY_ARG
@USERNAME_ARG
@IMG_NUM_ARG
@SIZE_ARG
@DL_LOC_ARG
@click.argument("prompt", type=str)
@_syncify
async def generate(
    key: str,
    username: str,
    number: int,
    size: str,
    download_location: Path,
    prompt: str,
) -> int:
    """Generate an image given a prompt."""
    parsed_size = IMAGE_SIZE_ARG_MAP[size.casefold()]

    async with OpenAiApiClient(key) as api:
        resp = await api.send_request(
            CreateImageRequest(
                prompt,
                number,
                parsed_size,
                response_format=ImageResponseFormat.B64_JSON,
                user=hash_username(username),
            ),
        )
        await resp.save_images(download_location)
    return 0


@images.command()
@KEY_ARG
@USERNAME_ARG
@IMG_NUM_ARG
@SIZE_ARG
@DL_LOC_ARG
@click.option(
    "-i",
    "--image",
    "image_path",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@_syncify
async def variations(
    key: str,
    username: str,
    number: int,
    size: str,
    download_location: Path,
    image_path: Path,
) -> int:
    """Generate variations of an image. The image must be a square PNG under 4MB"""
    parsed_size = IMAGE_SIZE_ARG_MAP[size.casefold()]

    with open(image_path, "rb") as image_data:
        async with OpenAiApiClient(key) as api:
            resp = await api.send_request(
                CreateImageVariationRequest(
                    image_data,
                    number,
                    parsed_size,
                    response_format=ImageResponseFormat.B64_JSON,
                    user=hash_username(username),
                ),
            )
            await resp.save_images(download_location)
    return 0


@images.command()
@KEY_ARG
@USERNAME_ARG
@IMG_NUM_ARG
@SIZE_ARG
@DL_LOC_ARG
@click.option(
    "-i",
    "--image",
    "image_path",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "-m",
    "--mask",
    "mask_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    required=True,
)
@click.argument("prompt")
@_syncify
async def edits(
    key: str,
    username: str,
    number: int,
    size: str,
    download_location: Path,
    image_path: Path,
    mask_path: Path,
    prompt: str,
) -> int:
    """
    Generate edits of an image. The image must be a square PNG under 4MB

    The mask is an image whose fully transparent areas (e.g. where alpha is zero) indicate where image should be edited.
    Must be a valid PNG file, less than 4MB, and have the same dimensions as image.
    """
    parsed_size = IMAGE_SIZE_ARG_MAP[size.casefold()]

    with ExitStack() as stack:
        image_data = stack.enter_context(open(image_path, "rb"))
        mask_data = stack.enter_context(open(mask_path, "rb"))

        async with OpenAiApiClient(key) as api:
            resp = await api.send_request(
                CreateImageEditRequest(
                    image=image_data,
                    prompt=prompt,
                    mask=mask_data,
                    size=parsed_size,
                    n=number,
                    response_format=ImageResponseFormat.B64_JSON,
                    user=hash_username(username),
                ),
            )
            await resp.save_images(download_location)
    return 0


###############
# Completions #
###############


@u9c.command()
@KEY_ARG
@USERNAME_ARG
@click.option("-m", "--model", default="text-davinci-003")
@click.argument("prompt")
@_syncify
async def completion(key, username, model, prompt):
    """Create a Completion from a prompt"""

    async with OpenAiApiClient(key) as api_client:
        resp = await api_client.send_request(
            CreateCompletionRequest(
                model=model,
                prompt=prompt,
                user=hash_username(username),
            )
        )
        console = Console()
        console.print(Markdown(resp.choices[0].text.strip()))
    return 0


####################
# Chat Completions #
####################


@u9c.command()
@KEY_ARG
@USERNAME_ARG
@click.option("-m", "--model", default="gpt-3.5-turbo")
@click.argument("message")
@_syncify
async def chat(key, username, model, message):
    """Send a one-off message to ChatGPT and get a response."""

    async with OpenAiApiClient(key) as api_client:
        req = CreateChatCompletionRequest(
            messages=[ChatMessage.from_str(message)],
            model=model,
            user=hash_username(username),
        )
        resp = await api_client.send_request(req)
        console = Console()
        console.print(Markdown(resp.choices[0].message.content.strip()))
    return 0


@u9c.command()
@KEY_ARG
@USERNAME_ARG
@click.option("-m", "--model", default="gpt-3.5-turbo")
@click.option("-c", "--conversation-history-file", default=None)
@_syncify
async def conversation(key, username, model, conversation_history_file):
    """Have a conversation with ChatGPT, saving history."""

    stack = AsyncExitStack()
    async with stack:
        api_client = await stack.enter_async_context(OpenAiApiClient(key))
        if conversation_history_file is not None:
            conversation_history = await stack.enter_async_context(
                JsonFileConversationHistory(Path(conversation_history_file))
            )
        else:
            conversation_history = await stack.enter_async_context(
                ConversationHistory()
            )
        convo = Conversation(
            api_client, conversation_history, user=hash_username(username)
        )

        console = Console()
        console.print(
            "[yellow]Entering an interactive conversation, type 'exit' or 'quit' to quit.[/yellow]"
        )
        while True:
            message = console.input("[blue]You:[/blue] ")
            if message in {"exit", "quit"}:
                break

            with Status("Thinking..."):
                resp_message = await convo.send_message(message, model=model)

            console.print("[red]AI:[/red] ", end="")
            console.print(Markdown(resp_message))

    return 0


#########
# Edits #
#########


@u9c.command()
@KEY_ARG
@USERNAME_ARG
@click.option("-m", "--model", default="text-davinci-edit-001")
@click.option("-i", "--input", "input_", default="")
@click.argument("instruction")
@_syncify
async def edit(key, username, model, input_, instruction):
    """Send an edit request to ChatGPT and get a response."""

    async with OpenAiApiClient(key) as api_client:
        req = CreateEditRequest(
            instruction=instruction,
            model=model,
            input=input_,
            user=hash_username(username),
        )
        resp = await api_client.send_request(req)
        console = Console()
        console.print(Markdown(resp.choices[0].text.strip()))
    return 0


#########
# Audio #
#########


@u9c.group()
def audio():
    """Subparser for audio based functionalities. Select from subcommands below."""


@audio.command()
@KEY_ARG
@click.option("-m", "--model", default="whisper-1")
@click.argument("audio_file")
@_syncify
async def translation(key, model, audio_file):
    """Translate an audio file via OpenAI"""

    async with OpenAiApiClient(key) as api_client:
        with open(audio_file, "rb") as audio_data:
            resp = await api_client.send_request(
                CreateTranslationRequest(
                    audio_data,
                    model=model,
                )
            )
            print(resp)
    return 0


@audio.command()
@KEY_ARG
@click.option("-m", "--model", default="whisper-1")
@click.argument("audio_file")
@_syncify
async def transcription(key, model, audio_file):
    """Transcribe an audio file via OpenAI"""

    async with OpenAiApiClient(key) as api_client:
        with open(audio_file, "rb") as audio_data:
            resp = await api_client.send_request(
                CreateTranscriptionRequest(
                    audio_data,
                    model=model,
                )
            )
            print(resp)
    return 0


if __name__ == "__main__":
    u9c()
