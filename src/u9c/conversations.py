"""Utilities for more easily interacting with chat completions in a chat-like way."""
from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType

import aiofiles  # type: ignore
from attrs import define, field

from u9c.api.client import OpenAiApiClient
from u9c.api.messages import MESSAGE_CONVERTER, ChatMessage, CreateChatCompletionRequest


@define
class ConversationHistory:
    """
    Base class for persisting a conversation history.

    This class doesn't persist the conversation history beyond its lifetime.
    """

    messages: list[ChatMessage] = field(factory=list, init=False)

    async def __aenter__(self):
        await self.load()
        return self

    async def __aexit__(
        self,
        exctype: type[BaseException] | None,
        excinst: BaseException | None,
        exctb: TracebackType | None,
    ) -> None:
        await self.dump()

    async def load(self) -> None:
        self.messages = []

    async def dump(self) -> None:
        pass

    async def add_message(self, msg: ChatMessage) -> None:
        """
        Add a message to the chat history.

        :param msg: The message to add to the chat history
        :type msg: ChatMessage
        """
        self.messages.append(msg)


@define
class JsonFileConversationHistory(ConversationHistory):
    """
    Conversation history that uses JSON files on disk to persist conversations.

    :param filepath: The filepath to load the conversation history from (if it exists)
        and write it to at the conclusion of the conversation.
    :type filepath: Path
    """

    filepath: Path  # TODO: validation

    async def load(self):
        """
        Load the conversation history from disk.
        """
        if self.filepath.exists():
            async with aiofiles.open(self.filepath) as f:
                json_str = await f.read()
                json_data = json.loads(json_str)
            self.messages = [
                MESSAGE_CONVERTER.structure(msg, ChatMessage) for msg in json_data
            ]
        else:
            self.messages = []

    async def dump(self):
        """
        Write the conversation history to disk.
        """
        json_data = [MESSAGE_CONVERTER.unstructure(msg) for msg in self.messages]
        async with aiofiles.open(self.filepath, "w") as f:
            json_str = json.dumps(json_data, indent=2)
            await f.write(json_str)


@define
class Conversation:
    """
    A conversation between a user and an AI.

    Handles formatting `ChatCompletionRequest`s and persisting to
    a conversation history automatically.
    """

    api_client: OpenAiApiClient
    history: ConversationHistory = field(factory=ConversationHistory)
    model: str = field(default="gpt-3.5-turbo")
    user: str | None = None

    async def send_message(
        self, msg: str, model: str | None = None, user: str | None = None
    ) -> str:
        """
        Send a user supplied message to an AI and return the response.
        """
        if model is None:
            model = self.model
        if user is None:
            user = self.user
        await self.history.add_message(ChatMessage.from_str(msg))
        ccreq = CreateChatCompletionRequest(
            messages=self.history.messages, model=model, user=user
        )
        ccresp = await self.api_client.send_request(ccreq)
        response_message = ccresp.choices[0].message
        await self.history.add_message(response_message)
        return response_message.content
