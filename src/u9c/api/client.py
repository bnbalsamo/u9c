from __future__ import annotations

from contextlib import AsyncExitStack
from functools import singledispatchmethod
from types import TracebackType
from typing import ClassVar, overload

from aiohttp import ClientResponse, ClientSession
from attrs import define, field

from .messages import (
    MESSAGE_CONVERTER,
    AudioResponseFormat,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateCompletionRequest,
    CreateCompletionResponse,
    CreateEditRequest,
    CreateEditResponse,
    CreateEmbeddingsRequest,
    CreateEmbeddingsResponse,
    CreateImageEditRequest,
    CreateImageRequest,
    CreateImageResponse,
    CreateImageVariationRequest,
    CreateModerationRequest,
    CreateModerationResponse,
    CreateTranscriptionRequest,
    CreateTranslationRequest,
    ListModelsRequest,
    ListModelsResponse,
)


class InvalidAuthenticationError(Exception):
    """Raised when an error regarding authentication occurs"""


class RateLimitError(Exception):
    """Raised when an error involving raite limits occurs."""


class ServerError(Exception):
    """Raised when a server error occurs."""


class UnknownError(Exception):
    """Raised when an unknown error is reported by the API."""


# TODO: some of the remaining ugliness in the individual handlers
# could probably be eliminated by using a custom cattrs converter

# TODO: Collapse "standard" endpoints into one method, and use a map
# of RequestT -> (endpoint, ResponseT) to handle dynamic-ness
# Get to remove several methods, but have to write more typing
# overloads >.<
# Also may be more brittle into the future?

# TODO: Configurable user agent string?


@define
class OpenAiApiClient:
    """
    A client for accessing the OpenAI API.

    This class is an async context manager, and should be used as follows:

    ```py
    async with OpenAiApiClient(api_key, username) as api_client:
        response = await api_client.send_request(
            CreateChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Tell me a joke")],
                model="gpt-3.5-turbo",
            )
        )
        print(response)
    ```

    :param api_key: The OpenAI API key to use to make requests by default.
    :type api_key: str

    :param api_root: The URL root for the API. Defaults to "https://api.openai.com/v1"
    :type api_root: str
    """

    api_key: str
    api_root: str = "https://api.openai.com/v1"

    session: ClientSession | None = field(init=False)
    stack: AsyncExitStack = field(init=False, factory=AsyncExitStack)

    # Response status code -> Exception class
    _respcode_to_error: ClassVar[dict[int, type]] = {
        401: InvalidAuthenticationError,
        429: RateLimitError,
        500: ServerError,
    }

    async def __aenter__(self, api_key=None) -> OpenAiApiClient:
        if api_key is None:
            api_key = self.api_key
        self.session = await self.stack.enter_async_context(
            ClientSession(headers={"Authorization": f"Bearer {api_key}"})
        )
        return self

    async def __aexit__(
        self,
        exctype: type[BaseException] | None,
        excinst: BaseException | None,
        exctb: TracebackType | None,
    ) -> None:
        await self.stack.aclose()

    async def _check_response(self, resp: ClientResponse) -> None:
        """Check the response for errors."""
        if resp.status == 200:
            return
        raise self._respcode_to_error.get(resp.status, UnknownError)(await resp.text())

    @overload
    async def send_request(
        self, req: ListModelsRequest, session: ClientSession | None = None
    ) -> ListModelsResponse:
        ...

    @overload
    async def send_request(
        self, req: CreateChatCompletionRequest, session: ClientSession | None = None
    ) -> CreateChatCompletionResponse:
        ...

    @overload
    async def send_request(
        self,
        req: (
            CreateImageRequest | CreateImageVariationRequest | CreateImageEditRequest
        ),
        session: ClientSession | None = None,
    ) -> CreateImageResponse:
        ...

    @overload
    async def send_request(
        self, req: CreateEditRequest, session: ClientSession | None = None
    ) -> CreateEditResponse:
        ...

    @overload
    async def send_request(
        self, req: CreateModerationRequest, session: ClientSession | None = None
    ) -> CreateModerationResponse:
        ...

    @overload
    async def send_request(
        self, req: CreateEmbeddingsRequest, session: ClientSession | None = None
    ) -> CreateEmbeddingsResponse:
        ...

    @overload
    async def send_request(
        self, req: CreateCompletionRequest, session: ClientSession | None = None
    ) -> CreateCompletionResponse:
        ...

    @overload
    async def send_request(
        self, req: CreateTranscriptionRequest, session: ClientSession | None = None
    ) -> str | dict:
        ...

    @overload
    async def send_request(
        self, req: CreateTranslationRequest, session: ClientSession | None = None
    ) -> str | dict:
        ...

    async def send_request(self, req, session=None):
        """
        Send a request to the OpenAI API.
        """
        if session is None:
            session = self.session

        return await self._send_req(req, session)

    @singledispatchmethod
    async def _send_req(self, req, session):
        raise NotImplementedError(f"No request handler for {type(req)}")

    @_send_req.register
    async def _create_completion(
        self, req: CreateCompletionRequest, session: ClientSession
    ) -> CreateCompletionResponse:
        payload = MESSAGE_CONVERTER.unstructure(req)
        async with session.post(f"{self.api_root}/completions", json=payload) as resp:
            await self._check_response(resp)
            return MESSAGE_CONVERTER.structure(
                await resp.json(), CreateCompletionResponse
            )

    @_send_req.register
    async def _create_chat_completion(
        self,
        req: CreateChatCompletionRequest,
        session: ClientSession,
    ) -> CreateChatCompletionResponse:
        payload = MESSAGE_CONVERTER.unstructure(req)
        async with session.post(
            f"{self.api_root}/chat/completions", json=payload
        ) as resp:
            await self._check_response(resp)
            return MESSAGE_CONVERTER.structure(
                await resp.json(), CreateChatCompletionResponse
            )

    @_send_req.register
    async def _list_models(
        self,
        req: ListModelsRequest,
        session: ClientSession,
    ) -> ListModelsResponse:
        async with session.get(f"{self.api_root}/models") as resp:
            await self._check_response(resp)
            resp_data = await resp.json()
        return MESSAGE_CONVERTER.structure(resp_data, ListModelsResponse)

    @_send_req.register
    async def _create_image(
        self,
        req: CreateImageRequest,
        session: ClientSession,
    ) -> CreateImageResponse:
        payload = MESSAGE_CONVERTER.unstructure(req)
        async with session.post(
            f"{self.api_root}/images/generations", json=payload
        ) as resp:
            await self._check_response(resp)
            return MESSAGE_CONVERTER.structure(await resp.json(), CreateImageResponse)

    @_send_req.register
    async def _create_image_variation(
        self,
        req: CreateImageVariationRequest,
        session: ClientSession,
    ) -> CreateImageResponse:
        payload = MESSAGE_CONVERTER.unstructure(req)
        # Do a bit of surgery here for form data/streams...
        if "n" in payload:  # TODO: Custom unstructure hook?
            payload["n"] = str(payload["n"])
        async with session.post(
            f"{self.api_root}/images/variations", data=payload
        ) as resp:
            await self._check_response(resp)
            return MESSAGE_CONVERTER.structure(await resp.json(), CreateImageResponse)

    @_send_req.register
    async def _create_image_edits(  # type: ignore
        self,
        req: CreateImageEditRequest,
        session: ClientSession,
    ) -> CreateImageResponse:
        payload = MESSAGE_CONVERTER.unstructure(req)
        # Do a bit of surgery here for form data/streams...
        if "n" in payload:  # TODO: Custom unstructure hook?
            payload["n"] = str(payload["n"])
        async with session.post(f"{self.api_root}/images/edits", data=payload) as resp:
            await self._check_response(resp)
            return MESSAGE_CONVERTER.structure(await resp.json(), CreateImageResponse)

    @_send_req.register
    async def _create_edit(
        self,
        req: CreateEditRequest,
        session: ClientSession,
    ) -> CreateEditResponse:
        payload = MESSAGE_CONVERTER.unstructure(req)
        async with session.post(f"{self.api_root}/edits", json=payload) as resp:
            await self._check_response(resp)
            return MESSAGE_CONVERTER.structure(await resp.json(), CreateEditResponse)

    @_send_req.register
    async def _create_moderation(
        self,
        req: CreateModerationRequest,
        session: ClientSession,
    ) -> CreateModerationResponse:
        payload = MESSAGE_CONVERTER.unstructure(req)
        async with session.post(f"{self.api_root}/moderations", json=payload) as resp:
            await self._check_response(resp)
            return MESSAGE_CONVERTER.structure(
                await resp.json(), CreateModerationResponse
            )

    @_send_req.register
    async def _create_embeddings(
        self,
        req: CreateEmbeddingsRequest,
        session: ClientSession,
    ) -> CreateEmbeddingsResponse:
        payload = MESSAGE_CONVERTER.unstructure(req)
        async with session.post(f"{self.api_root}/embeddings", json=payload) as resp:
            await self._check_response(resp)
            return MESSAGE_CONVERTER.structure(
                await resp.json(), CreateEmbeddingsResponse
            )

    # There's a lot of response formats here, so just return the data "raw"
    @_send_req.register
    async def _create_transcription(
        self,
        req: CreateTranscriptionRequest,
        session: ClientSession,
    ) -> dict | str:
        payload = MESSAGE_CONVERTER.unstructure(req)
        # Req surgery for format input...
        if "temperature" in payload:  # TODO: Custom unstructure hook?
            payload["temperature"] = str(payload["temperature"])
        async with session.post(
            f"{self.api_root}/audio/transcriptions", data=payload
        ) as resp:
            await self._check_response(resp)
            if req.response_format in {
                AudioResponseFormat.JSON,
                AudioResponseFormat.VERBOSE_JSON,
            }:
                return await resp.json()
            return await resp.text()

    # There's a lot of response formats here, so just return the data "raw"
    @_send_req.register
    async def _create_translation(
        self,
        req: CreateTranslationRequest,
        session: ClientSession,
    ) -> dict | str:
        payload = MESSAGE_CONVERTER.unstructure(req)
        # Req surgery for form input...
        if "temperature" in payload:  # TODO: Custom unstructure hook?
            payload["temperature"] = str(payload["temperature"])
        async with session.post(
            f"{self.api_root}/audio/translations", data=payload
        ) as resp:
            await self._check_response(resp)
            if req.response_format in {
                AudioResponseFormat.JSON,
                AudioResponseFormat.VERBOSE_JSON,
            }:
                return await resp.json()
            return await resp.text()


# TODO: File endpoint handlers
# TODO: Fine-tunes endpoint handlers
