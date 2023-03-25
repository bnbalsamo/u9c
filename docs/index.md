# Unofficial AsyncIO OpenAI API Client

**U**nofficial **A**sync**IO** **O**pen**AI** **API** **C**lient -> **uaiooaiapic** -> **u9c**

**Currently a work in progress**

`u9c` is an unofficial asynchronous OpenAI API client.

It also includes a command line interface (CLI) for utilizing the OpenAI API
and some other utilities.

## Usage

### As a Library

```python
import asyncio

from u9c.api.client import OpenAiApiClient
from u9c.api.messages import ChatMessage, CreateChatCompletionRequest


OPENAI_API_KEY = "abc123"  # Insert your API key here.


async def main():
    async with OpenAiApiClient(OPENAI_API_KEY) as api_client:
        response = await api_client.send_request(
            CreateChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Tell me a joke")],
                model="gpt-3.5-turbo",
            )
        )
    print(response)

if __name__ == "__main__":
    asyncio.run(main)
```

### On the CLI

See `u9c --help` for details on CLI usage.

## Installation

For the moment u9c is only installable via source.

First git clone the repository:

```sh
$ git clone https://github.com/bnbalsamo/u9c.git
```

Then `cd` into the cloned repository:

```sh
$ cd u9c
```

Then use `pip` to install the package:

```sh
$ pip install .
```

!!! note

    Remember to have your preferred virtual environment active before installing the package!

## Contributing

Contributions are welcome in the form of issues or pull requests.

See the instructions below for details on how to set up a development environment if you would like to contribute code.

## Development

### Setting Up a Development Environment

Git clone the repository:

```sh
$ git clone https://github.com/bnbalsamo/u9c.git
```

Then `cd` into the cloned repository:

```sh
$ cd u9c
```

Create and activate a virtual environment:

```sh
$ python3 -m venv venv
```

```sh
$ source venv/bin/activate
```

Install the development dependencies:

```sh
$ pip install -r dev-requirements.txt
```

Install the project in editable mode:

```sh
$ python -m pip install -e .
```

Confirm the package has been installed by printing the CLI help:

```sh
$ u9c --help
```
