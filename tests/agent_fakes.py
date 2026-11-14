from llm_harness_lite.llm.types import ChatResponse, Usage


class FakeLLM:
    model = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, temperature=0.2):
        self.calls.append(list(messages))
        return self.responses.pop(0)


def response(message, finish="stop", cost=0.001):
    return ChatResponse(
        message=message,
        finish=finish,
        usage=Usage(input_tokens=2, output_tokens=1, cost_usd=cost, model="fake"),
    )
