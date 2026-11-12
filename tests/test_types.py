from llm_harness_lite.llm.types import Message, ToolSpec
from llm_harness_lite.tools.registry import ToolRegistry

def test_registry_schema():
    r = ToolRegistry()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    r.register(add)
    assert r.specs()[0].parameters["required"] == ["a", "b"]
    assert r.call("add", {"a": 2, "b": 3}) == "5"

def test_message_shape():
    m = Message(role="user", content="hi")
    assert m.role == "user"
