import pytest

from llm_harness_lite.llm.factory import create_llm


def test_unknown_provider_is_rejected_instead_of_defaulting():
    with pytest.raises(ValueError, match="Unknown LLM provider.*openai_compat"):
        create_llm("opena_compat", "model")


def test_custom_provider_factory_is_registered():
    class CustomBackend:
        def __init__(self, model, endpoint=None):
            self.model = model
            self.endpoint = endpoint

    from llm_harness_lite.llm.factory import register_provider

    register_provider("test_custom_backend", CustomBackend)
    backend = create_llm("TEST_CUSTOM_BACKEND", "custom-model", endpoint="local")

    assert backend.model == "custom-model"
    assert backend.endpoint == "local"
