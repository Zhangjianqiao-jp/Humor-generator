from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from src.models.qwen_vl_lora_loader import load_qwen_vl_with_lora
from src.models.qwen_vl_lora_inference import load_qwen_vl_lora_for_inference


class _Parameter:
    requires_grad = False

    def numel(self) -> int:
        return 1


class _Model:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_cache=True)
        self.generation_config = SimpleNamespace(use_cache=True)
        self.quantization_config = None
        self.parameters = [
            ("q_proj.lora_A.default.weight", _Parameter()),
            ("q_proj.lora_B.default.weight", _Parameter()),
        ]

    def named_parameters(self):
        return self.parameters


def _attach_lora(model: _Model, _config: object) -> _Model:
    for _, parameter in model.named_parameters():
        parameter.requires_grad = True
    return model


def test_qlora_branch_builds_nf4_config_and_prepares_base(monkeypatch) -> None:
    calls: dict[str, object] = {}
    base_model = _Model()

    class BitsAndBytesConfig:
        def __init__(self, **kwargs) -> None:
            calls["quantization"] = kwargs

    class QwenModel:
        @staticmethod
        def from_pretrained(*_args, **kwargs):
            base_model.quantization_config = kwargs["quantization_config"]
            return base_model

    class AutoProcessor:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return SimpleNamespace(image_processor=SimpleNamespace(min_pixels=0, max_pixels=0))

    peft = ModuleType("peft")
    peft.LoraConfig = lambda **kwargs: kwargs
    peft.PeftModel = object
    peft.get_peft_model = _attach_lora
    peft.prepare_model_for_kbit_training = lambda model: calls.setdefault("prepared", model) or model
    transformers = ModuleType("transformers")
    transformers.AutoProcessor = AutoProcessor
    transformers.BitsAndBytesConfig = BitsAndBytesConfig
    transformers.Qwen2_5_VLForConditionalGeneration = QwenModel
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    model, processor = load_qwen_vl_with_lora(
        model_name="unit-test-model",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj"],
        load_in_4bit=True,
        image_max_pixels=200704,
    )

    assert model is base_model
    assert calls["prepared"] is base_model
    assert calls["quantization"] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": __import__("torch").bfloat16,
    }
    assert processor.image_processor.max_pixels == 200704
    assert all(parameter.requires_grad for _, parameter in model.named_parameters())


def test_qlora_allows_processor_without_pixel_attributes(monkeypatch) -> None:
    base_model = _Model()

    class BitsAndBytesConfig:
        def __init__(self, **_kwargs) -> None:
            pass

    class QwenModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return base_model

    class AutoProcessor:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return SimpleNamespace(image_processor=object())

    peft = ModuleType("peft")
    peft.LoraConfig = lambda **kwargs: kwargs
    peft.PeftModel = object
    peft.get_peft_model = _attach_lora
    peft.prepare_model_for_kbit_training = lambda model: model
    transformers = ModuleType("transformers")
    transformers.AutoProcessor = AutoProcessor
    transformers.BitsAndBytesConfig = BitsAndBytesConfig
    transformers.Qwen2_5_VLForConditionalGeneration = QwenModel
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    model, _ = load_qwen_vl_with_lora(
        model_name="unit-test-model",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj"],
        load_in_4bit=True,
        image_max_pixels=200704,
    )

    assert model is base_model


def test_inference_loader_quantizes_base_before_loading_adapter(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}
    base_model = SimpleNamespace(eval=lambda: None)
    adapter_model = SimpleNamespace(eval=lambda: calls.setdefault("eval", True))

    class BitsAndBytesConfig:
        def __init__(self, **kwargs) -> None:
            calls["quantization"] = kwargs

    class QwenModel:
        @staticmethod
        def from_pretrained(*_args, **kwargs):
            calls["model_kwargs"] = kwargs
            return base_model

    class AutoProcessor:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return object()

    class PeftModel:
        @staticmethod
        def from_pretrained(model, adapter_dir):
            calls["adapter"] = (model, adapter_dir)
            return adapter_model

    peft = ModuleType("peft")
    peft.PeftModel = PeftModel
    qwen_utils = ModuleType("qwen_vl_utils")
    qwen_utils.process_vision_info = object()
    transformers = ModuleType("transformers")
    transformers.AutoProcessor = AutoProcessor
    transformers.BitsAndBytesConfig = BitsAndBytesConfig
    transformers.Qwen2_5_VLForConditionalGeneration = QwenModel
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", qwen_utils)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    model, _, _ = load_qwen_vl_lora_for_inference(
        "unit-test-model",
        adapter_dir,
        load_in_4bit=True,
    )

    assert model is adapter_model
    assert calls["adapter"] == (base_model, adapter_dir)
    assert calls["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert "quantization_config" in calls["model_kwargs"]
    assert calls["eval"] is True


def test_inference_loader_can_use_quantized_base_without_adapter(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class BaseModel:
        def eval(self):
            calls["eval"] = True

    base_model = BaseModel()

    class BitsAndBytesConfig:
        def __init__(self, **kwargs) -> None:
            calls["quantization"] = kwargs

    class QwenModel:
        @staticmethod
        def from_pretrained(*_args, **kwargs):
            calls["model_kwargs"] = kwargs
            return base_model

    class AutoProcessor:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return object()

    peft = ModuleType("peft")
    peft.PeftModel = object
    qwen_utils = ModuleType("qwen_vl_utils")
    qwen_utils.process_vision_info = object()
    transformers = ModuleType("transformers")
    transformers.AutoProcessor = AutoProcessor
    transformers.BitsAndBytesConfig = BitsAndBytesConfig
    transformers.Qwen2_5_VLForConditionalGeneration = QwenModel
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", qwen_utils)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    model, _, _ = load_qwen_vl_lora_for_inference(
        "unit-test-model",
        None,
        load_in_4bit=True,
    )

    assert model is base_model
    assert calls["eval"] is True
    assert "quantization_config" in calls["model_kwargs"]
