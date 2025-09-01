import re
import torch
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Optional
from collections import Counter


def get_peft_regex(
        model,
        finetune_vision_layers: bool = True,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
        target_modules: List[str] = None,
        vision_tags: Optional[List[str]] = None,
        language_tags: Optional[List[str]] = None,
        attention_tags: Optional[List[str]] = None,
        mlp_tags: Optional[List[str]] = None
) -> str:
    """
    Create a regex pattern to apply LoRA to only select layers of a model.
    This version is optimized for DeepSpeed compatibility.
    """

    if vision_tags is None:
        vision_tags = ["vision", "image", "visual", "patch"]
    if language_tags is None:
        language_tags = ["language", "text"]
    if attention_tags is None:
        attention_tags = ["self_attn", "attention", "attn"]
    if mlp_tags is None:
        mlp_tags = ["mlp", "feed_forward", "ffn", "dense"]

    if not finetune_vision_layers and not finetune_language_layers:
        raise RuntimeError(
                "No layers to finetune - please select to finetune the vision and/or the language layers!"
        )
    if not finetune_attention_modules and not finetune_mlp_modules:
        raise RuntimeError(
                "No modules to finetune - please select to finetune the attention and/or the mlp modules!"
        )

    # Get only linear layers
    modules = model.named_modules()
    linear_modules = [name for name, module in modules if isinstance(module, torch.nn.Linear)]
    all_linear_modules = Counter(x.rsplit(".")[-1] for x in linear_modules)

    # Isolate lm_head / projection matrices if count == 1
    if target_modules is None:
        only_linear_modules = []
        projection_modules = {}
        for j, (proj, count) in enumerate(all_linear_modules.items()):
            if count != 1:
                only_linear_modules.append(proj)
            else:
                projection_modules[proj] = j
    else:
        assert (type(target_modules) is list)
        only_linear_modules = list(target_modules)

    # Create regex matcher
    regex_model_parts = []
    if finetune_vision_layers:
        regex_model_parts += vision_tags
    if finetune_language_layers:
        regex_model_parts += language_tags
    regex_components = []
    if finetune_attention_modules:
        regex_components += attention_tags
    if finetune_mlp_modules:
        regex_components += mlp_tags

    regex_model_parts = "|".join(regex_model_parts)
    regex_components = "|".join(regex_components)

    match_linear_modules = r"(?:" + "|".join(re.escape(x) for x in only_linear_modules) + r")"
    regex_matcher = \
        r".*?(?:" + regex_model_parts + \
        r").*?(?:" + regex_components + \
        r").*?" + match_linear_modules + ".*?"

    # Also account for model.layers.0.self_attn/mlp type modules like Qwen
    if finetune_language_layers:
        regex_matcher = r"(?:" + regex_matcher + \
                        r")|(?:\bmodel\.layers\.[\d]{1,}\.(?:" + regex_components + \
                        r")\.(?:" + match_linear_modules + r"))"

    # Check if regex is wrong since model does not have vision parts
    check = any(re.search(regex_matcher, name, flags=re.DOTALL) for name in linear_modules)
    if not check:
        regex_matcher = \
            r".*?(?:" + regex_components + \
            r").*?" + match_linear_modules + ".*?"

    # Final check to confirm if matches exist
    check = any(re.search(regex_matcher, name, flags=re.DOTALL) for name in linear_modules)
    if not check and target_modules is not None:
        raise RuntimeError(
                f"No layers to finetune? You most likely specified target_modules = {target_modules} incorrectly!"
        )
    elif not check:
        raise RuntimeError(
                f"No layers to finetune for {model.config._name_or_path}. Please file a bug report!"
        )

    return regex_matcher


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    return lora_module_names


class DeepSpeedCompatibleModelParameterManager:
    """
    Enhanced parameter manager specifically designed for DeepSpeed Zero-2 compatibility.
    This version addresses common issues with empty tensor lists and parameter initialization.
    """


    def __init__(
            self,
            model,
            lora_namespan_exclude=[],
            training_args=None,
            **kwargs
            ):
        assert model is not None, "Model cannot be None"
        self.training_args = training_args
        if self.training_args.regex_module:
            self.target_modules = get_peft_regex(
                    model,
                    finetune_vision_layers= (not training_args.freeze_vision_tower and training_args.vision_lora),
                    finetune_language_layers= not training_args.freeze_llm,
                    finetune_attention_modules=True,
                    finetune_mlp_modules=True,
            )
        else:
            self.target_modules = find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules)

    def return_target_modules(self):
        """Return the target modules for LoRA."""
        return self.target_modules

    @staticmethod
    def verify_trainable_parameters(model):
        """Verify that model has trainable parameters and no empty tensors."""
        trainable_params = []
        empty_tensors = []
        total_params = 0

        for name, param in model.named_parameters():
            total_params += param.numel()

            if param.requires_grad:
                trainable_params.append((name, param))

                # Check for empty tensors
                if hasattr(param, 'shape') and any(dim == 0 for dim in param.shape):
                    empty_tensors.append((name, param.shape))
                    print(f"❌ EMPTY TENSOR DETECTED: {name} with shape {param.shape}")

        trainable_count = sum(p.numel() for _, p in trainable_params)

        print(f"📊 Parameter verification:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_count:,}")
        print(f"   Trainable percentage: {100 * trainable_count / total_params:.2f}%")
        print(f"   Empty tensors found: {len(empty_tensors)}")

        if len(empty_tensors) > 0:
            raise RuntimeError(f"❌ Found {len(empty_tensors)} empty tensors that will cause DeepSpeed errors!")

        if len(trainable_params) == 0:
            raise RuntimeError("❌ No trainable parameters found!")

        return trainable_params

    def apply_lora_with_deepspeed_safety(
            self,
            model,
            r: int = 8,
            lora_alpha: int = 16,
            lora_dropout: float = 0.05,
            bias: str = "none",
            **kwargs
    ):

        # Step 2: Get target modules
        target_modules = self.return_target_modules()
        print(f"🎯 Target modules: {target_modules}")

        # Step 3: Create LoRA config with DeepSpeed-compatible settings
        peft_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=bias,
                target_modules=target_modules,

        )
        # Step 4: Apply PEFT
        model = get_peft_model(model, peft_config)
        print("✅ LoRA applied successfully")

        if not self.training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "vision_tower" in name:
                    param.requires_grad = True

        if not self.training_args.freeze_projector:
            for name, param in model.named_parameters():
                if "multi_modal_projector" in name:
                    param.requires_grad = True

        # Step 5: Verify trainable parameters

        return model



def configure_model_for_training(
        model,
        training_args=None,
        **kwargs
):
    """
    DeepSpeed-compatible model configuration function.
    """



    # Create manager with model
    manager = DeepSpeedCompatibleModelParameterManager(
            model,
            training_args=training_args,
            **kwargs
    )

    # Apply the strategy
    try:
        configured_model = manager.apply_lora_with_deepspeed_safety(model, **kwargs)
        print("✅ Model configuration completed successfully")
        return configured_model
    except Exception as e:
        print(f"❌ Model configuration failed: {e}")
        print("💡 Try using 'lora_conservative' strategy for maximum compatibility")
        raise


# Test function for debugging
def test_model_configuration():
    """Test function to verify the configuration works."""
    try:
        from transformers import AutoModelForCausalLM

        print("🧪 Testing model configuration...")

        # Load a small model for testing
        model_name = "microsoft/DialoGPT-small"
        model = AutoModelForCausalLM.from_pretrained(model_name)

        print(f"📥 Loaded model: {model_name}")

        # Test conservative configuration
        configured_model = configure_model_for_training(
                model,
                strategy="lora_conservative",
                r=8,
                lora_alpha=16,
                target_modules=["c_attn", "c_proj"]
        )

        print("✅ Test completed successfully!")
        return configured_model

    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers")
        return None
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    print("🚀 DeepSpeed-Compatible PEFT Configuration loaded!")

    # Run test if requested
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_model_configuration()
    else:
        print("💡 To run test: python fixed_peft.py --test")