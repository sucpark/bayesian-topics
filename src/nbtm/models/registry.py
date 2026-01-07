"""Model registry for creating models by name."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Type

if TYPE_CHECKING:
    from nbtm.models.base import BaseTopicModel

# Model registry
MODEL_REGISTRY: Dict[str, Type[BaseTopicModel]] = {}


def register_model(name: str):
    """
    Decorator to register a model class.

    Usage:
        @register_model("lda_gibbs")
        class GibbsLDA(BaseTopicModel):
            ...
    """

    def decorator(cls: Type[BaseTopicModel]) -> Type[BaseTopicModel]:
        MODEL_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def create_model(name: str, **kwargs: Any) -> BaseTopicModel:
    """
    Create a model by name.

    Args:
        name: Model name (e.g., "lda_gibbs", "hdp")
        **kwargs: Model-specific arguments

    Returns:
        Model instance

    Raises:
        ValueError: If model name is not registered
    """
    name = name.lower()

    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys()) or "none"
        raise ValueError(
            f"Unknown model: '{name}'. Available models: {available}"
        )

    model_class = MODEL_REGISTRY[name]

    # Filter kwargs to valid parameters
    import inspect

    valid_params = set(inspect.signature(model_class.__init__).parameters.keys())
    valid_params.discard("self")
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return model_class(**filtered_kwargs)


def get_available_models() -> List[str]:
    """Get list of available model names."""
    return list(MODEL_REGISTRY.keys())
