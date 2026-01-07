"""Topic model implementations."""

from nbtm.models.base import BaseTopicModel, TopicModelState
from nbtm.models.registry import create_model, get_available_models, register_model

# Import models to register them
from nbtm.models.lda_gibbs import GibbsLDA

__all__ = [
    "BaseTopicModel",
    "TopicModelState",
    "create_model",
    "get_available_models",
    "register_model",
    "GibbsLDA",
]
