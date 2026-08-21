"""distill-kura — a distilled long-term memory (蔵 / *kura*) for agents.

    from distill_kura import Registry, recall
    reg = Registry.load("kura.toml")
    print(recall(reg.store("eq"), reg.models.thinker, "what did we decide about X?")["context"])
"""
from .recall import recall
from .registry import Registry
from .store import Store
from .thinker import Endpoint, Models

__version__ = "0.1.0"
__all__ = ["Registry", "Store", "Endpoint", "Models", "recall", "__version__"]
