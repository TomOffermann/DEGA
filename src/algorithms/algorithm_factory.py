from typing import Type, Dict
from algorithms.algorithm import Algorithm

class AlgorithmFactory:
    """A registry for all Algorithm implementations."""
    _registry: Dict[str, Type[Algorithm]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Use as a decorator to register new algorithms.
        e.g.:

            @AlgorithmFactory.register('dega_a')
            class DEGA_A(Algorithm):
                ...
        """
        def decorator(alg_cls: Type[Algorithm]):
            if name in cls._registry:
                raise KeyError(f"Algorithm '{name}' already registered")
            cls._registry[name] = alg_cls
            return alg_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> Algorithm:
        """
        Instantiate an algorithm by its registry name.
        """
        try:
            alg_cls = cls._registry[name]
        except KeyError:
            raise ValueError(f"No algorithm registered under '{name}', "
                             f"available: {list(cls._registry)}")
        return alg_cls(**kwargs)

    @classmethod
    def available(cls):
        """List all registered algorithm names."""
        return list(cls._registry.keys())
