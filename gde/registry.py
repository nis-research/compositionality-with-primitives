import types

from gde.config import logger

VALID_TYPES = frozenset({"factorizer"})


class _Registry:
    def __init__(self):
        self._store: dict[str, dict[str, type]] = {t: {} for t in VALID_TYPES}

    def register(self, type_: str, name: str):
        if type_ not in VALID_TYPES:
            raise ValueError(
                f"Unknown registry type '{type_}'. Expected one of {sorted(VALID_TYPES)}."
            )

        def decorator(cls):
            if name in self._store[type_]:
                raise KeyError(f"'{name}' is already registered under '{type_}'.")
            self._store[type_][name] = cls
            logger.debug("Registered {} '{}': {}", type_, name, cls.__qualname__)
            return cls

        return decorator

    def __getitem__(self, key):
        if isinstance(key, tuple):
            type_, name = key
            return self._store[type_][name]
        if key not in VALID_TYPES:
            raise KeyError(
                f"Unknown type '{key}'. Expected one of {sorted(VALID_TYPES)}."
            )
        return types.MappingProxyType(self._store[key])

    def __setitem__(self, key, value):
        raise TypeError("Use @registry.register() to add entries.")

    def __contains__(self, key):
        if isinstance(key, tuple):
            type_, name = key
            return name in self._store.get(type_, {})
        return key in VALID_TYPES

    def __repr__(self):
        parts = [f"  [{t}] {n}" for t, d in self._store.items() for n in d]
        return "Registry(\n" + "\n".join(parts) + "\n)" if parts else "Registry(empty)"


registry = _Registry()
