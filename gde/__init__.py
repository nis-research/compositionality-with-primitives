import importlib
import pkgutil

from gde.registry import registry, VALID_TYPES  # noqa: F401


def _autodiscover():
    for pkg_name in (f"gde.{_type}" for _type in VALID_TYPES):
        try:
            pkg = importlib.import_module(pkg_name)
        except ModuleNotFoundError:
            continue
        for _, module_name, _ in pkgutil.walk_packages(
            pkg.__path__, pkg.__name__ + "."
        ):
            importlib.import_module(module_name)


_autodiscover()
