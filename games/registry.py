"""Central game registry with decorator-based registration"""
from typing import Dict, Any, Callable, Type
from importlib import import_module

class GameRegistration:
    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        loader_path: str,
        simulator_path: str,
        display_name: str
    ) -> Callable[[Type], Type]:
        """Decorator factory for game registration"""
        def decorator(cls: Type) -> Type:
            self._registry[name] = {
                "loader_path": loader_path,
                "simulator_path": simulator_path,
                "display_name": display_name,
                "config_class": cls
            }
            return cls
        return decorator

    def get_display_name(self, name: str) -> str:
        """Get human-readable display name for a game"""
        if name not in self._registry:
            raise ValueError(f"Game '{name}' not registered")
        return self._registry[name]["display_name"]

    def get_game_loader(self, name: str) -> Callable:
        """Get registered game loader"""
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise ValueError(f"Game '{name}' not found. Available games: {available}")

        module_path, func_name = self._registry[name]["loader_path"].rsplit(".", 1)
        return getattr(import_module(module_path), func_name)

    def get_simulator_class(self, name: str) -> Type:
        """Get registered simulator class"""
        module_path, class_name = self._registry[name]["simulator_path"].rsplit(".", 1)
        return getattr(import_module(module_path), class_name)

# Singleton registry instance
registry = GameRegistration()