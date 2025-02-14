"""
   registry.py
   Central game registry with decorator-based registration
   Dynamically register all games at runtime
"""
from typing import Dict, Any, Callable, Type, Optional
from importlib import import_module


class GameRegistration:
    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        module_path: str,
        class_name: str,
        simulator_path: str,
        display_name: str
    ) -> Callable[[Type], Type]:
        """
        Decorator factory for game registration.

        Args:
            name: The internal game name used for lookup.
            module_path: Path to the module containing the loader class (e.g., "games.loaders").
            class_name: The name of the loader class (e.g., "TicTacToeLoader").
            simulator_path: Path to the simulator class (e.g., "simulators.tic_tac_toe.TicTacToeSimulator").
            display_name: Human-readable name for the game.

        Returns:
            Callable decorator that adds the class to the registry.
        """
        def decorator(cls: Type) -> Type:
            if name in self._registry:
                raise ValueError(f"Game '{name}' is already registered.")
            self._registry[name] = {
                "module_path": module_path,
                "class_name": class_name,
                "simulator_path": simulator_path,
                "display_name": display_name,
                "config_class": cls # The class being decorated
            }
            return cls # Return the class unmodified
        return decorator

    def get_game_loader(self, name: str) -> Callable:
        """
        Get the game loader function for a registered game.

        Args:
            name: The internal name of the game.

        Returns:
            The game loader function.

        Raises:
            ValueError: If the game is not registered.
        """
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise ValueError(f"Game '{name}' not found. Available games: {available}")

        # Retrieve paths from the registry
        module_path = self._registry[name]["module_path"]
        class_name = self._registry[name]["class_name"]
        method_name = "load" # all loaders have a static load method

        # Import the module and get the class
        cls = getattr(import_module(module_path), class_name)

        # Retrieve and return the method
        return getattr(cls, method_name)


    def get_simulator_instance(self,
                                game_name,
                                game,
                                player_types,
                                max_game_rounds=None,
                                seed: Optional[int] = None) -> Any:
        """
        Get an initialized simulator instance for a registered game.

        Args:
            name: The internal name of the game.
            *args: Positional arguments for the simulator's constructor.
            **kwargs: Keyword arguments for the simulator's constructor.

        Returns:
            An initialized simulator instance.

        Raises:
            ValueError: If the game is not registered.
        """
        if game_name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise ValueError(f"Game '{game_name}' not found. Available games: {available}")

        module_path, class_name = self._registry[game_name]["simulator_path"].rsplit(".", 1)
        simulator_class = getattr(import_module(module_path), class_name)

        return simulator_class(game, game_name, player_types, max_game_rounds, seed)


# Singleton registry instance
registry = GameRegistration()
