# In registry.py

from typing import Dict, Any, Callable, Type, Optional, List
from importlib import import_module
from games.game_specs import ENV_SPECS

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
        def decorator(cls: Type) -> Type:
            if name in self._registry:
                raise ValueError(f"Game '{name}' is already registered.")
            self._registry[name] = {
                "module_path": module_path,
                "class_name": class_name,
                "simulator_path": simulator_path,
                "display_name": display_name,
                "config_class": cls
            }
            return cls
        return decorator

    def get_game_loader(self, name: str) -> Callable:
        # Lazy-load the game module if not already registered.
        if name not in self._registry:
            if name in ENV_SPECS:
                import_module(ENV_SPECS[name])
            if name not in self._registry:
                available = ", ".join(self._registry.keys())
                raise ValueError(f"Game '{name}' not found. Available games: {available}")
        module_path = self._registry[name]["module_path"]
        class_name = self._registry[name]["class_name"]
        method_name = "load"  # Assumes every loader has a static load() method.
        cls = getattr(import_module(module_path), class_name)
        return getattr(cls, method_name)

    def get_simulator_instance(self,
                               game_name: str,
                               game: Any,
                               player_types: List[str],
                               max_game_rounds: Optional[int] = None,
                               seed: Optional[int] = None) -> Any:
        if game_name not in self._registry:
            if game_name in ENV_SPECS:
                import_module(ENV_SPECS[game_name])
            if game_name not in self._registry:
                available = ", ".join(self._registry.keys())
                raise ValueError(f"Game '{game_name}' not found. Available games: {available}")
        module_path, class_name = self._registry[game_name]["simulator_path"].rsplit(".", 1)
        simulator_class = getattr(import_module(module_path), class_name)
        return simulator_class(game, game_name, player_types, max_game_rounds, seed)

    def make_env(self, game_name: str, config: Dict[str, Any]) -> Any:
        """
        Creates an environment instance for the given game.
        This function is analogous to gym.make().

        Args:
            game_name: The internal game name.
            config: The simulation configuration (must include keys such as 'env_config', 'agents', and optionally 'seed').

        Returns:
            An initialized environment simulator instance.
        """
        # Get player types from agent configuration.
        player_types = [agent["type"] for _, agent in sorted(config.get("agents", {}).items())]
        max_game_rounds = config["env_config"].get("max_game_rounds")
        seed = config.get("seed", 42)

        # Call the static load() method.
        loader = self.get_game_loader(game_name)()

        # Retrieve the simulator instance.
        env = self.get_simulator_instance(
            game_name=game_name,
            game=loader,
            player_types=player_types,  #TODO: see if this is still needed
            max_game_rounds=max_game_rounds,
            seed=seed
        )
        return env

# Singleton registry instance.
registry = GameRegistration()
