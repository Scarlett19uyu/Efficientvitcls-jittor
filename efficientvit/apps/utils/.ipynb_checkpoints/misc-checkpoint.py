import os
import yaml
from typing import Optional, Dict, Any, Union

__all__ = [
    "parse_with_yaml",
    "parse_unknown_args",
    "partial_update_config",
    "resolve_and_load_config",
    "load_config",
    "dump_config",
]

def parse_with_yaml(config_str: str) -> Union[str, Dict]:
    """Safely parse a string that may contain YAML data."""
    if not isinstance(config_str, str):
        return config_str
        
    try:
        # Handle special cases for dictionary-like strings
        stripped = config_str.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            config_str = stripped.replace(":", ": ")
        return yaml.safe_load(config_str)
    except (yaml.YAMLError, ValueError) as e:
        # Return raw string if parsing fails
        return config_str

def parse_unknown_args(unknown: list) -> Dict[str, Any]:
    """Parse unknown command line arguments into a nested dictionary."""
    parsed_dict = {}
    index = 0
    while index < len(unknown):
        key = unknown[index]
        if not key.startswith("--"):
            index += 1
            continue
            
        key = key[2:]  # Remove '--'
        if index + 1 >= len(unknown):
            parsed_dict[key] = None
            break
            
        val = unknown[index + 1]
        index += 2

        # Handle nested keys with dot notation
        if "." in key:
            keys = key.split(".")
            current = parsed_dict
            for k in keys[:-1]:
                if k not in current or not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = parse_with_yaml(val)
        else:
            parsed_dict[key] = parse_with_yaml(val)
            
    return parsed_dict

def partial_update_config(config: Dict, partial_config: Optional[Dict]) -> Dict:
    """Recursively update a config dictionary with partial config."""
    if partial_config is None:
        return config
        
    if not isinstance(partial_config, dict):
        raise TypeError(f"partial_config must be dict, got {type(partial_config)}")
        
    for key, value in partial_config.items():
        if (key in config and isinstance(config[key], dict) 
            and isinstance(value, dict)):
            partial_update_config(config[key], value)
        else:
            config[key] = value
    return config

def resolve_and_load_config(path: str, config_name: str = "config.yaml") -> Dict:
    """Resolve a config path (file or directory) and load the config."""
    path = os.path.realpath(os.path.expanduser(path))
    
    if os.path.isdir(path):
        config_path = os.path.join(path, config_name)
    else:
        config_path = path
        
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    return load_config(config_path)

class SafeLoaderWithTuple(yaml.SafeLoader):
    """YAML SafeLoader with support for Python tuples."""
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

SafeLoaderWithTuple.add_constructor(
    "tag:yaml.org,2002:python/tuple", 
    SafeLoaderWithTuple.construct_python_tuple
)

def load_config(filename: str) -> Dict:
    """Load a YAML config file with enhanced error handling."""
    filename = os.path.realpath(os.path.expanduser(filename))
    
    try:
        with open(filename, "r") as f:
            config = yaml.load(f, Loader=SafeLoaderWithTuple)
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parsing error in {filename}: {str(e)}")
    except IOError as e:
        raise IOError(f"Failed to read config file {filename}: {str(e)}")
    
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a dictionary, got {type(config)}")
        
    return config

def dump_config(config: Dict, filename: str) -> None:
    """Dump a config dictionary to a YAML file."""
    filename = os.path.realpath(os.path.expanduser(filename))
    
    try:
        with open(filename, "w") as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=False)
    except IOError as e:
        raise IOError(f"Failed to write config file {filename}: {str(e)}")