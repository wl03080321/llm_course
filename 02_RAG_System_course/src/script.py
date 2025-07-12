from typing import List, Dict, Optional, Any
import yaml
from pathlib import Path

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load LLM configuration from file.
    
    Args:
        config_path: Path to the configuration file. If None, uses default path.
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
    """
    if config_path is None:
        config_path = str(Path(__file__).parent.parent.parent / "config" / "llm.yaml")
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config 