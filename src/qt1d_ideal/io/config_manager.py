"""
Configuration file parser for quantum tunneling simulations.

Handles loading and parsing of configuration files with support for:
- Key-value pairs (key = value)
- Inline comments (# comment)
- Boolean values (true/false)
- Numeric values (int and float)
- String values
"""

from pathlib import Path


class ConfigManager:
    """
    Parse configuration files for quantum tunneling simulations.
    
    File Format:
        # Comment lines start with #
        key = value          # Inline comments supported
        
    Supported Types:
        - Boolean: true, false (case-insensitive)
        - Integer: 2048, -5
        - Float: 2.0, -10.5, 0.001
        - String: anything else
    """
    
    @staticmethod
    def load(config_path: str) -> dict:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        
        Returns:
            Dictionary of configuration parameters
        
        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        config = {}
        
        with open(path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Strip whitespace
                line = line.strip()
                
                # Skip empty lines and comment-only lines
                if not line or line.startswith('#'):
                    continue
                
                # Check for key=value format
                if '=' not in line:
                    continue
                
                # Split on first '=' only
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove inline comments
                # Handle comments after the value
                if '#' in value:
                    value = value.split('#')[0].strip()
                
                # Parse value type
                value = ConfigManager._parse_value(value)
                
                config[key] = value
        
        return config
    
    @staticmethod
    def _parse_value(value: str):
        """
        Parse string value to appropriate Python type.
        
        Args:
            value: String value to parse
        
        Returns:
            Parsed value (bool, int, float, or str)
        """
        # Handle boolean values
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Try to parse as number
        try:
            # Check if it's a float (has decimal point or scientific notation)
            if '.' in value or 'e' in value.lower():
                return float(value)
            else:
                # Try integer
                return int(value)
        except ValueError:
            # Return as string if parsing fails
            return value
