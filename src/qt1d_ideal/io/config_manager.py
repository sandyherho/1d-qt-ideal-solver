from pathlib import Path

class ConfigManager:
    @staticmethod
    def load(config_path: str):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        config = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key, value = key.strip(), value.strip()
                    try:
                        if value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                        elif '.' in value:
                            try:
                                value = float(value)
                            except ValueError:
                                pass
                        else:
                            try:
                                value = int(value)
                            except ValueError:
                                pass
                    except (ValueError, AttributeError):
                        pass
                    config[key] = value
        return config
