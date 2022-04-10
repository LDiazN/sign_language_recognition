"""
    In this module you will find multiple configurations for the app. You can override them by specifying a different
    .env file, exporting environment variables properly named, or providing a settings.toml file.
    Remember that .env files should be in the project's root.
"""

from dynaconf import Dynaconf, Validator
import os

from pathlib import Path

_HOME = os.environ.get("HOME")

settings = Dynaconf(
    envvar_prefix="USB",
    settings_files=["./settings.toml", "config/.secrets.toml"],
    load_dotenv=True,
    validators=[
        # Root folder for local files, such as configs or dataset config files
        Validator("SLR_HOME", default=str(Path(_HOME, ".usb_slr"))) #type: ignore
    ]
)