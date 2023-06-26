import logging
import os


def _setup_logger() -> "logging.Logger":
    from rich.console import Console
    from rich.logging import RichHandler

    logger = logging.getLogger(__name__)
    level = os.environ.get("LOGLEVEL", logging.INFO)
    logger.setLevel(level=level)
    console = Console(force_terminal=True)
    if console.is_jupyter is True:
        console.is_jupyter = False
    ch = RichHandler(show_path=False, console=console, show_time=logger.level == logging.DEBUG)
    logger.addHandler(ch)

    # this prevents double outputs
    logger.propagate = False
    return logger


logger = _setup_logger()
