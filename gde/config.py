import os
import sys
import json

from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

PROJ_ROOT = Path(Path(__file__).resolve().parents[1])

DATASET_DIR = Path(os.environ.get("DATASET_DIR", PROJ_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", PROJ_ROOT / "result"))
EMBEDDING_DIR = Path(os.environ.get("EMBEDDING_DIR", PROJ_ROOT / "embedding"))
CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINT_DIR", PROJ_ROOT / "checkpoint"))


def _format_message(record):
    # process record to generate
    t = record["time"]
    level = record["level"]

    name = record["name"]
    function = record["function"]
    line = record["line"]

    message = record["message"]

    # escape tags in message, tags are "<some_tag>"
    if function.startswith("<") and function.endswith(">"):
        function = rf"\{function}"

    message = message.replace("{", "{{").replace("}", "}}")

    # strip the path to the project root and replace it with a dot for brevity in both file and message
    message = message.replace(str(PROJ_ROOT), ".")

    final = (
        f"<green>{t:YYYY-MM-DD HH:mm:ss.SSS!UTC}</green> | "
        f"<level>{level: <8}</level> | "
        f"<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    if extra := record["extra"]:
        extra = json.dumps(extra)
        extra = extra.replace("{", "{{").replace("}", "}}").replace('"', "")
        final += f" | <level>{extra}</level>\n"
    else:
        final += "\n"

    return final


logger.remove(0)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=_format_message,
        colorize=True,
        level=LOG_LEVEL,
    )
except ModuleNotFoundError:
    logger.add(sys.stderr, format=_format_message, colorize=True, level=LOG_LEVEL)

logger.info(f"LOG_LEVEL is: {LOG_LEVEL}")

# Paths
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

logger.info("Dataset directory: {}", DATASET_DIR)
logger.info("Results directory: {}", RESULTS_DIR)
logger.info("Embedding directory: {}", EMBEDDING_DIR)
logger.info("Checkpoint directory: {}", CHECKPOINTS_DIR)
