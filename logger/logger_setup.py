import json
import os.path
import pathlib
import logging.config
import sys
from datetime import datetime
from colorama import Fore, Style
from pathlib import Path

class ColoredFormatter(logging.Formatter):

    def format(self, record):
        # Obtener la fecha formateada de `asctime`
        record.asctime = self.formatTime(record, self.datefmt)

        # Colorear `asctime` en magenta
        coloredAsctime = f"{Fore.MAGENTA}{record.asctime}{Style.RESET_ALL}"

        # Colorear el resto de los campos (levelname, module, funcName) en azul y verde
        record.levelname = f"{Fore.BLUE}{record.levelname}{Style.RESET_ALL}"
        record.module = f"{Fore.GREEN}{record.module}{Style.RESET_ALL}"
        record.funcName = f"{Fore.GREEN}{record.funcName}{Style.RESET_ALL}"

        # Generar el mensaje final con los colores aplicados
        logMessage = super().format(record)

        # Reemplazar el asctime original con el coloreado
        return logMessage.replace(record.asctime, coloredAsctime)
    
def getProjectDirectory():
        return str(Path(__file__).resolve().parent)

def setup_logging(path=None):
    currentFilePath = pathlib.Path(__file__).resolve()
    sys.path.append(str(currentFilePath.parent))

    currentDay = datetime.now().strftime("%d-%m-%Y")
    logDirectory = getProjectDirectory() + "/logger/logs/" + currentDay \
        if path is None else path + currentDay
    if not os.path.exists(logDirectory):
        os.makedirs(logDirectory)

    timestamp = datetime.now().strftime("%H-%M-%S")
    logFilename = logDirectory + "/log_" + timestamp + ".txt"

    # Load the configuration
    config_file = pathlib.Path(getProjectDirectory() + "/logging_config.json")
    with open(config_file) as f_in:
        config = json.load(f_in)

    # Update the filename in the configuration
    config["handlers"]["file"]["filename"] = logFilename
    logging.config.dictConfig(config)

setup_logging()
