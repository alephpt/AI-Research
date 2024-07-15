from enum import Enum

from ..settings import Settings, LogLevel


    ############
    ## LOGGER ##
    ############


ANSI_COLORS = {
    -4: "\033[90m",  # VERBOSE - Gray
    -3: "\033[94m",   # DEBUG - Blue
    -2: "\033[92m",   # INFO - Green
    -1: "\033[96m",   # ALERT - Cyan
    0: "",   # Release - None
    1: "\033[93m",   # WARNING - Yellow
    2: "\033[91m",   # ERROR - Red
    3: "\033[95m"    # FATAL - Magenta
}

EXIT_CHAR = "\033[0m"



class Log:
    def __init__(self, level, sector, message):
        if level.value >= Settings.DEBUG_LEVEL.value:
            #print(f"Checking Logging {level.value} >= Setting {Settings.DEBUG_LEVEL.value}")
            print(f"{ANSI_COLORS[level.value]}[{level}] :: {sector} :: {EXIT_CHAR} {message}")

        if level == LogLevel.FATAL:
            exit(1)
