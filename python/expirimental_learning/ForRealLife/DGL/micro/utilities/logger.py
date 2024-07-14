from enum import Enum

from DGL.micro import Settings, LogLevel


    ############
    ## LOGGER ##
    ############


ANSI_COLORS = {
    -1: "\033[90m",  # VERBOSE - Gray
    0: "\033[94m",   # DEBUG - Blue
    1: "\033[92m",   # INFO - Green
    2: "\033[96m",   # ALERT - Cyan
    3: "\033[93m",   # WARNING - Yellow
    4: "\033[91m",   # ERROR - Red
    5: "\033[95m"    # FATAL - Magenta
}

EXIT_CHAR = "\033[0m"



class Log:
    def __init__(self, level, sector, message):
        if level.value >= Settings.DEBUG_LEVEL.value.value:
            print(f"{ANSI_COLORS[level.value]}[{level}] :: {sector} :: {EXIT_CHAR} {message}")

        if level == LogLevel.FATAL:
            exit(1)
