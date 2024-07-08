from DGL.micro import Settings, LogLevel


    ############
    ## LOGGER ##
    ############

class Log:
    def __init__(self, level, message):
        if level == LogLevel.CRITICAL:
            print(f" [{level}] :: {message}")
            exit(1)

        if level.value >= Settings.DEBUG_LEVEL.value.value:
            print(f" [{level}] :: {message}")