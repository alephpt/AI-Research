import DGL
from DGL.micro import Log, LogLevel

if __name__ == '__main__':
    Log(LogLevel.INFO, "Generating a Deep Genetic Society!")
    DGL.Engine().run()