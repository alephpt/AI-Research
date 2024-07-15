import DGL
from DGL import Log
from DGL.cosmos import Settings

if __name__ == '__main__':
    Log(DGL.LogLevel.INFO, "main", "Generating a Deep Genetic Society!")
    
    DGL.World().run()
    #Settings.UnitTest()

    Log(DGL.LogLevel.INFO, "main", "Exiting..")