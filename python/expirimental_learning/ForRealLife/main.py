import DGL
from DGL import Log
from DGL.cosmos import Settings
from DGL.engine.network.network import testDRL

if __name__ == '__main__':
    Log(DGL.LogLevel.INFO, "main", "Generating a Deep Genetic Society!")
    
    #DGL.World().run()
    #Settings.UnitTest()
    testDRL()

    Log(DGL.LogLevel.INFO, "main", "Exiting..")