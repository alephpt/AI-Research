import DGL
from DGL import Log
from DGL.cosmos import Settings
from DGL.engine.learning.network import testDRL
from DGL.engine.learning.targeting.targetingsystem import testTargetingSystem
from DGL.society.unit import testEthicsMatrix

if __name__ == '__main__':
    Log(DGL.LogLevel.INFO, "main", "Generating a Deep Genetic Society!")

    # Unit Tests
    # Settings.UnitTest()
    # testEthicsMatrix()
    # testTargetingSystem()
    # testDRL()

    DGL.World().run()

    Log(DGL.LogLevel.INFO, "main", "Exiting..")