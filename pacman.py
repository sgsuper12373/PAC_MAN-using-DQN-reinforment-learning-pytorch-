# pacman.py
# Modernized Python-3 version (merged parts). Keeps public API and behavior.
# Comments concise; no license header as requested.

from game import GameStateData, Game, Directions, Actions
from util import nearestPoint, manhattanDistance
import util
import layout
import sys
import time
import random
import types
import os

import pacmanDQN_Agents
import ghostAgents
# import ghostAgent

###################################################
# GameState Class
###################################################

class GameState:
    """
    A GameState stores the full game snapshot: food, capsules,
    agent configurations and score.
    """

    explored = set()   # static: track which states called getLegalActions

    @staticmethod
    def getAndResetExplored():
        tmp = GameState.explored.copy()
        GameState.explored = set()
        return tmp

    def __init__(self, prevState=None):
        if prevState is not None:
            self.data = GameStateData(prevState.data)
        else:
            self.data = GameStateData()

    def getLegalActions(self, agentIndex=0):
        if self.isWin() or self.isLose():
            return []
        if agentIndex == 0:
            return PacmanRules.getLegalActions(self)
        else:
            return GhostRules.getLegalActions(self, agentIndex)

    def generateSuccessor(self, agentIndex, action):
        if self.isWin() or self.isLose():
            raise Exception("Cannot generate successor of terminal state.")

        state = GameState(self)

        if agentIndex == 0:
            state.data._eaten = [False for _ in range(state.getNumAgents())]
            PacmanRules.applyAction(state, action)
        else:
            GhostRules.applyAction(state, action, agentIndex)

        # Time passes
        if agentIndex == 0:
            state.data.scoreChange += -TIME_PENALTY
        else:
            GhostRules.decrementTimer(state.data.agentStates[agentIndex])

        GhostRules.checkDeath(state, agentIndex)

        state.data._agentMoved = agentIndex
        state.data.score += state.data.scoreChange

        GameState.explored.add(self)
        GameState.explored.add(state)
        return state

    def getLegalPacmanActions(self):
        return self.getLegalActions(0)

    def generatePacmanSuccessor(self, action):
        return self.generateSuccessor(0, action)

    def getPacmanState(self):
        return self.data.agentStates[0].copy()

    def getPacmanPosition(self):
        return self.data.agentStates[0].getPosition()

    def getGhostStates(self):
        return self.data.agentStates[1:]

    def getGhostState(self, agentIndex):
        if agentIndex == 0 or agentIndex >= self.getNumAgents():
            raise Exception("Invalid ghost index.")
        return self.data.agentStates[agentIndex]

    def getGhostPosition(self, agentIndex):
        if agentIndex == 0:
            raise Exception("Pacman is not a ghost.")
        return self.data.agentStates[agentIndex].getPosition()

    def getGhostPositions(self):
        return [g.getPosition() for g in self.getGhostStates()]

    def getNumAgents(self):
        return len(self.data.agentStates)

    def getScore(self):
        return float(self.data.score)

    def getCapsules(self):
        return self.data.capsules

    def getNumFood(self):
        return self.data.food.count()

    def getFood(self):
        return self.data.food

    def getWalls(self):
        return self.data.layout.walls

    def hasFood(self, x, y):
        return self.data.food[x][y]

    def hasWall(self, x, y):
        return self.data.layout.walls[x][y]

    def isLose(self):
        return self.data._lose

    def isWin(self):
        return self.data._win

    def deepCopy(self):
        state = GameState(self)
        state.data = self.data.deepCopy()
        return state

    def __eq__(self, other):
        return hasattr(other, 'data') and self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def __str__(self):
        return str(self.data)

    def initialize(self, layout, numGhostAgents=1000):
        self.data.initialize(layout, numGhostAgents)

###################################################
# Hidden constants used by rules
###################################################

SCARED_TIME = 40
COLLISION_TOLERANCE = 0.7
TIME_PENALTY = 1

###################################################
# ClassicGameRules — high-level game control
###################################################

class ClassicGameRules:
    def __init__(self, timeout=30):
        self.timeout = timeout

    def newGame(self, layoutObj, pacmanAgent, ghostAgentsList, display,
                quiet=False, catchExceptions=False):

        agents = [pacmanAgent] + ghostAgentsList[:layoutObj.getNumGhosts()]
        initState = GameState()
        initState.initialize(layoutObj, len(ghostAgentsList))
        game = Game(agents, display, self, catchExceptions=catchExceptions)

        game.state = initState
        self.initialState = initState.deepCopy()
        self.quiet = quiet

        return game

    def process(self, state, game):
        if state.isWin(): self.win(state, game)
        if state.isLose(): self.lose(state, game)

    def win(self, state, game):
        if not self.quiet:
            print(f"Pacman emerges victorious! Score: {state.data.score}")
        game.gameOver = True

    def lose(self, state, game):
        if not self.quiet:
            print(f"Pacman died! Score: {state.data.score}")
        game.gameOver = True

    def getProgress(self, game):
        return float(game.state.getNumFood()) / self.initialState.getNumFood()

    def agentCrash(self, game, agentIndex):
        if agentIndex == 0:
            print("Pacman crashed")
        else:
            print("A ghost crashed")

    # time limits
    def getMaxTotalTime(self, agentIndex): return self.timeout
    def getMaxStartupTime(self, agentIndex): return self.timeout
    def getMoveWarningTime(self, agentIndex): return self.timeout
    def getMoveTimeout(self, agentIndex): return self.timeout
    def getMaxTimeWarnings(self, agentIndex): return 0

###################################################
# Pacman Rules
###################################################

class PacmanRules:
    PACMAN_SPEED = 1

    @staticmethod
    def getLegalActions(state):
        return Actions.getPossibleActions(
            state.getPacmanState().configuration,
            state.data.layout.walls
        )

    @staticmethod
    def applyAction(state, action):
        legal = PacmanRules.getLegalActions(state)
        if action not in legal:
            raise Exception("Illegal pacman action: " + str(action))

        pac = state.data.agentStates[0]
        vector = Actions.directionToVector(action, PacmanRules.PACMAN_SPEED)
        pac.configuration = pac.configuration.generateSuccessor(vector)

        # Eat food/capsule if close enough to a grid point
        pos = pac.configuration.getPosition()
        nearest = nearestPoint(pos)
        if manhattanDistance(nearest, pos) <= 0.5:
            PacmanRules.consume(nearest, state)

    @staticmethod
    def consume(position, state):
        x, y = position

        # Eat food
        if state.data.food[x][y]:
            state.data.scoreChange += 10
            state.data.food = state.data.food.copy()
            state.data.food[x][y] = False
            state.data._foodEaten = position

            if state.getNumFood() == 0 and not state.data._lose:
                state.data.scoreChange += 500
                state.data._win = True

        # Eat capsule
        if position in state.getCapsules():
            state.data.capsules.remove(position)
            state.data._capsuleEaten = position
            for g in state.data.agentStates[1:]:
                g.scaredTimer = SCARED_TIME

###################################################
# Ghost Rules
###################################################

class GhostRules:
    GHOST_SPEED = 1.0

    @staticmethod
    def getLegalActions(state, ghostIndex):
        conf = state.getGhostState(ghostIndex).configuration
        possible = Actions.getPossibleActions(conf, state.data.layout.walls)

        reverse = Actions.reverseDirection(conf.direction)
        if Directions.STOP in possible:
            possible.remove(Directions.STOP)
        if reverse in possible and len(possible) > 1:
            possible.remove(reverse)

        return possible

    @staticmethod
    def applyAction(state, action, ghostIndex):
        legal = GhostRules.getLegalActions(state, ghostIndex)
        if action not in legal:
            raise Exception("Illegal ghost action: " + str(action))

        ghostState = state.data.agentStates[ghostIndex]
        speed = GhostRules.GHOST_SPEED / 2.0 if ghostState.scaredTimer > 0 else GhostRules.GHOST_SPEED
        vector = Actions.directionToVector(action, speed)
        ghostState.configuration = ghostState.configuration.generateSuccessor(vector)

    @staticmethod
    def decrementTimer(ghostState):
        timer = ghostState.scaredTimer
        if timer == 1:
            ghostState.configuration.pos = nearestPoint(ghostState.configuration.pos)
        ghostState.scaredTimer = max(0, timer - 1)

    @staticmethod
    def checkDeath(state, agentIndex):
        pacpos = state.getPacmanPosition()
        if agentIndex == 0:
            # pacman moved → ghosts may kill pacman
            for i, ghost in enumerate(state.data.agentStates[1:], start=1):
                if GhostRules.canKill(pacpos, ghost.configuration.getPosition()):
                    GhostRules.collide(state, ghost, i)
        else:
            ghost = state.data.agentStates[agentIndex]
            if GhostRules.canKill(pacpos, ghost.configuration.getPosition()):
                GhostRules.collide(state, ghost, agentIndex)

    @staticmethod
    def canKill(pacpos, ghostpos):
        return manhattanDistance(ghostpos, pacpos) <= COLLISION_TOLERANCE

    @staticmethod
    def collide(state, ghostState, agentIndex):
        if ghostState.scaredTimer > 0:
            state.data.scoreChange += 200
            GhostRules.placeGhost(state, ghostState)
            ghostState.scaredTimer = 0
            state.data._eaten[agentIndex] = True
        else:
            if not state.data._win:
                state.data.scoreChange -= 500
                state.data._lose = True

    @staticmethod
    def placeGhost(state, ghostState):
        ghostState.configuration = ghostState.start

# ---------------------------
# Utilities for CLI & agent loading
# ---------------------------

def default(s):
    return s + ' [Default: %default]'

def parseAgentArgs(s):
    if s is None:
        return {}
    pieces = s.split(',')
    opts = {}
    for p in pieces:
        if '=' in p:
            k, v = p.split('=', 1)
        else:
            k, v = p, '1'
        opts[k] = v
    return opts

def readCommand(argv):
    """
    Parse command-line args and return args dict for runGames/replay.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python pacman.py <options>
    EXAMPLES:   (1) python pacman.py
                    - starts an interactive game
                (2) python pacman.py --layout smallClassic --zoom 2
    """
    parser = OptionParser(usageStr)

    parser.add_option('-n', '--numGames', dest='numGames', type='int',
                      help=default('the number of GAMES to play'), metavar='GAMES', default=6000)
    parser.add_option('-l', '--layout', dest='layout',
                      help=default('the LAYOUT_FILE from which to load the map layout'),
                      metavar='LAYOUT_FILE', default='smallGrid')
    parser.add_option('-p', '--pacman', dest='pacman',
                      help=default('the agent TYPE in the pacmanAgents module to use'),
                      metavar='TYPE', default='PacmanDQN')
    parser.add_option('-t', '--textGraphics', action='store_true', dest='textGraphics',
                      help='Display output as text only', default=False)
    parser.add_option('-q', '--quietTextGraphics', action='store_true', dest='quietGraphics',
                      help='Generate minimal output and no graphics', default=False)
    parser.add_option('-g', '--ghosts', dest='ghost',
                      help=default('the ghost agent TYPE in the ghostAgents module to use'),
                      metavar='TYPE', default='RandomGhost')
    parser.add_option('-k', '--numghosts', type='int', dest='numGhosts',
                      help=default('The maximum number of ghosts to use'), default=4)
    parser.add_option('-z', '--zoom', type='float', dest='zoom',
                      help=default('Zoom the size of the graphics window'), default=1.0)
    parser.add_option('-f', '--fixRandomSeed', action='store_true', dest='fixRandomSeed',
                      help='Fixes the random seed to always play the same game', default=False)
    parser.add_option('-r', '--recordActions', action='store_true', dest='record',
                      help='Writes game histories to a file (named by the time they were played)', default=False)
    parser.add_option('--replay', dest='gameToReplay',
                      help='A recorded game file (pickle) to replay', default=None)
    parser.add_option('-a', '--agentArgs', dest='agentArgs',
                      help='Comma separated values sent to agent. e.g. "opt1=val1,opt2,opt3=val3"')
    parser.add_option('-x', '--numTraining', dest='numTraining', type='int',
                      help=default('How many episodes are training (suppresses output)'), default=5000)
    parser.add_option('--frameTime', dest='frameTime', type='float',
                      help=default('Time to delay between frames; <0 means keyboard'), default=0.1)
    parser.add_option('-c', '--catchExceptions', action='store_true', dest='catchExceptions',
                      help='Turns on exception handling and timeouts during games', default=False)
    parser.add_option('--timeout', dest='timeout', type='int',
                      help=default('Maximum length of time an agent can spend computing in a single game'), default=30)

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Fix the random seed
    if options.fixRandomSeed:
        random.seed('cs188')

    # Choose a layout
    args['layout'] = layout.getLayout(options.layout)
    if args['layout'] is None:
        raise Exception("The layout " + options.layout + " cannot be found")

    # Choose a Pacman agent
    noKeyboard = options.gameToReplay is None and (options.textGraphics or options.quietGraphics)
    pacmanType = loadAgent(options.pacman, noKeyboard)
    agentOpts = parseAgentArgs(options.agentArgs) if options.agentArgs else {}

    # Provide width/height to agent options for convenience
    agentOpts['width'] = layout.getLayout(options.layout).width
    agentOpts['height'] = layout.getLayout(options.layout).height

    if options.numTraining > 0:
        args['numTraining'] = options.numTraining
        if 'numTraining' not in agentOpts:
            agentOpts['numTraining'] = options.numTraining

    # Instantiate the pacman agent. Some agents (like DQN) accept an
    # options dict while others (like KeyboardAgent) expect an index or no
    # arguments. Try keyword-argument construction first and fall back to
    # constructors that accept a single argument or no arguments.
    try:
        if isinstance(agentOpts, dict) and agentOpts:
            pacman = pacmanType(**agentOpts)
        else:
            pacman = pacmanType()
    except TypeError:
        # Fall back: pass the dict as a single positional argument if the
        # agent expects it, or construct without args.
        try:
            pacman = pacmanType(agentOpts) if isinstance(agentOpts, dict) else pacmanType()
        except Exception:
            pacman = pacmanType()
    args['pacmanAgent'] = pacman
    # Some legacy agents expect an integer index as their first parameter
    # but may have been constructed with the whole agentOpts dict by
    # mistake; protect against that by coercing a dict index to 0.
    if hasattr(pacman, 'index') and isinstance(pacman.index, dict):
        pacman.index = 0
    pacman.width = agentOpts.get('width', pacman.width if hasattr(pacman, 'width') else None)
    pacman.height = agentOpts.get('height', pacman.height if hasattr(pacman, 'height') else None)

    # Choose ghost agent
    ghostType = loadAgent(options.ghost, noKeyboard)
    args['ghosts'] = [ghostType(i + 1) for i in range(options.numGhosts)]

    # Choose display
    if options.quietGraphics:
        import textDisplay
        args['display'] = textDisplay.NullGraphics()
    elif options.textGraphics:
        import textDisplay
        textDisplay.SLEEP_TIME = options.frameTime
        args['display'] = textDisplay.PacmanGraphics()
    else:
        import graphicsDisplay
        args['display'] = graphicsDisplay.PacmanGraphics(options.zoom, frameTime=options.frameTime)

    args['numGames'] = options.numGames
    args['record'] = options.record
    args['catchExceptions'] = options.catchExceptions
    args['timeout'] = options.timeout

    # Replay handling
    if options.gameToReplay is not None:
        print(f"Replaying recorded game {options.gameToReplay}.")
        import pickle
        with open(options.gameToReplay, 'rb') as f:
            recorded = pickle.load(f)
        recorded['display'] = args['display']
        replayGame(**recorded)
        sys.exit(0)

    return args

def loadAgent(agentName, nographics):
    """
    Find and return the agent class with name `agentName` from modules
    in the python path ending with 'gents.py' (e.g., pacmanAgents.py).
    """
    pythonPathStr = os.path.expandvars("$PYTHONPATH")
    if ';' in pythonPathStr:
        pythonPathDirs = pythonPathStr.split(';')
    else:
        pythonPathDirs = pythonPathStr.split(':')
    pythonPathDirs.append('.')

    for moduleDir in pythonPathDirs:
        if not os.path.isdir(moduleDir):
            continue
        moduleNames = [f for f in os.listdir(moduleDir) if f.endswith('gents.py')]
        for modulename in moduleNames:
            try:
                module = __import__(modulename[:-3])
            except ImportError:
                continue
            if agentName in dir(module):
                if nographics and modulename == 'keyboardAgents.py':
                    raise Exception('Using the keyboard requires graphics (not text display)')
                return getattr(module, agentName)
    raise Exception('The agent ' + agentName + ' is not specified in any *Agents.py.')

def replayGame(layout, actions, display):
    import pacmanAgents
    import ghostAgents
    rules = ClassicGameRules()
    agents = [pacmanAgents.GreedyAgent()] + [ghostAgents.RandomGhost(i + 1) for i in range(layout.getNumGhosts())]
    game = rules.newGame(layout, agents[0], agents[1:], display)
    state = game.state
    display.initialize(state.data)

    for action in actions:
        state = state.generateSuccessor(*action)
        display.update(state.data)
        rules.process(state, game)

    display.finish()

def runGames(layout, pacmanAgent, ghosts, display, numGames, record, numTraining=0, catchExceptions=False, timeout=30):
    import __main__
    __main__.__dict__['_display'] = display

    rules = ClassicGameRules(timeout)
    games = []

    for i in range(numGames):
        beQuiet = i < numTraining
        if beQuiet:
            import textDisplay
            gameDisplay = textDisplay.NullGraphics()
            rules.quiet = True
        else:
            gameDisplay = display
            rules.quiet = False

        game = rules.newGame(layout, pacmanAgent, ghosts, gameDisplay, beQuiet, catchExceptions)
        game.run()

        if not beQuiet:
            games.append(game)

        if record:
            import time, pickle
            fname = f"recorded-game-{i+1}-" + "-".join(str(t) for t in time.localtime()[1:6])
            with open(fname, 'wb') as f:
                components = {'layout': layout, 'actions': game.moveHistory}
                pickle.dump(components, f)

    if (numGames - numTraining) > 0:
        scores = [game.state.getScore() for game in games]
        wins = [game.state.isWin() for game in games]
        winRate = wins.count(True) / float(len(wins))
        print(('Average Score:', sum(scores) / float(len(scores))))
        print(('Scores:       ', ', '.join([str(score) for score in scores])))
        print(('Win Rate:      %d/%d (%.2f)' % (wins.count(True), len(wins), winRate)))
        print(('Record:       ', ', '.join([['Loss', 'Win'][int(w)] for w in wins])))

    return games

if __name__ == '__main__':
    args = readCommand(sys.argv[1:])
    runGames(**args)
