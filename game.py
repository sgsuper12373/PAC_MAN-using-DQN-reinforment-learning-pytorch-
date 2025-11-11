from utils import *
import time
import traceback
import sys
import io

class Agent:
    """An agent must define a getAction method."""
    def __init__(self, index=0):
        self.index = index

    def getAction(self, state):
        """Return an action from Directions.{North, South, East, West, Stop}."""
        raiseNotDefined()


class Directions:
    NORTH, SOUTH, EAST, WEST, STOP = 'North', 'South', 'East', 'West', 'Stop'

    LEFT = {
        NORTH: WEST, SOUTH: EAST, EAST: NORTH, WEST: SOUTH, STOP: STOP
    }
    RIGHT = {y: x for x, y in LEFT.items()}
    REVERSE = {
        NORTH: SOUTH, SOUTH: NORTH, EAST: WEST, WEST: EAST, STOP: STOP
    }


class Configuration:
    """A Configuration holds the (x,y) coordinate of a character and direction."""
    def __init__(self, pos, direction):
        self.pos = pos
        self.direction = direction

    def getPosition(self):
        return self.pos

    def getDirection(self):
        return self.direction

    def isInteger(self):
        x, y = self.pos
        return x.is_integer() and y.is_integer()

    def __eq__(self, other):
        return isinstance(other, Configuration) and \
               self.pos == other.pos and self.direction == other.direction

    def __hash__(self):
        return hash((self.pos, self.direction))

    def __str__(self):
        return f"(x,y)={self.pos}, {self.direction}"

    def generateSuccessor(self, vector):
        """Generate new configuration reached by translating by a vector."""
        x, y = self.pos
        dx, dy = vector
        direction = Actions.vectorToDirection(vector)
        if direction == Directions.STOP:
            direction = self.direction
        return Configuration((x + dx, y + dy), direction)


class AgentState:
    """Holds configuration, type (Pacman/Ghost), scared timer, etc."""
    def __init__(self, startConfiguration, isPacman):
        self.start = startConfiguration
        self.configuration = startConfiguration
        self.isPacman = isPacman
        self.scaredTimer = 0
        self.numCarrying = 0
        self.numReturned = 0

    def __str__(self):
        role = "Pacman" if self.isPacman else "Ghost"
        return f"{role}: {self.configuration}"

    def __eq__(self, other):
        return isinstance(other, AgentState) and \
               self.configuration == other.configuration and \
               self.scaredTimer == other.scaredTimer

    def __hash__(self):
        return hash((self.configuration, self.scaredTimer))

    def copy(self):
        new_state = AgentState(self.start, self.isPacman)
        new_state.configuration = self.configuration
        new_state.scaredTimer = self.scaredTimer
        new_state.numCarrying = self.numCarrying
        new_state.numReturned = self.numReturned
        return new_state

    def getPosition(self):
        return None if self.configuration is None else self.configuration.getPosition()

    def getDirection(self):
        return self.configuration.getDirection() if self.configuration else None


class Grid:
    """2D boolean grid for walls, food, etc."""
    CELLS_PER_INT = 30

    def __init__(self, width, height, initialValue=False, bitRepresentation=None):
        if initialValue not in [False, True]:
            raise ValueError('Grids can only contain booleans')
        self.width = width
        self.height = height
        self.data = [[initialValue for _ in range(height)] for _ in range(width)]
        if bitRepresentation:
            self._unpackBits(bitRepresentation)

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __str__(self):
        rows = [''.join(str(self.data[x][y])[0] for x in range(self.width))
                for y in reversed(range(self.height))]
        return '\n'.join(rows)

    def __eq__(self, other):
        return isinstance(other, Grid) and self.data == other.data

    def __hash__(self):
        return hash(tuple(tuple(col) for col in self.data))

    def copy(self):
        g = Grid(self.width, self.height)
        g.data = [row[:] for row in self.data]
        return g

    def deepCopy(self):
        return self.copy()

    def shallowCopy(self):
        g = Grid(self.width, self.height)
        g.data = self.data
        return g

    def count(self, item=True):
        return sum(col.count(item) for col in self.data)

    def asList(self, key=True):
        return [(x, y) for x in range(self.width)
                for y in range(self.height) if self[x][y] == key]

    def packBits(self):
        bits = [self.width, self.height]
        current_int = 0
        for i in range(self.width * self.height):
            bit = self.CELLS_PER_INT - (i % self.CELLS_PER_INT) - 1
            x, y = divmod(i, self.height)
            if self[x][y]:
                current_int |= 1 << bit
            if (i + 1) % self.CELLS_PER_INT == 0:
                bits.append(current_int)
                current_int = 0
        bits.append(current_int)
        return tuple(bits)

    def _unpackBits(self, bits):
        cell = 0
        for packed in bits:
            for bit in self._unpackInt(packed, self.CELLS_PER_INT):
                if cell >= self.width * self.height:
                    return
                x, y = divmod(cell, self.height)
                self[x][y] = bit
                cell += 1

    @staticmethod
    def _unpackInt(packed, size):
        return [(packed >> (size - i - 1)) & 1 == 1 for i in range(size)]


def reconstituteGrid(bitRep):
    """Recreate Grid from bit-level tuple."""
    if not isinstance(bitRep, tuple):
        return bitRep
    width, height = bitRep[:2]
    return Grid(width, height, bitRepresentation=bitRep[2:])


####################################
# Parts you shouldn't have to read #
####################################


class Actions:
    """Collection of static methods for manipulating move actions."""
    TOLERANCE = 0.001
    _directions = {
        Directions.NORTH: (0, 1),
        Directions.SOUTH: (0, -1),
        Directions.EAST: (1, 0),
        Directions.WEST: (-1, 0),
        Directions.STOP: (0, 0),
    }
    _directionsAsList = list(_directions.items())

    @staticmethod
    def reverseDirection(action):
        return Directions.REVERSE.get(action, action)

    @staticmethod
    def vectorToDirection(vector):
        dx, dy = vector
        if dy > 0: return Directions.NORTH
        if dy < 0: return Directions.SOUTH
        if dx < 0: return Directions.WEST
        if dx > 0: return Directions.EAST
        return Directions.STOP

    @staticmethod
    def directionToVector(direction, speed=1.0):
        dx, dy = Actions._directions[direction]
        return dx * speed, dy * speed

    @staticmethod
    def getPossibleActions(config, walls):
        x, y = config.pos
        x_int, y_int = round(x), round(y)

        if abs(x - x_int) + abs(y - y_int) > Actions.TOLERANCE:
            return [config.getDirection()]

        possible = []
        for direction, (dx, dy) in Actions._directionsAsList:
            nx, ny = x_int + dx, y_int + dy
            if not walls[nx][ny]:
                possible.append(direction)
        return possible

    @staticmethod
    def getLegalNeighbors(position, walls):
        x, y = map(round, position)
        neighbors = []
        for _, (dx, dy) in Actions._directionsAsList:
            nx, ny = x + dx, y + dy
            if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                neighbors.append((nx, ny))
        return neighbors

    @staticmethod
    def getSuccessor(position, action):
        dx, dy = Actions.directionToVector(action)
        x, y = position
        return x + dx, y + dy


class GameStateData:
    """Stores game-specific state info."""
    def __init__(self, prevState=None):
        if prevState:
            self.food = prevState.food.shallowCopy()
            self.capsules = prevState.capsules[:]
            self.agentStates = [a.copy() for a in prevState.agentStates]
            self.layout = prevState.layout
            self._eaten = prevState._eaten
            self.score = prevState.score
        else:
            self.food = None
            self.capsules = []
            self.agentStates = []
            self.layout = None
            self.score = 0
            self._eaten = []

        self._foodEaten = self._foodAdded = self._capsuleEaten = None
        self._agentMoved = None
        self._lose = self._win = False
        self.scoreChange = 0

    def deepCopy(self):
        state = GameStateData(self)
        state.food = self.food.deepCopy()
        state.layout = self.layout.deepCopy()
        return state

    def __eq__(self, other):
        return (
            isinstance(other, GameStateData)
            and self.agentStates == other.agentStates
            and self.food == other.food
            and self.capsules == other.capsules
            and self.score == other.score
        )

    def __hash__(self):
        return hash((
            tuple(self.agentStates),
            self.food,
            tuple(self.capsules),
            self.score,
        ))

    def __str__(self):
        width, height = self.layout.width, self.layout.height
        grid = Grid(width, height)
        food, walls = reconstituteGrid(self.food), self.layout.walls
        for x in range(width):
            for y in range(height):
                grid[x][y] = '.' if food[x][y] else ('%' if walls[x][y] else ' ')
        for agent in self.agentStates:
            if agent and agent.configuration:
                x, y = map(int, nearestPoint(agent.configuration.pos))
                grid[x][y] = 'v' if agent.isPacman else 'G'
        for x, y in self.capsules:
            grid[x][y] = 'o'
        return f"{grid}\nScore: {self.score}\n"

    def initialize(self, layout, numGhostAgents):
        self.food = layout.food.copy()
        self.capsules = layout.capsules[:]
        self.layout = layout
        self.score = 0
        self.agentStates = []
        ghosts = 0
        for isPacman, pos in layout.agentPositions:
            if not isPacman:
                if ghosts == numGhostAgents:
                    continue
                ghosts += 1
            self.agentStates.append(AgentState(Configuration(pos, Directions.STOP), isPacman))
        self._eaten = [False] * len(self.agentStates)


try:
    import boinc
    _BOINC_ENABLED = True
except ImportError:
    _BOINC_ENABLED = False


class Game:
    """Main control flow for game play."""
    def __init__(self, agents, display, rules, startingIndex=0,
                 muteAgents=False, catchExceptions=False):
        self.agents = agents
        self.display = display
        self.rules = rules
        self.startingIndex = startingIndex
        self.muteAgents = muteAgents
        self.catchExceptions = catchExceptions
        self.gameOver = False
        self.agentCrashed = False
        self.moveHistory = []
        self.agentTimeout = False
        self.totalAgentTimes = [0] * len(agents)
        self.totalAgentTimeWarnings = [0] * len(agents)
        self.agentOutput = [io.StringIO() for _ in agents]

    def getProgress(self):
        return 1.0 if self.gameOver else self.rules.getProgress(self)

    def _agentCrash(self, agentIndex, quiet=False):
        if not quiet:
            traceback.print_exc()
        self.gameOver = True
        self.agentCrashed = True
        self.rules.agentCrash(self, agentIndex)

    def mute(self, agentIndex):
        if not self.muteAgents:
            return
        self._old_stdout, self._old_stderr = sys.stdout, sys.stderr
        sys.stdout = self.agentOutput[agentIndex]
        sys.stderr = self.agentOutput[agentIndex]

    def unmute(self):
        if not self.muteAgents:
            return
        sys.stdout, sys.stderr = self._old_stdout, self._old_stderr

    def run(self):
        """Main game loop."""
        self.display.initialize(self.state.data)
        self.numMoves = 0

        for i, agent in enumerate(self.agents):
            if not agent:
                print(f"Agent {i} failed to load")
                self._agentCrash(i, quiet=True)
                return
            if hasattr(agent, "registerInitialState"):
                self.mute(i)
                try:
                    timed_func = TimeoutFunction(agent.registerInitialState,
                                                 int(self.rules.getMaxStartupTime(i)))
                    start = time.time()
                    timed_func(self.state.deepCopy())
                    self.totalAgentTimes[i] += time.time() - start
                except TimeoutFunctionException:
                    print(f"Agent {i} ran out of time on startup!", file=sys.stderr)
                    self.agentTimeout = True
                    self._agentCrash(i, quiet=True)
                    return
                except Exception:
                    self._agentCrash(i)
                    return
                finally:
                    self.unmute()

        agentIndex, numAgents = self.startingIndex, len(self.agents)
        while not self.gameOver:
            agent = self.agents[agentIndex]
            move_time, skip = 0, False
            self.mute(agentIndex)

            try:
                if hasattr(agent, "observationFunction"):
                    timed = TimeoutFunction(agent.observationFunction,
                                            int(self.rules.getMoveTimeout(agentIndex)))
                    start = time.time()
                    observation = timed(self.state.deepCopy())
                    move_time += time.time() - start
                else:
                    observation = self.state.deepCopy()
            except TimeoutFunctionException:
                skip = True
            except Exception:
                self._agentCrash(agentIndex)
                return
            finally:
                self.unmute()

            action = None
            self.mute(agentIndex)
            try:
                timed = TimeoutFunction(agent.getAction,
                                        int(self.rules.getMoveTimeout(agentIndex)) - int(move_time))
                start = time.time()
                if not skip:
                    action = timed(observation)
                move_time += time.time() - start
                self.totalAgentTimes[agentIndex] += move_time
            except TimeoutFunctionException:
                print(f"Agent {agentIndex} timed out on a move!", file=sys.stderr)
                self.agentTimeout = True
                self._agentCrash(agentIndex, quiet=True)
                return
            except Exception:
                self._agentCrash(agentIndex)
                return
            finally:
                self.unmute()

            self.moveHistory.append((agentIndex, action))
            try:
                self.state = self.state.generateSuccessor(agentIndex, action)
            except Exception:
                self._agentCrash(agentIndex)
                return

            self.display.update(self.state.data)
            self.rules.process(self.state, self)
            agentIndex = (agentIndex + 1) % numAgents

            if _BOINC_ENABLED:
                boinc.set_fraction_done(self.getProgress())

        for i, agent in enumerate(self.agents):
            if hasattr(agent, "final"):
                self.mute(i)
                try:
                    agent.final(self.state)
                except Exception:
                    self._agentCrash(i)
                finally:
                    self.unmute()
        self.display.finish()