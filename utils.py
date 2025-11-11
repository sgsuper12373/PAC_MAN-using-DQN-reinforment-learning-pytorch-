import sys
import inspect
import heapq
import random
import signal
import time
import io

class FixedRandom:
    """Random generator with fixed state (for reproducibility)."""
    def __init__(self):
        fixed_state = (
            3,
            (2147483648, 507801126, 683453281, 310439348, 2597246090, 2209084787),
            None
        )
        self.random = random.Random()
        self.random.setstate(fixed_state)

class Stack:
    """Simple stack (LIFO)."""
    def __init__(self):
        self.list = []

    def push(self, item):
        self.list.append(item)

    def pop(self):
        return self.list.pop()

    def isEmpty(self):
        return not self.list

class Queue:
    """Simple queue (FIFO)."""
    def __init__(self):
        self.list = []

    def push(self, item):
        self.list.insert(0, item)

    def pop(self):
        return self.list.pop()

    def isEmpty(self):
        return not self.list

class PriorityQueue:
    """Priority queue with stable ordering."""
    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        heapq.heappush(self.heap, (priority, self.count, item))
        self.count += 1

    def pop(self):
        return heapq.heappop(self.heap)[2]

    def isEmpty(self):
        return not self.heap

class PriorityQueueWithFunction(PriorityQueue):
    """Priority queue using a function to determine priority."""
    def __init__(self, priorityFunction):
        super().__init__()
        self.priorityFunction = priorityFunction

    def push(self, item):
        super().push(item, self.priorityFunction(item))


def manhattanDistance(xy1, xy2):
    """Compute Manhattan distance between two points."""
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


class Counter(dict):
    """A dictionary with default 0 for missing keys and some utility methods."""
    def __getitem__(self, idx):
        return super().setdefault(idx, 0)

    def incrementAll(self, keys, count):
        for k in keys:
            self[k] += count

    def argMax(self):
        return max(self.items(), key=lambda x: x[1])[0] if self else None

    def totalCount(self):
        return sum(self.values())

    def normalize(self):
        total = float(self.totalCount())
        if total:
            for k in self:
                self[k] /= total

    def divideAll(self, d):
        d = float(d)
        for k in self:
            self[k] /= d

    def copy(self):
        return Counter(super().copy())

    def __mul__(self, y):
        return sum(self[k] * y[k] for k in self if k in y)

    def __radd__(self, y):
        for k, v in y.items():
            self[k] += v

    def __add__(self, y):
        res = Counter(self)
        for k, v in y.items():
            res[k] = res.get(k, 0) + v
        return res

    def __sub__(self, y):
        res = Counter(self)
        for k, v in y.items():
            res[k] = res.get(k, 0) - v
        return res
def raiseNotDefined():
    """Raise a standardized error for undefined methods."""
    frame = inspect.stack()[1]
    file, line, method = frame[1:4]
    print(f"*** Method not implemented: {method} at line {line} of {file}")
    sys.exit(1)


def normalize(v):
    """Normalize a Counter or list of numbers to sum to 1."""
    if isinstance(v, Counter):
        total = float(v.totalCount())
        return Counter({k: v[k] / total for k in v}) if total else v
    total = float(sum(v))
    return v if total == 0 else [x / total for x in v]


def sample(dist, values=None):
    """Sample a random element from a distribution."""
    if isinstance(dist, Counter):
        items = sorted(dist.items())
        dist, values = zip(*items)
        dist = list(dist)
        values = list(values)
    dist = normalize(dist)
    r = random.random()
    total = 0
    for prob, val in zip(dist, values):
        total += prob
        if r < total:
            return val
    return values[-1]


def flipCoin(p):
    """Return True with probability p."""
    return random.random() < p


def sign(x):
    """Return +1 for nonnegative x, -1 otherwise."""
    return 1 if x >= 0 else -1


class TimeoutFunctionException(Exception):
    """Raised when a timeout occurs."""
    pass


class TimeoutFunction:
    """Runs a function with a timeout (using SIGALRM on Unix)."""
    def __init__(self, func, timeout):
        self.function = func
        self.timeout = timeout

    def handle_timeout(self, *_):
        raise TimeoutFunctionException()

    def __call__(self, *args, **kwargs):
        if hasattr(signal, 'SIGALRM'):
            old = signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.timeout)
            try:
                result = self.function(*args, **kwargs)
            finally:
                signal.signal(signal.SIGALRM, old)
                signal.alarm(0)
        else:
            start = time.time()
            result = self.function(*args, **kwargs)
            if time.time() - start >= self.timeout:
                self.handle_timeout()
        return result


class WritableNull:
    """A writable object that discards everything (like /dev/null)."""
    def write(self, _):
        pass


_MUTED = False
_ORIGINAL_STDOUT = None


def mutePrint():
    """Suppress printing to stdout."""
    global _MUTED, _ORIGINAL_STDOUT
    if _MUTED:
        return
    _MUTED, _ORIGINAL_STDOUT = True, sys.stdout
    sys.stdout = WritableNull()


def unmutePrint():
    """Re-enable printing to stdout."""
    global _MUTED, _ORIGINAL_STDOUT
    if not _MUTED:
        return
    _MUTED = False
    sys.stdout = _ORIGINAL_STDOUT

def nearestPoint(pos):
    """Return the nearest grid point to a position (x, y)."""
    x, y = pos
    return round(x), round(y)
