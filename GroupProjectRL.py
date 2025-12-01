## The implementation of the RL machine enviroment learning from gymnasium

## Contributers
# Callum Firth 2635930
# firstName lastName MatriculationNumber


from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np # Library for numerical computing
from gymnasium.envs.registration import register
import pygame


# movement directions
class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

# custom enviroment
class ProjectEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    #initialize enviroment
    def __init__(self, grid, render_mode=None): #size=5):

        self.grid = np.array(grid)
        self.size = self.grid.shape[0]
        self.window_size = 512

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
            "target": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
        })

        # 5 actions: "north", "south", "east", "west", "enter building"
        self.action_space = spaces.Discrete(5)

        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0]),
            Actions.UP.value: np.array([0, 1]),
            Actions.LEFT.value: np.array([-1, 0]),
            Actions.DOWN.value: np.array([0, -1]),
            4: np.array([0, 0])
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    # returns current observation of enviroment
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    # prvides info about current state
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    # resets enviroment to initial state
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = np.array([0, 0])  # Red
        self._target_location = np.array([4, 4]) # Blue
        return self._get_obs(), self._get_info()
    
    # updates agent in enviroment
    def step(self, action):
        if action == 4:  # enter building
            terminated = np.array_equal(self._agent_location, self._target_location)
            reward = 10 if terminated else -1
        else:
            direction = self._action_to_direction[action]
            new_loc = self._agent_location + direction
            new_loc = np.clip(new_loc, 0, self.size - 1)
            # check if walkable
            r, c = new_loc
            if self.grid[r][c] != '0':  # grid position is valid
                self._agent_location = new_loc
            reward = -1
            terminated = False

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, False, info

    # renders current state if enviroment
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        # Draw the grid cells
        for r in range(self.size):
            for c in range(self.size):
                cell = self.grid[r][c]
                rect = pygame.Rect(c * pix_square_size, r * pix_square_size,
                                pix_square_size, pix_square_size)
                if cell == 'R':
                    pygame.draw.rect(canvas, (255, 0, 0), rect)      # Red building
                elif cell == 'G':
                    pygame.draw.rect(canvas, (0, 255, 0), rect)      # Green building
                elif cell == 'Y':
                    pygame.draw.rect(canvas, (255, 255, 0), rect)    # Yellow building
                elif cell == 'B':
                    pygame.draw.rect(canvas, (0, 0, 255), rect)      # Blue building
                elif cell == '1':
                    pygame.draw.rect(canvas, (0, 150, 255), rect)    # Path in blue
                # '0' stays white (background)

        # draw agent
        pygame.draw.circle(
            canvas,
            (0, 0, 0), 
            (self._agent_location[::-1] + 0.5) * pix_square_size,  # note: swap row/col
            pix_square_size / 3,
        )

        # draw gridlines
        for x in range(self.size + 1):
            pygame.draw.line(canvas, (0, 0, 0),
                            (0, pix_square_size * x),
                            (self.window_size, pix_square_size * x), 2)
            pygame.draw.line(canvas, (0, 0, 0),
                            (pix_square_size * x, 0),
                            (pix_square_size * x, self.window_size), 2)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

import math
import heapq

# Define the Cell class
class Cell:
    def __init__(self):
        self.parent_i = 0
        self.parent_j = 0
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0

# Check if a cell is valid (within the grid bounds)
def is_valid(row, col, ROW, COL):
    return (row >= 0) and (row < ROW) and (col >= 0) and (col < COL)

# Check if a cell is unblocked (walkable path)
def is_unblocked(grid, row, col):
    # In your env: '1' is path, buildings are 'R','G','Y','B', '0' is blocked
    return grid[row][col] == '1' or grid[row][col] in ['R','G','Y','B']

# Check if a cell is the destination
def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]

# Heuristic: Euclidean distance
def calculate_h_value(row, col, dest):
    return math.sqrt((row - dest[0])**2 + (col - dest[1])**2)

# Trace path from source to destination
def trace_path(cell_details, dest):
    path = []
    row, col = dest
    while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
        path.append((row, col))
        temp_row = cell_details[row][col].parent_i
        temp_col = cell_details[row][col].parent_j
        row, col = temp_row, temp_col
    path.append((row, col))
    path.reverse()
    print("The Path is:")
    for step in path:
        print("->", step, end=" ")
    print()
    return path

# A* search algorithm
def a_star_search(grid, src, dest):
    ROW, COL = len(grid), len(grid[0])

    # Validity checks
    if not is_valid(src[0], src[1], ROW, COL) or not is_valid(dest[0], dest[1], ROW, COL):
        print("Source or destination is invalid")
        return None
    if not is_unblocked(grid, src[0], src[1]) or not is_unblocked(grid, dest[0], dest[1]):
        print("Source or destination is blocked")
        return None
    if is_destination(src[0], src[1], dest):
        print("We are already at the destination")
        return [tuple(src)]

    
    closed_list = [[False for _ in range(COL)] for _ in range(ROW)]
    
    cell_details = [[Cell() for _ in range(COL)] for _ in range(ROW)]

    i, j = src
    cell_details[i][j].f = 0
    cell_details[i][j].g = 0
    cell_details[i][j].h = 0
    cell_details[i][j].parent_i = i
    cell_details[i][j].parent_j = j
    
    open_list = []
    heapq.heappush(open_list, (0.0, i, j))

    found_dest = False

    while open_list:
        p = heapq.heappop(open_list)
        i, j = p[1], p[2]
        closed_list[i][j] = True

        directions = [(0,1),(0,-1),(1,0),(-1,0)]

        for d in directions:
            new_i, new_j = i + d[0], j + d[1]

            if is_valid(new_i, new_j, ROW, COL) and is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j]:
                if is_destination(new_i, new_j, dest):
                    cell_details[new_i][new_j].parent_i = i
                    cell_details[new_i][new_j].parent_j = j
                    print("Destination found!")
                    return trace_path(cell_details, dest)

                g_new = cell_details[i][j].g + 1.0
                h_new = calculate_h_value(new_i, new_j, dest)
                f_new = g_new + h_new

                if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                    heapq.heappush(open_list, (f_new, new_i, new_j))
                    cell_details[new_i][new_j].f = f_new
                    cell_details[new_i][new_j].g = g_new
                    cell_details[new_i][new_j].h = h_new
                    cell_details[new_i][new_j].parent_i = i
                    cell_details[new_i][new_j].parent_j = j

    print("Failed to find the destination")
    return None


# registers enviroment
register(
    id="GroupProject-v0", # unique name for your env
    entry_point="GroupProjectRL:ProjectEnv", # module:class path
)


