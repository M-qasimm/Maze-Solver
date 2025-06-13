import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq

st.set_page_config(page_title="Maze Solver AI", layout="wide")
st.markdown("""
<style>
    .big-font {
        font-size: 28px !important;
        font-weight: bold;
        color: #3b3b98;
    }
    .small-font {
        font-size: 16px !important;
        color: #666;
    }
    .stButton>button {
        background-color: #3b3b98;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">🧭 Maze Solver using AI </p>', unsafe_allow_html=True)

st.markdown('<p class="small-font">Draw your own maze and watch BFS, DFS, or A* solve it step by step!</p>', unsafe_allow_html=True)

cols = st.columns(3)
rows = cols[0].number_input("Rows", min_value=5, max_value=15, value=7)
columns = cols[1].number_input("Columns", min_value=5, max_value=15, value=7)
algo_choice = cols[2].selectbox("Algorithm", ["BFS", "DFS", "A*"])

# Maze Input
custom_maze = []
st.subheader("🧱 Maze Editor")
st.caption("Enter each row as comma-separated 0s and 1s. 0 = path, 1 = wall")
with st.form("maze_form"):
    for i in range(int(rows)):
        default = ",".join(["0"]*int(columns))
        custom_maze.append(st.text_input(f"Row {i+1}", value=default))
    submitted = st.form_submit_button("🚀 Solve Maze")

# Algorithms
def bfs(maze, start, goal):
    queue = deque([start])
    visited = set([start])
    parent = {start: None}
    while queue:
        curr = queue.popleft()
        if curr == goal:
            break
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = curr[0]+dx, curr[1]+dy
            if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]):
                if maze[nx][ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    parent[(nx, ny)] = curr
                    queue.append((nx, ny))
    return build_path(parent, goal)

def dfs(maze, start, goal):
    stack = [start]
    visited = set([start])
    parent = {start: None}
    while stack:
        curr = stack.pop()
        if curr == goal:
            break
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = curr[0]+dx, curr[1]+dy
            if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]):
                if maze[nx][ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    parent[(nx, ny)] = curr
                    stack.append((nx, ny))
    return build_path(parent, goal)

def astar(maze, start, goal):
    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = [(0 + heuristic(start, goal), 0, start)]
    parent = {start: None}
    g_score = {start: 0}
    visited = set()
    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            break
        if current in visited:
            continue
        visited.add(current)
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = current[0]+dx, current[1]+dy
            neighbor = (nx, ny)
            if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx][ny] == 0:
                new_cost = g_score[current] + 1
                if neighbor not in g_score or new_cost < g_score[neighbor]:
                    g_score[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, new_cost, neighbor))
                    parent[neighbor] = current
    return build_path(parent, goal)

def build_path(parent, goal):
    path = []
    if goal in parent:
        node = goal
        while node:
            path.append(node)
            node = parent[node]
        path.reverse()
    return path

# Main solver
if submitted:
    try:
        maze = [[int(x) for x in row.strip().split(",")] for row in custom_maze]
        maze_np = np.array(maze)

        st.subheader("🧮 Maze Grid")
        st.dataframe(maze_np)

        start = (0, 0)
        goal = (int(rows)-1, int(columns)-1)

        if algo_choice == "BFS":
            path = bfs(maze_np, start, goal)
        elif algo_choice == "DFS":
            path = dfs(maze_np, start, goal)
        else:
            path = astar(maze_np, start, goal)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(maze_np, cmap="gray_r")

        if path:
            px, py = zip(*path)
            ax.plot(py, px, color="red", linewidth=2, label="Path")
            st.success(f"✅ Path found using {algo_choice}!")
        else:
            st.error("🚫 No path found")

        ax.scatter(0, 0, color="green", s=100, label="Start")
        ax.scatter(columns-1, rows-1, color="blue", s=100, label="End")
        ax.set_xticks(np.arange(-0.5, columns, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=0.5)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_title(f"Maze Solved with {algo_choice}")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
