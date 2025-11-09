"""
Hyperspectral Cube Processing Library (v2.2.0 - Simplified API + ML Demonstrations)

Provides a CubeProcessor for hyperspectral cube processing, along with MLExamples
for educational machine learning code snippets (PCA, KMeans, KNN, Linear Regression).
"""

import numpy as np
import pandas as pd
import spectral as spy
import os
import gc
from typing import Tuple, Optional
import random
__version__ = "2.2.0"
__author__ = "Prasad, Aryan, Tanishka"


class CubeProcessor:
    """
    Main class for processing hyperspectral data cubes via a memory-efficient pipeline.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.source_metadata = {}

    def _print(self, message: str):
        if self.verbose:
            print(message)

    def open_cube(self, hdr_path: str, data_path: str) -> spy.io.spyfile.SpyFile:
        if not os.path.exists(hdr_path) or not os.path.exists(data_path):
            raise FileNotFoundError("Header or data file not found")

        img = spy.envi.open(hdr_path, data_path)
        self.source_metadata = {
            'samples': img.shape[1],
            'lines': img.shape[0],
            'bands': img.shape[2],
            'byte order': img.byte_order,
            'interleave': img.interleave
        }

        self._print(f"Cube opened (not loaded). Shape: {img.shape}")
        return img

    def parse_geometric_param(self, file_path: str, fallback_value: float = 0.0) -> float:
        if not os.path.exists(file_path):
            self._print(f"Geometric param file not found. Using fallback: {fallback_value}")
            return fallback_value

        values = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) > 0:
                        try:
                            values.append(float(parts[-1]))
                        except ValueError:
                            continue

            if values:
                mean_val = np.mean(values)
                self._print(f"Parsed geometric parameter: {mean_val:.2f}")
                return mean_val
            else:
                self._print(f"No valid values found. Using fallback: {fallback_value}")
                return fallback_value

        except Exception as e:
            self._print(f"Error parsing file: {e}. Using fallback: {fallback_value}")
            return fallback_value

    def load_flux_data(self, file_path: str) -> np.ndarray:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Flux data file not found: {file_path}")

        flux_data = np.loadtxt(file_path)
        flux_vector = flux_data[:, 1]
        self._print(f"Flux data loaded. Shape: {flux_vector.shape}")
        return flux_vector

    def radiance_to_reflectance(
        self,
        radiance_img: spy.io.spyfile.SpyFile,
        output_path_base: str,
        flux_data: np.ndarray,
        incidence_angle_deg: float,
        distance_au: float = 1.0,
        band_range: Tuple[int, int] = (5, 255),
        chunk_size: int = 256,
        interleave_format: str = 'bsq'
    ):
        self._print("Streaming radiance-to-reflectance conversion...")

        valid_formats = ['bsq', 'bil', 'bip']
        if interleave_format.lower() not in valid_formats:
            raise ValueError(f"Invalid interleave_format. Choose from {valid_formats}")

        cos_i = np.cos(np.deg2rad(incidence_angle_deg))
        eps = 1e-12

        start_band, end_band = band_range
        flux_data_cleaned = flux_data[start_band:end_band]

        lines, samples, _ = radiance_img.shape
        num_output_bands = end_band - start_band

        output_metadata = {
            'description': 'Reflectance Cube',
            'samples': str(samples),
            'lines': str(lines),
            'bands': str(num_output_bands),
            'data type': '4',
            'interleave': interleave_format,
            'file type': 'ENVI Standard',
            'byte order': self.source_metadata.get('byte order', 0)
        }

        output_hdr_path = output_path_base + '.hdr'
        os.makedirs(os.path.dirname(output_hdr_path), exist_ok=True)
        refl_file = spy.envi.create_image(
            output_hdr_path, output_metadata, ext='.qub', force=True
        )
        refl_mm = refl_file.open_memmap(writable=True)

        denominator = flux_data_cleaned[None, None, :] * cos_i * (distance_au**2) + eps

        for i in range(0, lines, chunk_size):
            chunk_end = min(i + chunk_size, lines)
            radiance_chunk = radiance_img[i:chunk_end, :, start_band:end_band]
            reflectance_chunk = (np.pi * radiance_chunk) / denominator
            refl_mm[i:chunk_end, :, :] = reflectance_chunk

        del refl_mm, refl_file
        gc.collect()
        self._print(f"Reflectance conversion complete. Saved to: {output_hdr_path}")

    def destripe_cube(
        self,
        input_img: spy.io.spyfile.SpyFile,
        output_path_base: str,
        method: str = 'median',
        chunk_size: int = 256,
        interleave_format: str = 'bsq'
    ):
        self._print(f"Destriping cube using two-pass '{method}' method...")

        valid_formats = ['bsq', 'bil', 'bip']
        if interleave_format.lower() not in valid_formats:
            raise ValueError(f"Invalid interleave_format. Choose from {valid_formats}")

        lines, samples, bands = input_img.shape
        col_stats = np.zeros((bands, samples))

        for i in range(bands):
            band_view = input_img.read_band(i)
            if method == 'median':
                col_stats[i, :] = np.median(band_view, axis=0)
            elif method == 'mean':
                col_stats[i, :] = np.mean(band_view, axis=0)
            else:
                raise ValueError("Method must be 'median' or 'mean'")

        output_metadata = {
            'description': 'Destriped Cube',
            'samples': str(samples),
            'lines': str(lines),
            'bands': str(bands),
            'data type': '4',
            'interleave': interleave_format,
            'file type': 'ENVI Standard',
            'byte order': input_img.byte_order
        }

        output_hdr_path = output_path_base + '.hdr'
        os.makedirs(os.path.dirname(output_hdr_path), exist_ok=True)
        destriped_file = spy.envi.create_image(
            output_hdr_path, output_metadata, ext='.qub', force=True
        )
        destriped_mm = destriped_file.open_memmap(writable=True)

        for i in range(0, lines, chunk_size):
            chunk_end = min(i + chunk_size, lines)
            chunk = input_img[i:chunk_end, :, :]
            corrected_chunk = chunk - col_stats[None, :, :]
            destriped_mm[i:chunk_end, :, :] = corrected_chunk

        del destriped_mm, destriped_file
        gc.collect()
        self._print(f"Destriping complete. Saved to: {output_hdr_path}")


# ==============================================================
# Machine Learning Demonstration Utilities
# ==============================================================
class MLExamples:
    """Educational ML & algorithm examples (prints code as plaintext)."""
    
    def pca(self):
        print('''\
# PCA on Iris Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("iris.csv")
X = df.iloc[:, 1:5].values
y = df['Species'].values
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Species'] = y
plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='Species', data=df_pca, s=100)
plt.title('PCA of Iris Dataset')
plt.show()
''')

    def kmeans(self):
        print('''\
# K-Means Clustering with Elbow Method
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
df = pd.read_csv("iris.csv")
X = df.iloc[:, 1:5].values
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()
''')

    def knn(self):
        print('''\
# KNN Classification on Iris Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv("iris.csv")
X = df.iloc[:, 1:5].values
y = LabelEncoder().fit_transform(df['Species'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
''')

    def linear_regression(self):
        print('''\
# Linear Regression (from scratch)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
m = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean())**2)
c = y.mean() - m * x.mean()
y_pred = m * x + c
print(f"Slope: {m:.3f}, Intercept: {c:.3f}")
print("R²:", r2_score(y, y_pred))
print("MSE:", mean_squared_error(y, y_pred))
print("MAE:", mean_absolute_error(y, y_pred))
plt.scatter(x, y, color='blue')
plt.plot(x, y_pred, color='red')
plt.title('Linear Regression (from scratch)')
plt.show()
''')

    def tictactoe_minimax(self):
        print('''\
# Tic-Tac-Toe Minimax
player, opponent = 'X', 'O'

def isMovesLeft(board):
    for row in board:
        if '_' in row:
            return True
    return False

def evaluate(board):
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != '_':
            if row[0] == player:
                return 10
            elif row[0] == opponent:
                return -10
    
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != '_':
            if board[0][col] == player:
                return 10
            elif board[0][col] == opponent:
                return -10
    
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != '_':
        if board[0][0] == player:
            return 10
        elif board[0][0] == opponent:
            return -10

    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != '_':
        if board[0][2] == player:
            return 10
        elif board[0][2] == opponent:
            return -10
    return 0

def minimax(board, depth, isMax):
    score = evaluate(board)
    if score == 10:
        return score
    if score == -10:
        return score
    if not isMovesLeft(board):
        return 0
    
    if isMax:
        best = -1000
        for i in range(3):
            for j in range(3):
                if board[i][j] == '_':
                    board[i][j] = player
                    best = max(best, minimax(board, depth+1, not isMax))
                    board[i][j] = '_'
        return best
    else:
        best = 1000
        for i in range(3):
            for j in range(3):
                if board[i][j] == '_':
                    board[i][j] = opponent
                    best = min(best, minimax(board, depth+1, not isMax))
                    board[i][j] = '_'
        return best
    
def findBestMove(board):
    bestVal = -1000
    bestMove = (-1, -1)
    for i in range(3):
        for j in range(3):
            if board[i][j] == '_':
                board[i][j] = player
                moveVal = minimax(board, 0, False)
                board[i][j] = '_'
                if moveVal > bestVal:
                    bestVal = moveVal
                    bestMove = (i, j)
    return bestMove

def printBoard(board):
    for row in board:
        print(' '.join(row))
    print()

if __name__ == "__main__":
    board = [
        ['X', 'O', 'X'],
        ['O', 'O', 'X'],
        ['_', '_', '_'],
    ]
    print("Original Board")
    printBoard(board)
    bestMove = findBestMove(board)  
    print("The Best Move is Row:", bestMove[0], " Col:", bestMove[1])
''')

    def sudoku_solver(self):
        print('''\
# Sudoku Solver (Backtracking)
def is_valid(board, r, c, num):
    for i in range(9):
        if board[r][i] == num or board[i][c] == num:
            return False
    start_row, start_col = 3 * (r // 3), 3 * (c // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False
    return True

def solve_sudoku(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                for num in range(1, 10):
                    if is_valid(board, r, c, num):
                        board[r][c] = num
                        if solve_sudoku(board):
                            return True
                        board[r][c] = 0
                return False
    return True

sudoku_board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

if solve_sudoku(sudoku_board):
    for row in sudoku_board:
        print(row)
else:
    print("No solution exists")
''')

    def graph_coloring(self):
        print('''\
# Graph Coloring (Backtracking)
def is_safe(graph, color, c, v):
    for i in range(len(graph)):
        if graph[v][i] == 1 and color[i] == c:
            return False
    return True

def graph_coloring(graph, m):
    def backtrack(v):
        if v == len(graph):
            return True
        for c in range(1, m + 1):
            if is_safe(graph, color, c, v):
                color[v] = c
                if backtrack(v + 1):
                    return True
                color[v] = 0
        return False

    color = [0] * len(graph)
    if backtrack(0):
        return color
    else:
        return None

graph = [
    [0, 1, 1, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 0]
]

m = 3
coloring = graph_coloring(graph, m)
print("Graph coloring solution:", coloring)
''')

    def chatbot(self):
        print('''\
# Simple ChatBot
import random

def chatbot():
    print("ChatBot: Hello! I'm your AI assistant.")
    print("You can say hi or ask me about: services, AI, data analysis, price or time.")
    print("(Type 'bye' to exit)\\n")

    while True:
        user = input("You: ").lower()

        if user == 'bye':
            print("ChatBot: Goodbye! Have a great day!")
            break

        elif "services" in user:
            replies = [
                "We offer AI, Data Analysis, and Cloud solutions.",
                "Our main services include AI development, analytics, and automation."
            ]

        elif "ai" in user:
            replies = [
                "Our AI solutions include model training, prediction, and automation.",
                "We develop and deploy AI systems for different use cases."
            ]

        elif "data" in user:
            replies = [
                "We perform data cleaning, visualization, and analysis.",
                "Our data analysis helps in making better business decisions."
            ]

        elif "price" in user or "charge" in user:
            replies = [
                "Pricing depends on project complexity and duration.",
                "We offer affordable plans based on your requirements."
            ]

        elif "time" in user:
            replies = [
                "Most AI projects take 1–6 months to complete.",
                "Timelines vary depending on the project scope."
            ]

        elif "hi" in user or "hello" in user:
            replies = [
                "Hello! How can I help you today?",
                "Hi there! What would you like to know?"
            ]

        elif "your name" in user:
            replies = ["I'm ChatBot, your friendly AI assistant."]

        else:
            replies = [
                "Sorry, I didn't understand that. Try asking about services, AI, or data analysis.",
                "Please ask about services, pricing, or timelines — I can help with those!"
            ]

        print("ChatBot:", random.choice(replies))

chatbot()
''')

    def alpha_beta_minimax(self):
        print('''\
# Minimax with Alpha-Beta Pruning
MIN, MAX = -1000, 1000

def minimax(depth, nodeIndex, maximizingPlayer, values, alpha, beta):
    if depth == 3:
        return values[nodeIndex]
    
    if maximizingPlayer:
        best = MIN
        for i in range(0, 2):
            val = minimax(depth+1, nodeIndex*2+i, False, values, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)
            
            if beta <= alpha:
                break
        return best
    else:
        best = MAX 
        for i in range(0, 2):
            val = minimax(depth+1, nodeIndex*2+i, True, values, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)
            
            if beta <= alpha:
                break
        return best
        
if __name__ == "__main__":
    values = [3, 5, 6, 9, 1, 2, 0, -1]
    print("The optimal value is", minimax(0, 0, True, values, MIN, MAX))
''')

    def astar(self):
        print('''\
# A* Algorithm
def astar_algo(start, goal):
    open_set = {start}
    closed_set = set()
    g = {start: 0}
    parents = {start: start}

    while open_set:
        current = min(open_set, key=lambda n: g[n] + heuristic(n))

        if current == goal:
            path = reconstructed_path(parents, start, goal)
            print(path)
            return path

        for neighbors, weight in get_neighbors(current):
            tentative_g = g[current] + weight

            if neighbors not in open_set and neighbors not in closed_set:
                open_set.add(neighbors)
                parents[neighbors] = current
                g[neighbors] = tentative_g

            elif tentative_g < g.get(neighbors, float('inf')):
                parents[neighbors] = current
                g[neighbors] = tentative_g
                if neighbors in closed_set:
                    closed_set.remove(neighbors)
                    open_set.add(neighbors)

        open_set.remove(current)
        closed_set.add(current)

    print("path not found")
    return None

def reconstructed_path(parents, start, goal):
    path = []
    node = goal
    while parents[node] != node:
        path.append(node)
        node = parents[node]

    path.append(start)
    path.reverse()

    return path

def get_neighbors(node):
    return Graph_nodes.get(node, [])

def heuristic(n):
    h_dist = {
        'A': 11,
        'B': 6,
        'C': 99,
        'D': 1,
        'E': 7,
        'G': 0,
    }
    return h_dist.get(n, float('inf'))

Graph_nodes = {
    'A': [('B', 2), ('E', 3)],
    'B': [('C', 1), ('G', 9)],
    'C': [],
    'D': [('G', 1)],
    'E': [('D', 6)],
}

# Run
astar_algo('A', 'G')
''')

    def n_queens(self):
        print('''\
# N-Queens Problem
def is_safe(board, row, col, n):
    for i in range(row):
        if board[i] == col or \\
           board[i] - i == col - row or \\
           board[i] + i == col + row:
            return False
    return True

def solve_n_queens(n):
    def backtrack(row, board):
        if row == n:
            solutions.append(board[:])
            return
        for col in range(n):
            if is_safe(board, row, col, n):
                board[row] = col
                backtrack(row + 1, board)

    solutions = []
    backtrack(0, [-1] * n)
    return solutions

solutions = solve_n_queens(8)
print(f"Number of solutions for 8-Queens: {len(solutions)}")
for sol in solutions[:1]: 
    print(sol)
''')

    def list_programs(self):
        print('''\
Available Programs:
1. PCA
2. K-Means
3. KNN Classification
4. Linear Regression
5. Tic-Tac-Toe Minimax
6. Sudoku Solver
7. Graph Coloring
8. ChatBot
9. Minimax Alpha-Beta
10. A* Algorithm
11. N-Queens Problem
''')
        
# Example usage
if __name__ == "__main__":
    demo = MLExamples()
    demo.list_programs()
    print("\n" + "="*50 + "\n")
    demo.pca()
