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
# PCA Implementation 

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create dataset (4 samples, 3 features)
X = np.array([
    [2.7, 2.4, 0.5],
    [0.5, 0.7, 2.2],
    [2.2, 2.9, 2.9],
    [1.9, 2.2, 3.1]
])
print("Original Data:\n", X)

# Step 2: Standardize data (mean=0, std=1)
mean = np.mean(X, axis=0)
std = np.std(X, axis=0, ddof=1)
Z = (X - mean) / std
print("\nStandardized Data:\n", Z)

# Step 3: Compute covariance matrix
cov_matrix = np.cov(Z.T)
print("\nCovariance Matrix:\n", cov_matrix)

# Step 4: Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

# Step 5: Sort eigenvalues & eigenvectors (descending)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
print("\nSorted Eigenvalues:\n", eigenvalues)
print("\nSorted Eigenvectors:\n", eigenvectors)

# Step 6: Select top k eigenvectors (say k=2)
W = eigenvectors[:, :2]

# Step 7: Project data onto new axes
Z_pca = Z.dot(W)
print("\nProjected Data (2D after PCA):\n", Z_pca)

# Step 8: Visualize before and after
fig = plt.figure(figsize=(12,5))

# Original (3D)
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(Z[:, 0], Z[:, 1], Z[:, 2], color='blue')
ax1.set_title("Original Standardized 3D Data")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.set_zlabel("Feature 3")

# After PCA (2D)
ax2 = fig.add_subplot(122)
ax2.scatter(Z_pca[:, 0], Z_pca[:, 1], color='red')
ax2.set_title("Data after PCA (2D Projection)")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.grid(True)

plt.tight_layout()
plt.show()

''')

    def kmeans(self):
        print('''\
# K-Means Clustering with Elbow Method
#kmeans clustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("sales_data_sample.csv", encoding='latin')

data = df.select_dtypes(include=np.number).dropna()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.show()
''')

    def knn(self):
        print('''\
# UNIVERSAL KNN CLASSIFICATION CODE

# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Step 2: Load dataset
df = pd.read_csv("iris_data.csv")
print(df.head())

# Step 3: Separate features (X) and target (y)
X = df.drop('Species' , axis=1)
y = df['Species']

# Step 4: Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train KNN model
k = int(input("\nEnter number of neighbors (k): "))
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
print("\n KNN Model trained successfully!")

# Step 7: Make predictions
y_pred = knn.predict(X_test)

# Step 8: Compute confusion matrix & metrics
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
error_rate = 1 - accuracy

print(f"\nAccuracy: {accuracy:.2f}")
print(f"Error Rate: {error_rate:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

#step 9: visualization
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap="YlGnBu",
            xticklabels=knn.classes_,
            yticklabels=knn.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix")
plt.show()
''')

    def linear_regression(self):
        print('''\
# Linear Regression (from scratch)
#Linear regression without library
# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 2: Create Manual Dataset
data = {
    'B1': [4, 2, 3, 5, 1],
    'B2': [3, 4, 2, 5, 3]
}
df = pd.DataFrame(data)
print("Dataset:\n", df)

# Step 3: Calculate Means
x_mean = df['B1'].mean()
y_mean = df['B2'].mean()
print("\nMean of B1:", x_mean)
print("Mean of B2:", y_mean)

# Step 4: Calculate Slope (β1)
df['xy'] = (df['B1'] - x_mean) * (df['B2'] - y_mean)
df['xx'] = (df['B1'] - x_mean) ** 2
beta_1 = df['xy'].sum() / df['xx'].sum()

# Step 5: Calculate Intercept (β0)
beta_0 = y_mean - (beta_1 * x_mean)

print("\nCalculated Coefficients:")
print("Intercept (β₀):", beta_0)
print("Slope (β₁):", beta_1)

# Step 6: Predict Values using Regression Equation
df['Predicted_B2'] = beta_0 + beta_1 * df['B1']
print("\nActual vs Predicted Values:\n", df[['B1', 'B2', 'Predicted_B2']])

# Step 7: Evaluate Model Performance
mse = mean_squared_error(df['B2'], df['Predicted_B2'])
mae = mean_absolute_error(df['B2'], df['Predicted_B2'])
rmse = np.sqrt(mse)
r2 = r2_score(df['B2'], df['Predicted_B2'])

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Step 8: Visualize Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(df['B1'], df['B2'], color='blue', label='Actual Data')
plt.scatter(df['B1'], df['Predicted_B2'], color='red', label='Predicted Data')
plt.plot(df['B1'], df['Predicted_B2'], color='green', label='Regression Line')
plt.xlabel('B1')
plt.ylabel('B2')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.show()

''')

    def tictactoe_minimax(self):
        print('''\
# Tic-Tac-Toe Minimax
import math

board = [' '] * 9

def print_board():
    for i in range(0,9,3):
        print(' | '.join(board[i:i+3]))
        if i < 6: print("--+---+--")
    print()

def check_winner(b):
    combos = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,b1,c in combos:
        if b[a]==b[b1]==b[c] != ' ':
            return b[a]
    return None

def is_full(b): return ' ' not in b

def minimax(b, maximizing):
    winner = check_winner(b)
    if winner == 'X': return 1
    if winner == 'O': return -1
    if is_full(b): return 0

    best = -math.inf if maximizing else math.inf
    for i in range(9):
        if b[i] == ' ':
            b[i] = 'X' if maximizing else 'O'
            val = minimax(b, not maximizing)
            b[i] = ' '
            best = max(best, val) if maximizing else min(best, val)
    return best

def ai_move():
    best, move = -math.inf, None
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'X'
            val = minimax(board, False)
            board[i] = ' '
            if val > best:
                best, move = val, i
    board[move] = 'X'

def play():
    print("You are O, AI is X")
    print_board()
    while True:
        move = int(input("Enter your move (1-9): ")) - 1
        if board[move] != ' ': print("Invalid!"); continue
        board[move] = 'O'
        print_board()
        if check_winner(board) == 'O': print(" You win!"); break
        if is_full(board): print("It's a tie!"); break

        print("AI thinking...")
        ai_move()
        print_board()
        if check_winner(board) == 'X': print(" AI wins!"); break
        if is_full(board): print("It's a tie!"); break

play()

''')

    def sudoku_solver(self):
        print('''\
# Sudoku Solver (Backtracking)
sudoku = [
 [5,3,0, 0,7,0, 0,0,0],
 [6,0,0, 1,9,5, 0,0,0],
 [0,9,8, 0,0,0, 0,6,0],

 [8,0,0, 0,6,0, 0,0,3],
 [4,0,0, 8,0,3, 0,0,1],
 [7,0,0, 0,2,0, 0,0,6],

 [0,6,0, 0,0,0, 2,8,0],
 [0,0,0, 4,1,9, 0,0,5],
 [0,0,0, 0,8,0, 0,7,9]
]

def find():
    for r in range(9):
        for c in range(9):
            if sudoku[r][c] == 0:
                return r, c
    return None

def valid(num, r, c):
    if num in sudoku[r]:
        return False
    if num in [sudoku[i][c] for i in range(9)]:
        return False
    br, bc = 3*(r//3), 3*(c//3)
    for i in range(br, br+3):
        for j in range(bc, bc+3):
            if sudoku[i][j] == num:
                return False
    return True

def solve():
    pos = find()
    if not pos:
        return True
    r, c = pos
    for num in range(1, 10):
        if valid(num, r, c):
            sudoku[r][c] = num
            if solve():
                return True
            sudoku[r][c] = 0
    return False

solve()

for row in sudoku:
    print(row)
''')

    def graph_coloring(self):
        print('''\
# Graph Coloring (Backtracking)
graph=[
    [0,1,1,1],
    [1,0,1,0],
    [1,1,0,1],
    [1,0,1,0]
]
V=len(graph)
colors=[0]*V
num_colors=3

def is_safe_color(node,color):
    for i in range(V):
        if graph[node][i]==1 and colors[i]==color:
            return False
        return True
def solve_graph_coloring(node=0):
    if node == V:
        print("graph coloring solution:",colors)
        return True
    for color in range(1,num_colors+1):
        if is_safe_color(node, color):
            colors[node]=color
            if solve_graph_coloring(node+1):
                return True
            colors[node]=0
    return False
print("\nGraph coloring using backtracking\n")
solve_graph_coloring()
''')

    def chatbot(self):
        print('''\
# Simple Customer Support Chatbot
# Greeting message
print(" Hello! Welcome to ShopSmart Customer Support.")
print("How can I assist you today?")
print("Type 'help' to see what I can do or 'exit' to end the chat.\n")
# Function to get chatbot response
def chatbot_response(user_input):
    user_input = user_input.lower()
    if "help" in user_input:
        return (
            "I can assist you with:\n"
            "- Order status\n"
            "- Return or refund\n"
            "- Product information\n"
            "- Contact support\n"
            "- Store hours"
        )
    elif "order" in user_input or "status" in user_input:
        return "Please provide your order ID (e.g., #1234), and I’ll check its status for you."
    elif "refund" in user_input or "return" in user_input:
        return "To initiate a return, visit your orders page → select the product → choose 'Return/Refund'."
    elif "product" in user_input or "details" in user_input:
        return "Please tell me the product name or category, and I’ll share the details."
    elif "contact" in user_input or "support" in user_input:
        return "You can contact our support team at support@shopsmart.com or   +1-800-555-0199."
    elif "hours" in user_input or "open" in user_input:
        return "Our stores are open 9 AM to 9 PM, Monday through Saturday."
    elif "thank" in user_input:
        return "You're most welcome!  Is there anything else I can help you with?"
    elif "exit" in user_input or "bye" in user_input:
        return "Goodbye! Thanks for visiting ShopSmart."
    else:
        return "I'm sorry, I didn’t quite get that. Type 'help' to see what I can assist you with."
 # Chat loop
while True:
    user_input = input("You: ")
    response = chatbot_response(user_input)
    print("Bot:", response)
    if "goodbye" in response.lower() or "thanks for visiting" in response.lower():
        break
''')

    def alpha_beta_minimax(self):
        print('''\
board = [' '] * 9

def print_board(b):
    for i in range(0, 9, 3):
        print(f"{b[i]} | {b[i+1]} | {b[i+2]}")
        if i < 6: print("--+---+--")
    print()

def check_winner(b):
    for a, b1, c in [(0,1,2),(3,4,5),(6,7,8),
                     (0,3,6),(1,4,7),(2,5,8),
                     (0,4,8),(2,4,6)]:
        if b[a] == b[b1] == b[c] != " ":
            return b[a]
    return None

def is_full(b):
    return ' ' not in b

def alphabeta(b, depth, alpha, beta, isMax):
    w = check_winner(b)
    if w == 'X': return 1      # AI wins
    if w == 'O': return -1     # Player wins
    if is_full(b): return 0

    if isMax:  # AI's turn (X)
        best = -999
        for i in range(9):
            if b[i] == ' ':
                b[i] = 'X'
                best = max(best, alphabeta(b, depth + 1, alpha, beta, False))
                b[i] = ' '
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
        return best
    else:  # Player's turn (O)
        best = 999
        for i in range(9):
            if b[i] == ' ':
                b[i] = 'O'
                best = min(best, alphabeta(b, depth + 1, alpha, beta, True))
                b[i] = ' '
                beta = min(beta, best)
                if beta <= alpha:
                    break
        return best

def best_move(b):
    best_val, move = -999, -1
    for i in range(9):
        if b[i] == ' ':
            b[i] = 'X'
            val = alphabeta(b, 0, -999, 999, False)
            b[i] = ' '
            if val > best_val:
                best_val, move = val, i
    return move

def play():
    print("Tic-Tac-Toe using Alpha-Beta Pruning (AI = X, You = O)")
    print_board(board)
    while True:
        m = int(input("Enter your move (1-9): ")) - 1
        if board[m] != ' ':
            print("Invalid move! Try again.")
            continue
        board[m] = 'O'
        if check_winner(board) == 'O':
            print_board(board)
            print("You win!")
            break
        if is_full(board):
            print_board(board)
            print("It's a Draw!")
            break
        ai = best_move(board)
        board[ai] = 'X'
        print_board(board)
        if check_winner(board) == 'X':
            print(" AI wins!")
            break
        if is_full(board):
            print("It's a Draw!")
            break

play()

''')

    def astar(self):
        print('''\
#ASTAR
def astar(start, stop):
    open_set = {start}
    closed_set = set()
    g, parents = {start: 0}, {start: start}

    while open_set:
        n = None
        for v in open_set:
            if n is None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v

        if n == stop:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start)
            path.reverse()
            print("Path found:", path)
            return path

        for (m, w) in get_neighbours(n):
            if m not in open_set and m not in closed_set:
                open_set.add(m)
                parents[m] = n
                g[m] = g[n] + w
            elif g[m] > g[n] + w:
                g[m] = g[n] + w
                parents[m] = n
                if m in closed_set:
                    closed_set.remove(m)
                    open_set.add(m)

        open_set.remove(n)
        closed_set.add(n)

    print("Path not found")
    return None


def get_neighbours(n):
    return Graph.get(n, [])


def heuristic(n):
    h = {'A': 11, 'B': 6, 'C': 99, 'D': 1, 'E': 7, 'G': 0}
    return h.get(n, float('inf'))


Graph = {
    'A': [('B', 2), ('E', 3)],
    'B': [('C', 1), ('G', 9)],
    'C': [],
    'D': [('G', 1)],
    'E': [('D', 6)],
}

astar('A', 'G')
''')

    def n_queens(self):
        print('''\
#N-queens
N = 4
def print_solution(board):
    for row in board:
        print(" ".join("Q" if col else "." for col in row))
    print()

def is_safe(board, row, col):
    for i in range(row):
        if board[i][col]:
            return False

    i, j = row, col
    while i >= 0 and j >= 0:
        if board[i][j]:
            return False
        i -= 1
        j -= 1

    i, j = row, col
    while i >= 0 and j < N:
        if board[i][j]:
            return False
        i -= 1
        j += 1

    return True

def solve_n_queens(board, row=0):
    if row == N:
        print_solution(board)
        return True
    for col in range(N):
        if is_safe(board, row, col):
            board[row][col] = 1
            solve_n_queens(board, row + 1)
            board[row][col] = 0
    return False

board = [[0]*N for _ in range(N)]
solve_n_queens(board)


#option2
# N-Queens using Branch and Bound
N = 4
cols = [False]*N
diag1 = [False]*(2*N)
diag2 = [False]*(2*N)
board = [-1]*N

def solve(row):
    if row == N:
        print(board)
        return
    for col in range(N):
        if not cols[col] and not diag1[row+col] and not diag2[row-col+N]:
            board[row] = col
            cols[col] = diag1[row+col] = diag2[row-col+N] = True
            solve(row+1)
            cols[col] = diag1[row+col] = diag2[row-col+N] = False

solve(0)

''')
    
    def logistic_regression(self):
        print('''\
# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Step 2: Load Dataset

# Option 1: Load from CSV
# df = pd.read_csv("iris_data.csv") 
# Option 2: Load built-in dataset (Iris example)
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.head())

# Step 3: Separate features (X) and target (y)
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']
                                      OR
X=df.drop('Species', axis=1)
y=df['Salary']


# Step 4: Encode categorical data (if any)(OPTIONAL STEP)
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# Step 5: Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create and train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))       
      ''')
        
    def svm(self):
        print('''\
#Support vector machine for email binary classification
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.svm import SVC

data = pd.read_csv("emails_data.csv")
data.head()
data = data.drop('Email No.', axis=1)

# Drop rows with missing values
data = data.dropna()

X = data.drop('Prediction', axis=1)
y = data['Prediction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(gamma='auto')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Distribution of Classes
plt.figure(figsize=(5, 4))
sns.countplot(x=y)
plt.title('Distribution of Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
#svm confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="YlGnBu", fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.show()             
''')
        
    def pandas(self):
        print('''\
# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load Dataset
df = pd.read_csv("Iris.csv")
df.head()
df.info()
df.describe()
df.nunique()
df.shape
df.columns

#Step3Check missing values
df.isnull().sum()
age_median=df['Age'].median()
df['Age'].fillna(age_median , inplace=True)

age_mode=df['Age'].mode()[0]
df['Age'].fillna(age_mode, inplace=True)
df['Temp']=df['Temp'].fillna(method='ffill')

#Step4 Outliers
# Select numeric columns
from scipy import stats
fare_zscores=np.abs(stats.zscore(df['Fare'])
df_clean=df[fare_zscores<3].copy()
print("Original shape:", df.shape)
print("After removing outliers:", df_clean.shape)

#option2 outliers
num_cols = df.select_dtypes(include=[np.number]).columns
z = np.abs(stats.zscore(df[num_cols]))
threshold = 3
df_clean = df[(z < threshold).all(axis=1)]
print("Original shape:", df.shape)
print("After removing outliers:", df_clean.shape)

# Optional: visualize
sns.boxplot(data=df_clean[num_cols])
plt.title("After Z-score Outlier Removal")
plt.show()

# Step 6: Data Transformation (Log Transform)
df['PetalLengthCm_log'] = np.log(df['PetalLengthCm'] + 1)

# Compare before and after
sns.histplot(df['PetalLengthCm'], color='blue', label='Original', kde=True)
sns.histplot(df['PetalLengthCm_log'], color='red', label='Log Transformed', kde=True)
plt.legend()
plt.show()


# Step 7: Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species_encoded'] = le.fit_transform(df['Species'])

df[['Species', 'Species_encoded']].head()              
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
12. Logistic Regression
13. SVM
14. Pandas
''')
        
# Example usage
if __name__ == "__main__":
    demo = MLExamples()
    demo.list_programs()
    print("\n" + "="*50 + "\n")
    demo.pca()
processor.py
Displaying processor.py.
