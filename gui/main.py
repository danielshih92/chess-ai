import pygame
import chess
from chess import Board, Move
from draw import draw_background, draw_pieces
from players import HumanPlayer
import globals
import torch
from model import ChessModel
from auxiliary_func import board_to_matrix
import pickle

# Initialize PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the mapping
with open("models_pt/move_to_int", "rb") as file:
    move_to_int = pickle.load(file)
int_to_move = {v: k for k, v in move_to_int.items()}

# Load PyTorch model
model = ChessModel(num_classes=len(move_to_int))
model.load_state_dict(torch.load("models_pt/TORCH_100EPOCHS.pth", map_location=device))
model.to(device)
model.eval()

# Prediction function
def predict_move(board: Board):
    matrix = board_to_matrix(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
    probabilities = torch.softmax(logits.squeeze(0), dim=0).cpu().numpy()
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    move_probabilities = {move: probabilities[move_to_int[move]] for move in legal_moves_uci if move in move_to_int}
    sorted_moves = sorted(move_probabilities.items(), key=lambda x: x[1], reverse=True)
    return sorted_moves[0][0] if sorted_moves else None

# Modify AIPlayer to use PyTorch model
class AIPlayer:
    def __init__(self, colour, model):
        self.colour = colour
        self.model = model

    def move(self, board, human_white):
        fen = board.fen()

        # Convert FEN to matrix using auxiliary function
        matrix = board_to_matrix(board)
        X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict the best move
        with torch.no_grad():
            logits = self.model(X_tensor)
        probabilities = torch.softmax(logits.squeeze(0), dim=0).cpu().numpy()

        legal_moves = list(board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves]

        move_probabilities = {move: probabilities[move_to_int[move]] for move in legal_moves_uci if move in move_to_int}
        sorted_moves = sorted(move_probabilities.items(), key=lambda x: x[1], reverse=True)

        # Analyze moves to prioritize stalemate and avoid checkmate
        for move_uci, _ in sorted_moves:
            move = chess.Move.from_uci(move_uci)
            board.push(move)
            if board.is_stalemate():
                # Prioritize stalemate
                board.pop()
                board.push(move)
                return
            elif not board.is_checkmate():
                # Avoid checkmate
                board.pop()
                board.push(move)
                return
            board.pop()

        # If no stalemate or safe move found, play the best move
        if sorted_moves:
            best_move = sorted_moves[0][0]
            board.push(Move.from_uci(best_move))


# Update HumanPlayer to handle clicks correctly
class HumanPlayer:
    def __init__(self, colour):
        self.colour = colour
        self.selected_square = None

    def move(self, board, event, human_white):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            x, y = event.pos
            square = self.coordinates_to_square(x, y, human_white)

            if self.selected_square is None:
                # Check if the selected square has a piece the player can move
                if any(move.uci().startswith(square) for move in board.legal_moves):
                    self.selected_square = square
                    globals.from_square = self.coordinates_to_numbers(x, y)
                return False
            else:
                # Attempt to make a move
                move_uci = self.selected_square + square
                if move_uci in [move.uci() for move in board.legal_moves]:
                    move = chess.Move.from_uci(move_uci)
                    board.push(move)
                    self.selected_square = None
                    globals.to_square = self.coordinates_to_numbers(x, y)
                    return True
                else:
                    # Reset selection if the move is invalid
                    self.selected_square = None
                    return False

    @staticmethod
    def coordinates_to_square(x, y, human_white):
        file = chr(ord('a') + x // 75)
        rank = 8 - y // 75 if human_white else y // 75 + 1
        return f"{file}{rank}"

    @staticmethod
    def coordinates_to_numbers(x, y):
        col = x // 75
        row = y // 75
        return col, row

# Initialize game components
pygame.init()
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 600
win = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Chess")
fps_clock = pygame.time.Clock()

# Players
board = Board()
human_white = True
white = HumanPlayer(colour="white")
white_ai = AIPlayer(colour="white", model=model)
black_ai = AIPlayer(colour="black", model=model)
black = black_ai

white_move = True
game_over_countdown = 50

def reset():
    global board, white_move
    board.reset()
    white_move = True
    globals.from_square = None
    globals.to_square = None

def display_game_result(screen, board):
    font = pygame.font.SysFont("Arial", 30)
    if board.is_checkmate():
        result = "White wins by checkmate!" if not board.turn else "Black wins by checkmate!"
    elif board.is_stalemate():
        result = "Draw by stalemate!"
    elif board.is_insufficient_material():
        result = "Draw by insufficient material!"
    elif board.is_seventyfive_moves():
        result = "Draw by 75-move rule!"
    elif board.is_fivefold_repetition():
        result = "Draw by fivefold repetition!"
    else:
        result = "Draw!"
    text = font.render(result, True, (255, 0, 0))
    screen.blit(text, (20, SCREEN_HEIGHT - 50))

def wait_and_display_result(screen, board, duration):
    display_game_result(screen, board)
    pygame.display.update()
    pygame.time.delay(duration)

# Main game loop
run = True
while run:
    fps_clock.tick(30)
    draw_background(win=win)
    draw_pieces(win=win, fen=board.fen(), human_white=human_white)

    if board.is_game_over():
        wait_and_display_result(win, board, duration=7000)  # 顯示 7 秒
        reset()
        game_over_countdown = 50
        continue

    pygame.display.update()

    if board.is_game_over():
        if game_over_countdown > 0:
            game_over_countdown -= 1
        else:
            reset()
            game_over_countdown = 50
        continue

    if white_move and not human_white:
        white_ai.move(board=board, human_white=human_white)
        white_move = not white_move

    if not white_move and human_white:
        black.move(board=board, human_white=human_white)
        white_move = not white_move

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            pygame.quit()
        elif white_move and human_white and white.move(board=board, event=event, human_white=human_white):
            white_move = not white_move
        elif not white_move and not human_white and black.move(board=board, event=event, human_white=human_white):
            white_move = not white_move

        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            if 625 <= x <= 675 and 200 <= y <= 260:  # Change sides
                human_white = not human_white
                white, black = black, white
                white.colour, black.colour = "white", "black"
                reset()
            elif 630 <= x <= 670 and 320 <= y <= 360:  # Reset
                reset()
