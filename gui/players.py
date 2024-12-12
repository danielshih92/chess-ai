import numpy as np
import pygame
import random
import chess
import torch
from chess import Move
from auxiliary_func import board_to_matrix
from globals import square_size
import globals

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
