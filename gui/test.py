import chess
import chess.engine
import chess.polyglot

# 假設你有一個評估函數 evaluate_board(board) 返回當前棋局的分數
def evaluate_board(board):
    # 這裡應該是你模型的評估邏輯
    pass

# Alpha-beta 搜索算法
def alpha_beta_search(board, depth, alpha, beta, maximizing_player, transposition_table):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    board_hash = chess.polyglot.zobrist_hash(board)
    if board_hash in transposition_table:
        return transposition_table[board_hash]

    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = alpha_beta_search(board, depth - 1, alpha, beta, False, transposition_table)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        transposition_table[board_hash] = max_eval
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = alpha_beta_search(board, depth - 1, alpha, beta, True, transposition_table)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        transposition_table[board_hash] = min_eval
        return min_eval

# 找到最佳走法
def find_best_move(board, depth):
    best_move = None
    best_value = float('-inf')
    transposition_table = {}

    for move in board.legal_moves:
        board.push(move)
        board_value = alpha_beta_search(board, depth - 1, float('-inf'), float('inf'), False, transposition_table)
        board.pop()
        if board_value > best_value:
            best_value = board_value
            best_move = move

    return best_move

# 測試
board = chess.Board()
depth = 3  # 搜索深度
best_move = find_best_move(board, depth)
print(f"Best move: {best_move}")