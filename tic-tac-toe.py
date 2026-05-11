import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from chess_piece_reader import draw_status_bar, identify_piece


GRID_SIZE = 3
CELL_SIZE = 140
WARP_SIZE = GRID_SIZE * CELL_SIZE
CALIBRATION_FILE = Path("tic_tac_toe_calibration.json")
GRAY_PIECE = "peao_cinza"
WHITE_PIECE = "peao_branco"
COMPUTER = "O"
HUMAN = "X"
EMPTY = ""


@dataclass
class Button:
    label: str
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class TicTacToeApp:
    def __init__(self, camera_index: int = 0, image_path: str | None = None) -> None:
        self.camera_index = camera_index
        self.image_path = image_path
        self.cap = None
        self.static_frame = None
        self.window_name = "Tic-Tac-Toe - PC cinza"
        self.corner_points: list[tuple[int, int]] = []
        self.homography: np.ndarray | None = None
        self.calibrating = False
        self.running = True
        self.difficulty = "normal"
        self.suggested_cell: tuple[int, int] | None = None
        self.message = "Clique em Calibrar e marque os 4 cantos externos da grade."
        self.buttons = [
            Button("Calibrar", 12, 12, 132, 48),
            Button("Jogar", 144, 12, 244, 48),
            Button("Facil", 256, 12, 336, 48),
            Button("Normal", 348, 12, 448, 48),
            Button("Dificil", 460, 12, 560, 48),
        ]
        self.load_saved_calibration()

    def run(self) -> None:
        if self.image_path:
            self.static_frame = cv2.imread(self.image_path)
            if self.static_frame is None:
                raise FileNotFoundError(f"Nao consegui abrir a imagem: {self.image_path}")
        else:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError("Nao consegui abrir a camera.")

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while self.running:
            frame = self.read_frame()
            display = self.draw_overlay(frame.copy())
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                self.start_calibration()
            if key in (ord("j"), 13):
                self.play_turn()

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def read_frame(self) -> np.ndarray:
        if self.static_frame is not None:
            return self.static_frame.copy()

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Falha ao ler a camera.")
        return frame

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.calibrating:
            self.add_calibration_point(x, y)
            return

        for button in self.buttons:
            if button.contains(x, y):
                if button.label in ("Calibrar", "Recalibrar"):
                    self.start_calibration()
                elif button.label == "Jogar":
                    self.play_turn()
                elif button.label == "Facil":
                    self.set_difficulty("facil")
                elif button.label == "Normal":
                    self.set_difficulty("normal")
                elif button.label == "Dificil":
                    self.set_difficulty("dificil")
                return

        if self.homography is None:
            self.calibrating = True
            self.add_calibration_point(x, y)

    def set_difficulty(self, difficulty: str) -> None:
        self.difficulty = difficulty
        self.message = f"Nivel: {difficulty}. Clique em Jogar."

    def start_calibration(self) -> None:
        self.corner_points.clear()
        self.homography = None
        self.calibrating = True
        self.suggested_cell = None
        self.message = "Calibrando: clique nos cantos: inferior esquerdo, inferior direito, superior direito, superior esquerdo."

    def add_calibration_point(self, x: int, y: int) -> None:
        self.corner_points.append((x, y))
        labels = ["inferior esquerdo", "inferior direito", "superior direito", "superior esquerdo"]
        if len(self.corner_points) < 4:
            self.message = f"Canto {labels[len(self.corner_points) - 1]} salvo. Clique no canto {labels[len(self.corner_points)]}."
            return

        self.homography = build_homography(self.corner_points)
        self.save_calibration()
        self.calibrating = False
        self.message = "Grade calibrada. Clique em Jogar."

    def load_saved_calibration(self) -> None:
        if not CALIBRATION_FILE.exists():
            return

        try:
            data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
            points = data["corner_points"]
            if len(points) != 4:
                raise ValueError("calibracao precisa de 4 pontos")
            self.corner_points = [(int(point[0]), int(point[1])) for point in points]
            self.homography = build_homography(self.corner_points)
            self.buttons[0].label = "Recalibrar"
            self.buttons[0].x2 = 156
            self.buttons[1].x1 = 168
            self.buttons[1].x2 = 268
            self.buttons[2].x1 = 280
            self.buttons[2].x2 = 360
            self.buttons[3].x1 = 372
            self.buttons[3].x2 = 472
            self.buttons[4].x1 = 484
            self.buttons[4].x2 = 584
            self.message = "Calibracao carregada. Clique em Jogar."
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.corner_points.clear()
            self.homography = None
            self.message = "Calibracao salva invalida. Clique em Calibrar."

    def save_calibration(self) -> None:
        data = {
            "corner_points": [[int(x), int(y)] for x, y in self.corner_points],
            "order": ["inferior_esquerdo", "inferior_direito", "superior_direito", "superior_esquerdo"],
            "warp_size": WARP_SIZE,
        }
        CALIBRATION_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.buttons[0].label = "Recalibrar"
        self.buttons[0].x2 = 156
        self.buttons[1].x1 = 168
        self.buttons[1].x2 = 268
        self.buttons[2].x1 = 280
        self.buttons[2].x2 = 360
        self.buttons[3].x1 = 372
        self.buttons[3].x2 = 472
        self.buttons[4].x1 = 484
        self.buttons[4].x2 = 584

    def play_turn(self) -> None:
        if self.homography is None:
            self.message = "Primeiro calibre a grade."
            return

        board = self.scan_board()
        winner = find_winner(board)
        if winner:
            self.handle_winner(winner)
            return

        move = choose_move(board, self.difficulty)
        self.suggested_cell = move

        if move is None:
            self.message = "Empate. Vamos tentar de novo!"
            speak("Draw. Let's try again!")
            return

        move_text = cell_name(move)
        self.message = f"PC joga em {move_text}"
        print(move_text)
        speak(move_text)

    def handle_winner(self, winner: str) -> None:
        self.suggested_cell = None
        if winner == COMPUTER:
            self.message = "I won, let's try again!"
            print(self.message)
            speak(self.message)
        elif winner == HUMAN:
            self.message = "You won, congratulations!"
            print(self.message)
            speak(self.message)

    def scan_board(self) -> list[list[str]]:
        board = [[EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        frame = self.read_frame()

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                crop = self.crop_cell(frame, row, col)
                name, _ = identify_piece(crop)
                if name == GRAY_PIECE:
                    board[row][col] = COMPUTER
                elif name == WHITE_PIECE:
                    board[row][col] = HUMAN

        return board

    def crop_cell(self, frame: np.ndarray, row: int, col: int) -> np.ndarray:
        warped = cv2.warpPerspective(frame, self.homography, (WARP_SIZE, WARP_SIZE))
        x1 = col * CELL_SIZE
        y1 = row * CELL_SIZE
        crop = warped[y1 : y1 + CELL_SIZE, x1 : x1 + CELL_SIZE]
        margin = int(CELL_SIZE * 0.12)
        return crop[margin:-margin, margin:-margin].copy()

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        self.draw_buttons(frame)

        if self.corner_points:
            for index, point in enumerate(self.corner_points, start=1):
                cv2.circle(frame, point, 6, (0, 220, 255), -1)
                cv2.putText(frame, str(index), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        if self.homography is not None:
            self.draw_grid(frame)

        if self.suggested_cell is not None:
            self.draw_cell_outline(frame, self.suggested_cell, (0, 255, 255), 4)

        draw_status_bar(frame, self.message)
        return frame

    def draw_buttons(self, frame: np.ndarray) -> None:
        for button in self.buttons:
            active = button.label.lower() == self.difficulty
            fill = (52, 84, 52) if active else (38, 38, 38)
            cv2.rectangle(frame, (button.x1, button.y1), (button.x2, button.y2), fill, -1)
            cv2.rectangle(frame, (button.x1, button.y1), (button.x2, button.y2), (230, 230, 230), 1)
            cv2.putText(
                frame,
                button.label,
                (button.x1 + 10, button.y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def draw_grid(self, frame: np.ndarray) -> None:
        inverse = np.linalg.inv(self.homography)
        for i in range(GRID_SIZE + 1):
            vertical = [
                np.float32([[[i * CELL_SIZE, 0]]]),
                np.float32([[[i * CELL_SIZE, WARP_SIZE]]]),
            ]
            horizontal = [
                np.float32([[[0, i * CELL_SIZE]]]),
                np.float32([[[WARP_SIZE, i * CELL_SIZE]]]),
            ]
            a = tuple(cv2.perspectiveTransform(vertical[0], inverse)[0][0].astype(int))
            b = tuple(cv2.perspectiveTransform(vertical[1], inverse)[0][0].astype(int))
            c = tuple(cv2.perspectiveTransform(horizontal[0], inverse)[0][0].astype(int))
            d = tuple(cv2.perspectiveTransform(horizontal[1], inverse)[0][0].astype(int))
            cv2.line(frame, a, b, (60, 220, 60), 1)
            cv2.line(frame, c, d, (60, 220, 60), 1)

    def draw_cell_outline(self, frame: np.ndarray, cell: tuple[int, int], color: tuple[int, int, int], thickness: int) -> None:
        row, col = cell
        corners = np.float32(
            [
                [[col * CELL_SIZE, row * CELL_SIZE]],
                [[(col + 1) * CELL_SIZE, row * CELL_SIZE]],
                [[(col + 1) * CELL_SIZE, (row + 1) * CELL_SIZE]],
                [[col * CELL_SIZE, (row + 1) * CELL_SIZE]],
            ]
        )
        original = cv2.perspectiveTransform(corners, np.linalg.inv(self.homography)).astype(int)
        cv2.polylines(frame, [original.reshape(-1, 2)], True, color, thickness)


def build_homography(corner_points: list[tuple[int, int]]) -> np.ndarray:
    source = np.float32(corner_points)
    destination = np.float32(
        [
            [0, WARP_SIZE],
            [WARP_SIZE, WARP_SIZE],
            [WARP_SIZE, 0],
            [0, 0],
        ]
    )
    return cv2.getPerspectiveTransform(source, destination)


def choose_move(board: list[list[str]], difficulty: str) -> tuple[int, int] | None:
    empty = empty_cells(board)
    if not empty:
        return None

    if difficulty == "facil":
        return random.choice(empty)

    if difficulty == "normal" and random.random() < 0.30:
        return random.choice(empty)

    return best_move(board)


def best_move(board: list[list[str]]) -> tuple[int, int] | None:
    empty = empty_cells(board)
    if not empty:
        return None

    for cell in empty:
        if would_win(board, cell, COMPUTER):
            return cell

    for cell in empty:
        if would_win(board, cell, HUMAN):
            return cell

    if (1, 1) in empty:
        return (1, 1)

    for cell in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        if cell in empty:
            return cell

    return random.choice(empty)


def would_win(board: list[list[str]], cell: tuple[int, int], player: str) -> bool:
    row, col = cell
    copy = [line[:] for line in board]
    copy[row][col] = player
    return find_winner(copy) == player


def empty_cells(board: list[list[str]]) -> list[tuple[int, int]]:
    return [(row, col) for row in range(GRID_SIZE) for col in range(GRID_SIZE) if board[row][col] == EMPTY]


def find_winner(board: list[list[str]]) -> str | None:
    lines = []
    lines.extend(board)
    lines.extend([[board[0][col], board[1][col], board[2][col]] for col in range(GRID_SIZE)])
    lines.append([board[0][0], board[1][1], board[2][2]])
    lines.append([board[0][2], board[1][1], board[2][0]])

    for line in lines:
        if line[0] and line[0] == line[1] == line[2]:
            return line[0]
    return None


def cell_name(cell: tuple[int, int]) -> str:
    row, col = cell
    return f"{chr(ord('A') + col)}{row + 1}"


def speak(text: str) -> None:
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return
    except Exception:
        pass

    try:
        import win32com.client

        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Speak(text)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tic-tac-toe com camera.")
    parser.add_argument("--camera", type=int, default=0, help="Indice da camera. Padrao: 0.")
    parser.add_argument("--image", help="Usa uma imagem em vez da camera.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = TicTacToeApp(camera_index=args.camera, image_path=args.image)
    app.run()


if __name__ == "__main__":
    main()
