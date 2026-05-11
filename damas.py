import argparse
import json
from dataclasses import dataclass

import cv2
import numpy as np

from chess_piece_reader import (
    BOARD_SIZE,
    CALIBRATION_FILE,
    SQUARE_SIZE,
    WARP_SIZE,
    build_homography,
    draw_status_bar,
    identify_piece,
    square_to_name,
)


GRAY_PAWN = "peao_cinza"
WHITE_PAWN = "peao_branco"
MISTAKES_FILE = "damas_erros.json"
MovePath = list[tuple[int, int]]


@dataclass
class Button:
    label: str
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class DamasApp:
    def __init__(self, camera_index: int = 0, image_path: str | None = None) -> None:
        self.camera_index = camera_index
        self.image_path = image_path
        self.cap = None
        self.static_frame = None
        self.window_name = "Damas - voce branco, PC cinza"
        self.corner_points: list[tuple[int, int]] = []
        self.homography: np.ndarray | None = None
        self.calibrating = False
        self.running = True
        self.scan_square: tuple[int, int] | None = None
        self.suggested_move: MovePath | None = None
        self.last_board: dict[tuple[int, int], str] | None = None
        self.rejected_moves: set[tuple[tuple[int, int], ...]] = set()
        self.learned_rejections: dict[str, set[tuple[tuple[int, int], ...]]] = load_learned_rejections()
        self.must_take_mode = False
        self.must_take_target: tuple[int, int] | None = None
        self.message = "Carregando calibracao..."
        self.buttons = [
            Button("Calibrar", 12, 12, 132, 48),
            Button("Jogar", 144, 12, 244, 48),
            Button("Refazer", 256, 12, 368, 48),
            Button("Must Take", 380, 12, 520, 48),
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
                self.play_turn(self.read_frame())

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
                    self.play_turn(self.read_frame())
                elif button.label == "Refazer":
                    self.retry_move()
                elif button.label == "Must Take":
                    self.start_must_take_selection()
                return

        if self.homography is None:
            self.calibrating = True
            self.add_calibration_point(x, y)
            return

        if self.must_take_mode:
            self.select_must_take_square(x, y)

    def load_saved_calibration(self) -> None:
        if not CALIBRATION_FILE.exists():
            self.message = "Clique em Calibrar e marque: a1, h1, h8, a8."
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
            self.buttons[2].x2 = 392
            self.buttons[3].x1 = 404
            self.buttons[3].x2 = 544
            self.message = "Calibracao carregada. Clique em Jogar."
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.corner_points.clear()
            self.homography = None
            self.message = "Calibracao salva invalida. Clique em Calibrar."

    def save_calibration(self) -> None:
        data = {
            "corner_points": [[int(x), int(y)] for x, y in self.corner_points],
            "order": ["a1", "h1", "h8", "a8"],
            "warp_size": WARP_SIZE,
        }
        CALIBRATION_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.buttons[0].label = "Recalibrar"
        self.buttons[0].x2 = 156
        self.buttons[1].x1 = 168
        self.buttons[1].x2 = 268
        self.buttons[2].x1 = 280
        self.buttons[2].x2 = 392
        self.buttons[3].x1 = 404
        self.buttons[3].x2 = 544

    def start_calibration(self) -> None:
        self.corner_points.clear()
        self.homography = None
        self.calibrating = True
        self.scan_square = None
        self.suggested_move = None
        self.last_board = None
        self.rejected_moves.clear()
        self.must_take_mode = False
        self.must_take_target = None
        self.message = "Calibrando: clique nos cantos a1, h1, h8, a8, nessa ordem."

    def add_calibration_point(self, x: int, y: int) -> None:
        self.corner_points.append((x, y))
        labels = ["a1", "h1", "h8", "a8"]
        if len(self.corner_points) < 4:
            self.message = f"Canto {labels[len(self.corner_points) - 1]} salvo. Clique em {labels[len(self.corner_points)]}."
            return

        self.homography = build_homography(self.corner_points)
        self.save_calibration()
        self.calibrating = False
        self.message = "Tabuleiro calibrado. Clique em Jogar."

    def play_turn(self, frame: np.ndarray) -> None:
        if self.homography is None:
            self.message = "Primeiro calibre o tabuleiro."
            return

        self.message = "Escaneando tabuleiro..."
        cv2.imshow(self.window_name, self.draw_overlay(frame.copy()))
        cv2.waitKey(1)

        board = self.scan_board()
        if self.must_take_target is not None:
            board[self.must_take_target] = WHITE_PAWN
        self.last_board = board
        self.rejected_moves = set(self.learned_rejections.get(board_key(board), set()))
        self.set_suggested_move(choose_gray_move(board, self.rejected_moves, self.must_take_target))

    def retry_move(self) -> None:
        if self.last_board is None:
            self.message = "Nenhuma leitura anterior. Clique em Jogar primeiro."
            return
        if self.suggested_move is not None:
            self.rejected_moves.add(move_key(self.suggested_move))
            key = board_key(self.last_board)
            self.learned_rejections.setdefault(key, set()).add(move_key(self.suggested_move))
            save_learned_rejections(self.learned_rejections)
            self.message = f"Aprendi a evitar: {format_move(self.suggested_move)}"

        self.set_suggested_move(choose_gray_move(self.last_board, self.rejected_moves, self.must_take_target))

    def start_must_take_selection(self) -> None:
        if self.homography is None:
            self.message = "Primeiro calibre o tabuleiro."
            return
        self.must_take_mode = True
        self.must_take_target = None
        self.suggested_move = None
        self.message = "Must Take: clique na peca branca que o PC deve comer."

    def select_must_take_square(self, x: int, y: int) -> None:
        square = self.point_to_square(x, y)
        if square is None:
            self.message = "Must Take: clique dentro do tabuleiro."
            return

        self.must_take_target = square
        self.must_take_mode = False
        self.message = f"Must Take: comer {square_to_name(square).upper()}. Clique em Jogar."

    def point_to_square(self, x: int, y: int) -> tuple[int, int] | None:
        if self.homography is None:
            return None

        point = np.array([[[x, y]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(point, self.homography)[0][0]
        wx, wy = float(warped[0]), float(warped[1])
        if wx < 0 or wy < 0 or wx >= WARP_SIZE or wy >= WARP_SIZE:
            return None

        file_index = min(BOARD_SIZE - 1, max(0, int(wx // SQUARE_SIZE)))
        rank_index = BOARD_SIZE - 1 - min(BOARD_SIZE - 1, max(0, int(wy // SQUARE_SIZE)))
        return file_index, rank_index

    def set_suggested_move(self, move: MovePath | None) -> None:
        self.suggested_move = move
        self.scan_square = None

        if move is None:
            if self.must_take_target is not None:
                self.message = f"Must Take {square_to_name(self.must_take_target).upper()}: sem captura detectada."
            else:
                self.message = "PC sem outra jogada legal para as cinzas."
            print(self.message)
            speak(self.message)
            return

        move_text = format_move(move)
        self.message = f"Jogada do PC: {move_text}"
        print(move_text)
        speak(move_text)

    def scan_board(self) -> dict[tuple[int, int], str]:
        board: dict[tuple[int, int], str] = {}
        frame = self.read_frame()

        for rank_index in range(BOARD_SIZE - 1, -1, -1):
            for file_index in range(BOARD_SIZE):
                square = (file_index, rank_index)
                crop = self.crop_square(frame, square)
                name, score = identify_piece(crop)

                if name in (GRAY_PAWN, WHITE_PAWN):
                    board[square] = name

        return board

    def crop_square(self, frame: np.ndarray, square: tuple[int, int]) -> np.ndarray:
        warped = cv2.warpPerspective(frame, self.homography, (WARP_SIZE, WARP_SIZE))
        file_index, rank_index = square
        x1 = file_index * SQUARE_SIZE
        y1 = (BOARD_SIZE - 1 - rank_index) * SQUARE_SIZE
        crop = warped[y1 : y1 + SQUARE_SIZE, x1 : x1 + SQUARE_SIZE]
        margin = int(SQUARE_SIZE * 0.10)
        return crop[margin:-margin, margin:-margin].copy()

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        self.draw_buttons(frame)

        if self.corner_points:
            for index, point in enumerate(self.corner_points, start=1):
                cv2.circle(frame, point, 6, (0, 220, 255), -1)
                cv2.putText(frame, str(index), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        if self.homography is not None:
            self.draw_board_grid(frame)

        if self.scan_square is not None:
            self.draw_square_outline(frame, self.scan_square, (0, 255, 255), 3)

        if self.must_take_target is not None:
            self.draw_square_outline(frame, self.must_take_target, (255, 0, 255), 4)

        if self.suggested_move is not None:
            self.draw_move(frame, self.suggested_move)

        draw_status_bar(frame, self.message)
        return frame

    def draw_buttons(self, frame: np.ndarray) -> None:
        for button in self.buttons:
            cv2.rectangle(frame, (button.x1, button.y1), (button.x2, button.y2), (38, 38, 38), -1)
            cv2.rectangle(frame, (button.x1, button.y1), (button.x2, button.y2), (230, 230, 230), 1)
            cv2.putText(
                frame,
                button.label,
                (button.x1 + 12, button.y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def draw_board_grid(self, frame: np.ndarray) -> None:
        inverse = np.linalg.inv(self.homography)
        for i in range(BOARD_SIZE + 1):
            points = [
                np.float32([[[i * SQUARE_SIZE, 0]]]),
                np.float32([[[i * SQUARE_SIZE, WARP_SIZE]]]),
                np.float32([[[0, i * SQUARE_SIZE]]]),
                np.float32([[[WARP_SIZE, i * SQUARE_SIZE]]]),
            ]
            a = tuple(cv2.perspectiveTransform(points[0], inverse)[0][0].astype(int))
            b = tuple(cv2.perspectiveTransform(points[1], inverse)[0][0].astype(int))
            c = tuple(cv2.perspectiveTransform(points[2], inverse)[0][0].astype(int))
            d = tuple(cv2.perspectiveTransform(points[3], inverse)[0][0].astype(int))
            cv2.line(frame, a, b, (60, 220, 60), 1)
            cv2.line(frame, c, d, (60, 220, 60), 1)

    def draw_square_outline(self, frame: np.ndarray, square: tuple[int, int], color: tuple[int, int, int], thickness: int) -> None:
        file_index, rank_index = square
        y_index = BOARD_SIZE - 1 - rank_index
        corners = np.float32(
            [
                [[file_index * SQUARE_SIZE, y_index * SQUARE_SIZE]],
                [[(file_index + 1) * SQUARE_SIZE, y_index * SQUARE_SIZE]],
                [[(file_index + 1) * SQUARE_SIZE, (y_index + 1) * SQUARE_SIZE]],
                [[file_index * SQUARE_SIZE, (y_index + 1) * SQUARE_SIZE]],
            ]
        )
        original = cv2.perspectiveTransform(corners, np.linalg.inv(self.homography)).astype(int)
        cv2.polylines(frame, [original.reshape(-1, 2)], True, color, thickness)

    def draw_move(self, frame: np.ndarray, move: MovePath) -> None:
        if len(move) < 2:
            return

        self.draw_square_outline(frame, move[0], (0, 165, 255), 3)
        self.draw_square_outline(frame, move[-1], (0, 255, 0), 3)

        for start, end in zip(move, move[1:]):
            start_center = square_center_in_frame(start, self.homography)
            end_center = square_center_in_frame(end, self.homography)
            cv2.arrowedLine(frame, start_center, end_center, (0, 255, 255), 4, cv2.LINE_AA, tipLength=0.25)


def choose_gray_move(
    board: dict[tuple[int, int], str],
    rejected: set[tuple[tuple[int, int], ...]] | None = None,
    must_take_target: tuple[int, int] | None = None,
) -> MovePath | None:
    if rejected is None:
        rejected = set()

    capture_paths = []
    moves = []

    for square, piece in board.items():
        if piece != GRAY_PAWN:
            continue

        capture_paths.extend(find_capture_paths(board, square))

        for dx in (-1, 1):
            target = (square[0] + dx, square[1] - 1)
            if must_take_target is None and is_inside(target) and target not in board:
                moves.append([square, target])

    if must_take_target is not None:
        capture_paths.extend(find_target_capture_paths(board, must_take_target))

    if capture_paths:
        if must_take_target is not None:
            capture_paths = [path for path in capture_paths if captured_squares(path) and must_take_target in captured_squares(path)]
        candidates = [path for path in capture_paths if move_key(path) not in rejected]
        if not candidates:
            return None
        return sorted(candidates, key=move_rank, reverse=True)[0]

    if moves:
        candidates = [path for path in moves if move_key(path) not in rejected]
        if not candidates:
            return None
        return sorted(candidates, key=move_rank, reverse=True)[0]
    return None


def move_key(move: MovePath) -> tuple[tuple[int, int], ...]:
    return tuple(move)


def move_rank(move: MovePath) -> tuple[int, int, int, int]:
    captures = len(move) - 1 if is_capture_path(move) else 0
    start = move[0]
    end = move[-1]
    return captures, start[1], -start[0], -end[0]


def is_capture_path(move: MovePath) -> bool:
    return any(abs(start[0] - end[0]) == 2 for start, end in zip(move, move[1:]))


def captured_squares(move: MovePath) -> list[tuple[int, int]]:
    captures = []
    for start, end in zip(move, move[1:]):
        if abs(start[0] - end[0]) == 2 and abs(start[1] - end[1]) == 2:
            captures.append(((start[0] + end[0]) // 2, (start[1] + end[1]) // 2))
    return captures


def find_target_capture_paths(board: dict[tuple[int, int], str], target: tuple[int, int]) -> list[MovePath]:
    paths = []
    target_board = dict(board)
    target_board[target] = WHITE_PAWN

    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        start = (target[0] + dx, target[1] + dy)
        landing = (target[0] - dx, target[1] - dy)
        if not is_inside(start) or not is_inside(landing):
            continue
        if target_board.get(start) == WHITE_PAWN:
            continue
        if landing in target_board:
            continue

        next_board = dict(target_board)
        next_board.pop(start, None)
        next_board.pop(target, None)
        next_board[landing] = GRAY_PAWN

        continuations = find_capture_paths(next_board, landing)
        if continuations:
            for continuation in continuations:
                paths.append([start] + continuation)
        else:
            paths.append([start, landing])

    return paths


def find_capture_paths(board: dict[tuple[int, int], str], start: tuple[int, int]) -> list[MovePath]:
    paths = []

    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        adjacent = (start[0] + dx, start[1] + dy)
        landing = (start[0] + (dx * 2), start[1] + (dy * 2))
        if not is_inside(adjacent) or not is_inside(landing):
            continue
        if board.get(adjacent) != WHITE_PAWN or landing in board:
            continue

        next_board = dict(board)
        next_board.pop(start, None)
        next_board.pop(adjacent, None)
        next_board[landing] = GRAY_PAWN

        continuations = find_capture_paths(next_board, landing)
        if continuations:
            for continuation in continuations:
                paths.append([start] + continuation)
        else:
            paths.append([start, landing])

    return paths


def is_inside(square: tuple[int, int]) -> bool:
    return 0 <= square[0] < BOARD_SIZE and 0 <= square[1] < BOARD_SIZE


def format_move(move: MovePath) -> str:
    return " to ".join(square_to_name(square).upper() for square in move)


def board_key(board: dict[tuple[int, int], str]) -> str:
    items = []
    for square, piece in sorted(board.items()):
        items.append(f"{square_to_name(square)}={piece}")
    return "|".join(items)


def move_to_storage(move: tuple[tuple[int, int], ...]) -> list[str]:
    return [square_to_name(square) for square in move]


def move_from_storage(items: list[str]) -> tuple[tuple[int, int], ...]:
    return tuple(name_to_square(item) for item in items)


def name_to_square(name: str) -> tuple[int, int]:
    clean = name.strip().lower()
    return ord(clean[0]) - ord("a"), int(clean[1]) - 1


def load_learned_rejections() -> dict[str, set[tuple[tuple[int, int], ...]]]:
    try:
        with open(MISTAKES_FILE, "r", encoding="utf-8") as file:
            raw = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

    learned: dict[str, set[tuple[tuple[int, int], ...]]] = {}
    for key, moves in raw.items():
        learned[key] = set()
        for move in moves:
            try:
                learned[key].add(move_from_storage(move))
            except (IndexError, ValueError, TypeError):
                continue
    return learned


def save_learned_rejections(learned: dict[str, set[tuple[tuple[int, int], ...]]]) -> None:
    raw = {
        key: [move_to_storage(move) for move in sorted(moves)]
        for key, moves in learned.items()
    }
    with open(MISTAKES_FILE, "w", encoding="utf-8") as file:
        json.dump(raw, file, ensure_ascii=False, indent=2, sort_keys=True)


def square_center_in_frame(square: tuple[int, int], homography: np.ndarray) -> tuple[int, int]:
    file_index, rank_index = square
    y_index = BOARD_SIZE - 1 - rank_index
    point = np.float32(
        [
            [
                [
                    (file_index * SQUARE_SIZE) + (SQUARE_SIZE / 2),
                    (y_index * SQUARE_SIZE) + (SQUARE_SIZE / 2),
                ]
            ]
        ]
    )
    original = cv2.perspectiveTransform(point, np.linalg.inv(homography))[0][0]
    return int(original[0]), int(original[1])


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
    parser = argparse.ArgumentParser(description="Jogo auxiliar de damas com camera.")
    parser.add_argument("--camera", type=int, default=0, help="Indice da camera. Padrao: 0.")
    parser.add_argument("--image", help="Usa uma imagem em vez da camera.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = DamasApp(camera_index=args.camera, image_path=args.image)
    app.run()


if __name__ == "__main__":
    main()
