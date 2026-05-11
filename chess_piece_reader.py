import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


BOARD_SIZE = 8
SQUARE_SIZE = 96
WARP_SIZE = BOARD_SIZE * SQUARE_SIZE
PIECES_DIR = Path("peace")
CALIBRATION_FILE = Path("calibration.json")
EMPTY_LIGHT_CLASS = "vazia_clara"
EMPTY_DARK_CLASS = "vazia_escura"
EMPTY_CLASSES = {EMPTY_LIGHT_CLASS, EMPTY_DARK_CLASS}
UNKNOWN_THRESHOLD = 0.56
UNKNOWN_MARGIN = 0.025
TOP_SAMPLE_COUNT = 4
OCCUPIED_EDGE_THRESHOLD = 0.075
OCCUPIED_CENTER_DIFF_THRESHOLD = 22.0
EMPTY_MATCH_THRESHOLD = 0.62
EMPTY_REJECT_THRESHOLD = 0.78
EMPTY_OVER_PIECE_MARGIN = 0.08
EXPORT_SCAN_DELAY_MS = 450


@dataclass
class Button:
    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    piece_name: str | None = None

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class ChessPieceReader:
    def __init__(self, camera_index: int = 0, image_path: str | None = None) -> None:
        self.camera_index = camera_index
        self.image_path = image_path
        self.cap = None
        self.static_frame = None
        self.window_name = "Leitor de pecas de xadrez"
        self.corner_points: list[tuple[int, int]] = []
        self.homography: np.ndarray | None = None
        self.calibrating = False
        self.running = True
        self.selected_square: tuple[int, int] | None = None
        self.training_piece_name: str | None = None
        self.mouse_position: tuple[int, int] = (0, 0)
        self.last_frame_width = 640
        self.message = "Clique em Calibrar e marque: a1, h1, h8, a8."
        self.buttons = [
            Button("Calibrar", 12, 12, 132, 48),
            Button("Exportar", 144, 12, 276, 48),
        ]
        self.piece_buttons = make_piece_buttons(self.last_frame_width)
        PIECES_DIR.mkdir(exist_ok=True)
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
        self.mouse_position = (x, y)

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.calibrating:
            self.add_calibration_point(x, y)
            return

        for button in self.piece_buttons:
            if button.contains(x, y):
                self.select_training_piece(button)
                return

        for button in self.buttons:
            if button.contains(x, y):
                if button.label in ("Calibrar", "Recalibrar"):
                    self.start_calibration()
                elif button.label == "Exportar":
                    self.export_board(self.read_frame())
                return

        if self.homography is None:
            self.calibrating = True
            self.add_calibration_point(x, y)
            return

        square = self.point_to_square(x, y)
        if square is None:
            self.message = "Clique dentro do tabuleiro."
            return

        self.selected_square = square
        frame = self.read_frame()
        if self.training_piece_name is not None:
            self.save_selected_piece_sample(frame, square)
            return
        self.guess_square(frame, square)

    def start_calibration(self) -> None:
        self.corner_points.clear()
        self.homography = None
        self.calibrating = True
        self.selected_square = None
        self.training_piece_name = None
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
        self.message = "Tabuleiro calibrado. Clique em uma casa para adivinhar."

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
            self.buttons[1].x2 = 300
            self.message = "Calibracao carregada. Clique em uma casa para adivinhar."
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
        self.buttons[1].x2 = 300

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

    def crop_square(self, frame: np.ndarray, square: tuple[int, int]) -> np.ndarray:
        warped = cv2.warpPerspective(frame, self.homography, (WARP_SIZE, WARP_SIZE))
        file_index, rank_index = square
        x1 = file_index * SQUARE_SIZE
        y1 = (BOARD_SIZE - 1 - rank_index) * SQUARE_SIZE
        crop = warped[y1 : y1 + SQUARE_SIZE, x1 : x1 + SQUARE_SIZE]

        margin = int(SQUARE_SIZE * 0.10)
        return crop[margin:-margin, margin:-margin].copy()

    def identify_selected_square(self, frame: np.ndarray) -> None:
        if self.homography is None:
            self.message = "Primeiro calibre o tabuleiro."
            return
        if self.selected_square is None:
            self.message = "Primeiro clique em uma casa."
            return

        self.identify_square(frame, self.selected_square)

    def guess_selected_square(self, frame: np.ndarray) -> None:
        if self.homography is None:
            self.message = "Primeiro calibre o tabuleiro."
            return
        if self.selected_square is None:
            self.message = "Primeiro clique em uma casa."
            return

        self.guess_square(frame, self.selected_square)

    def identify_square(self, frame: np.ndarray, square: tuple[int, int]) -> None:
        crop = self.crop_square(frame, square)
        name, score = identify_piece(crop)
        square_name = square_to_name(square)

        if name is not None:
            self.message = f"{name} em {square_name} (confianca {score:.2f})"
        else:
            self.message = f"Nao reconheci {square_name}. Escolha uma peca lateral e clique na casa para gravar."
        print(self.message)

    def guess_square(self, frame: np.ndarray, square: tuple[int, int]) -> None:
        crop = self.crop_square(frame, square)
        square_name = square_to_name(square)
        occupied, occupancy_score = is_square_occupied(crop)
        name, piece_score = identify_piece(crop)
        occupied = occupied or name is not None

        if not occupied:
            self.message = f"{square_name}: livre (ocupacao {occupancy_score:.2f})"
            print(self.message)
            return

        if name is not None:
            self.message = f"{square_name}: ocupada por {name} (peca {piece_score:.2f}, ocupacao {occupancy_score:.2f})"
        else:
            self.message = f"{square_name}: ocupada, mas ainda nao sei qual peca. Escolha a peca lateral e clique na casa."
        print(self.message)

    def export_board(self, frame: np.ndarray) -> None:
        if self.homography is None:
            self.message = "Primeiro calibre o tabuleiro."
            return

        pieces = {}
        previous_square = self.selected_square

        for rank_index in range(BOARD_SIZE - 1, -1, -1):
            for file_index in range(BOARD_SIZE):
                square = (file_index, rank_index)
                square_name = square_to_name(square)
                self.selected_square = square
                self.message = f"Exportando: lendo {square_name}..."

                scan_frame = self.read_frame()
                crop = self.crop_square(scan_frame, square)
                occupied, occupancy_score = is_square_occupied(crop)
                name, piece_score = identify_piece(crop)

                if name is not None:
                    pieces[square_name] = name
                    self.message = f"Exportando {square_name}: {name} ({piece_score:.2f})"
                elif occupied:
                    self.message = f"Exportando {square_name}: ocupada, nao reconhecida ({occupancy_score:.2f})"
                else:
                    self.message = f"Exportando {square_name}: livre ({occupancy_score:.2f})"

                cv2.imshow(self.window_name, self.draw_overlay(scan_frame.copy()))
                key = cv2.waitKey(EXPORT_SCAN_DELAY_MS) & 0xFF
                if key in (ord("q"), 27):
                    self.selected_square = previous_square
                    self.message = "Exportacao cancelada."
                    return

        self.selected_square = previous_square
        board_json = json.dumps(pieces, ensure_ascii=False, indent=2, sort_keys=True)
        print(board_json)
        self.message = f"Exportado JSON com {len(pieces)} pecas reconhecidas."

    def select_training_piece(self, button: Button) -> None:
        if not button.piece_name:
            return
        self.training_piece_name = button.piece_name
        self.message = f"Selecionado: {format_piece_name(button.piece_name)}. Clique em uma casa para gravar exemplo."

    def save_selected_piece_sample(self, frame: np.ndarray, square: tuple[int, int]) -> None:
        if self.training_piece_name is None:
            return

        crop = self.crop_square(frame, square)
        square_name = square_to_name(square)
        saved_path = save_piece_sample(crop, self.training_piece_name)
        self.message = f"Aprendi: {format_piece_name(self.training_piece_name)} em {square_name}. Salvo em {saved_path}"
        self.training_piece_name = None
        print(self.message)

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] != self.last_frame_width:
            self.last_frame_width = frame.shape[1]
            self.piece_buttons = make_piece_buttons(self.last_frame_width)

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

        self.draw_piece_buttons(frame)
        self.draw_piece_tooltip(frame)

        if self.corner_points:
            for index, point in enumerate(self.corner_points, start=1):
                cv2.circle(frame, point, 6, (0, 220, 255), -1)
                cv2.putText(frame, str(index), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        if self.homography is not None:
            self.draw_board_grid(frame)

        if self.selected_square is not None:
            self.draw_selected_square(frame)

        draw_status_bar(frame, self.message)
        return frame

    def draw_piece_buttons(self, frame: np.ndarray) -> None:
        active = self.training_piece_name is not None
        for button in self.piece_buttons:
            selected = button.piece_name == self.training_piece_name
            if button.piece_name == EMPTY_LIGHT_CLASS:
                color = (222, 222, 210) if active else (150, 150, 140)
                text_color = (20, 20, 20)
            elif button.piece_name == EMPTY_DARK_CLASS:
                color = (76, 76, 82) if active else (45, 45, 52)
                text_color = (245, 245, 245)
            elif button.piece_name and button.piece_name.endswith("_branco"):
                color = (228, 228, 220) if active else (130, 130, 126)
                text_color = (18, 18, 18)
            else:
                color = (34, 34, 38) if active else (24, 24, 28)
                text_color = (245, 245, 245)
            border = (0, 255, 255) if selected else ((250, 250, 250) if active else (100, 100, 100))
            cv2.rectangle(frame, (button.x1, button.y1), (button.x2, button.y2), color, -1)
            cv2.rectangle(frame, (button.x1, button.y1), (button.x2, button.y2), border, 2 if selected else 1)
            if button.piece_name in EMPTY_CLASSES:
                cv2.putText(
                    frame,
                    "0",
                    (button.x1 + 14, button.y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.82,
                    text_color,
                    2,
                    cv2.LINE_AA,
                )
            else:
                draw_piece_icon(frame, button, text_color)

    def draw_piece_tooltip(self, frame: np.ndarray) -> None:
        mouse_x, mouse_y = self.mouse_position
        hovered = None
        for button in self.piece_buttons:
            if button.contains(mouse_x, mouse_y):
                hovered = button
                break

        if hovered is None or not hovered.piece_name:
            return

        text = format_piece_name(hovered.piece_name)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        padding_x = 8
        padding_y = 6

        frame_height, frame_width = frame.shape[:2]
        x1 = min(mouse_x + 12, frame_width - text_width - (padding_x * 2) - 4)
        y1 = min(mouse_y + 12, frame_height - text_height - baseline - (padding_y * 2) - 4)
        x1 = max(4, x1)
        y1 = max(4, y1)
        x2 = x1 + text_width + (padding_x * 2)
        y2 = y1 + text_height + baseline + (padding_y * 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (235, 235, 235), 1)
        cv2.putText(
            frame,
            text,
            (x1 + padding_x, y2 - padding_y - baseline),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
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

    def draw_selected_square(self, frame: np.ndarray) -> None:
        file_index, rank_index = self.selected_square
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
        cv2.polylines(frame, [original.reshape(-1, 2)], True, (0, 255, 255), 3)


def identify_piece(crop: np.ndarray) -> tuple[str | None, float]:
    empty_name, empty_score = identify_empty_square(crop)

    samples = [path for path in PIECES_DIR.glob("*/*.png") if path.parent.name not in EMPTY_CLASSES]
    if not samples:
        return None, 0.0

    crop_features = make_match_features(crop)
    scores_by_name: dict[str, list[float]] = {}

    for sample_path in samples:
        sample = cv2.imread(str(sample_path))
        if sample is None:
            continue
        score = compare_piece_images(crop_features, make_match_features(sample))
        scores_by_name.setdefault(sample_path.parent.name, []).append(score)

    if not scores_by_name:
        return None, 0.0

    class_scores = []
    for name, scores in scores_by_name.items():
        top_scores = sorted(scores, reverse=True)[:TOP_SAMPLE_COUNT]
        best = top_scores[0]
        average = sum(top_scores) / len(top_scores)
        class_score = (best * 0.65) + (average * 0.35)
        class_scores.append((class_score, name, len(scores)))

    class_scores.sort(reverse=True)
    best_score, best_name, sample_count = class_scores[0]
    second_score = class_scores[1][0] if len(class_scores) > 1 else 0.0
    margin = best_score - second_score

    if best_score < threshold_for_class(sample_count):
        return None, max(0.0, best_score)
    if len(class_scores) > 1 and margin < UNKNOWN_MARGIN and best_score < 0.70:
        return None, max(0.0, best_score)
    if empty_name is not None and empty_score >= EMPTY_REJECT_THRESHOLD and empty_score > best_score + EMPTY_OVER_PIECE_MARGIN:
        return None, max(0.0, best_score)
    return best_name, best_score


def threshold_for_class(sample_count: int) -> float:
    if sample_count >= 6:
        return 0.50
    if sample_count >= 3:
        return 0.54
    if sample_count == 2:
        return 0.64
    return 0.68


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


def make_piece_buttons(frame_width: int) -> list[Button]:
    pieces = [
        ("P", "peao"),
        ("C", "cavalo"),
        ("B", "bispo"),
        ("T", "torre"),
        ("D", "dama"),
        ("R", "rei"),
    ]
    buttons: list[Button] = []
    button_size = 42
    gap = 8
    top = 72
    left_x = 12
    right_x = max(12, frame_width - button_size - 12)

    for index, (label, piece) in enumerate(pieces):
        y1 = top + index * (button_size + gap)
        y2 = y1 + button_size
        buttons.append(Button(label, left_x, y1, left_x + button_size, y2, f"{piece}_branco"))
        buttons.append(Button(label, right_x, y1, right_x + button_size, y2, f"{piece}_cinza"))

    empty_y1 = top + len(pieces) * (button_size + gap) + 10
    empty_y2 = empty_y1 + button_size
    buttons.append(Button("V", left_x, empty_y1, left_x + button_size, empty_y2, EMPTY_LIGHT_CLASS))
    buttons.append(Button("V", right_x, empty_y1, right_x + button_size, empty_y2, EMPTY_DARK_CLASS))

    return buttons


def is_square_occupied(crop: np.ndarray) -> tuple[bool, float]:
    empty_name, empty_score = identify_empty_square(crop)
    if empty_name is not None:
        return False, max(0.0, 1.0 - empty_score)

    return measure_visual_occupancy(crop)


def measure_visual_occupancy(crop: np.ndarray) -> tuple[bool, float]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    size = gray.shape[0]
    margin = int(size * 0.16)
    center_margin = int(size * 0.26)
    center = gray[center_margin : size - center_margin, center_margin : size - center_margin]

    border_mask = np.ones_like(gray, dtype=np.uint8)
    border_mask[margin : size - margin, margin : size - margin] = 0
    border_pixels = gray[border_mask == 1]

    background_value = float(np.median(border_pixels))
    center_diff = float(np.mean(np.abs(center.astype(np.float32) - background_value)))

    edges = cv2.Canny(gray, 35, 110)
    center_edges = edges[center_margin : size - center_margin, center_margin : size - center_margin]
    edge_density = float(np.count_nonzero(center_edges)) / float(center_edges.size)

    diff_score = min(1.0, center_diff / 60.0)
    edge_score = min(1.0, edge_density / 0.18)
    occupancy_score = (diff_score * 0.55) + (edge_score * 0.45)

    occupied = edge_density >= OCCUPIED_EDGE_THRESHOLD or center_diff >= OCCUPIED_CENTER_DIFF_THRESHOLD
    return occupied, occupancy_score


def identify_empty_square(crop: np.ndarray) -> tuple[str | None, float]:
    samples = [path for empty_class in EMPTY_CLASSES for path in (PIECES_DIR / empty_class).glob("*.png")]
    if not samples:
        return None, 0.0

    crop_features = make_empty_features(crop)
    best_name = None
    best_score = -1.0

    for sample_path in samples:
        sample = cv2.imread(str(sample_path))
        if sample is None:
            continue
        score = compare_signatures(crop_features.copy(), make_empty_features(sample).copy())
        if score > best_score:
            best_score = score
            best_name = sample_path.parent.name

    if best_name is None or best_score < EMPTY_MATCH_THRESHOLD:
        return None, max(0.0, best_score)
    return best_name, best_score


def make_empty_features(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.equalizeHist(gray).astype(np.float32)


def make_match_features(image: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (72, 72), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    normalized = cv2.equalizeHist(gray)
    edges = cv2.Canny(normalized, 30, 100)
    mask = make_piece_mask(gray)
    hu = make_hu_features(mask)

    return {
        "gray": normalized.astype(np.float32),
        "edges": edges.astype(np.float32),
        "mask": mask.astype(np.float32),
        "hu": hu.astype(np.float32),
    }


def make_piece_mask(gray: np.ndarray) -> np.ndarray:
    size = gray.shape[0]
    margin = max(4, int(size * 0.12))
    border_mask = np.ones_like(gray, dtype=np.uint8)
    border_mask[margin : size - margin, margin : size - margin] = 0
    background = float(np.median(gray[border_mask == 1]))

    diff = cv2.absdiff(gray, np.full_like(gray, int(background)))
    _, otsu = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, fixed = cv2.threshold(diff, 16, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(otsu, fixed)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    center_bias = np.zeros_like(mask)
    center_margin = int(size * 0.10)
    center_bias[center_margin : size - center_margin, center_margin : size - center_margin] = 255
    return cv2.bitwise_and(mask, center_bias)


def make_hu_features(mask: np.ndarray) -> np.ndarray:
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)


def compare_piece_images(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    gray_score = compare_signatures(a["gray"].copy(), b["gray"].copy())
    edge_score = compare_signatures(a["edges"].copy(), b["edges"].copy())
    mask_score = compare_signatures(a["mask"].copy(), b["mask"].copy())
    hu_distance = float(np.linalg.norm(a["hu"] - b["hu"]))
    hu_score = 1.0 / (1.0 + hu_distance)

    return (gray_score * 0.38) + (edge_score * 0.22) + (mask_score * 0.28) + (hu_score * 0.12)


def make_signature(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 40, 120)
    signature = cv2.addWeighted(gray, 0.55, edges, 0.45, 0)
    return signature.astype(np.float32)


def compare_signatures(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    a -= float(a.mean())
    b -= float(b.mean())
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if denominator == 0:
        return 0.0
    return (float(np.dot(a, b)) / denominator + 1.0) / 2.0


def save_piece_sample(crop: np.ndarray, piece_name: str) -> Path:
    piece_dir = PIECES_DIR / piece_name
    piece_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{int(time.time() * 1000)}.png"
    path = piece_dir / filename
    cv2.imwrite(str(path), crop)
    return path


def format_piece_name(piece_name: str) -> str:
    if piece_name == EMPTY_LIGHT_CLASS:
        return "casa clara vazia"
    if piece_name == EMPTY_DARK_CLASS:
        return "casa escura vazia"
    return piece_name.replace("_", " ")


def square_to_name(square: tuple[int, int]) -> str:
    file_index, rank_index = square
    return f"{chr(ord('a') + file_index)}{rank_index + 1}"


def draw_status_bar(frame: np.ndarray, text: str) -> None:
    height, width = frame.shape[:2]
    bar_height = 44
    cv2.rectangle(frame, (0, height - bar_height), (width, height), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text[:120],
        (12, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_piece_icon(frame: np.ndarray, button: Button, color: tuple[int, int, int]) -> None:
    if not button.piece_name:
        return

    piece = button.piece_name.split("_")[0]
    x1, y1, x2, y2 = button.x1, button.y1, button.x2, button.y2
    cx = (x1 + x2) // 2
    top = y1 + 8
    bottom = y2 - 7
    base_y = y2 - 11

    cv2.line(frame, (x1 + 11, bottom), (x2 - 11, bottom), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x1 + 14, base_y), (x2 - 14, base_y), color, 2, cv2.LINE_AA)

    if piece == "peao":
        cv2.circle(frame, (cx, top + 5), 6, color, 2, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, top + 19), (9, 12), 0, 0, 360, color, 2, cv2.LINE_AA)
    elif piece == "cavalo":
        points = np.array(
            [
                [cx - 9, bottom - 2],
                [cx - 3, top + 6],
                [cx + 10, top + 10],
                [cx + 4, top + 16],
                [cx + 12, top + 22],
                [cx - 2, top + 25],
                [cx - 10, bottom - 2],
            ],
            dtype=np.int32,
        )
        cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx + 4, top + 13), 1, color, -1, cv2.LINE_AA)
    elif piece == "bispo":
        cv2.ellipse(frame, (cx, top + 15), (9, 16), 0, 0, 360, color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx + 3, top + 6), (cx - 5, top + 20), color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, top + 2), 3, color, 2, cv2.LINE_AA)
    elif piece == "torre":
        cv2.rectangle(frame, (cx - 10, top + 10), (cx + 10, bottom - 3), color, 2)
        for offset in (-8, 0, 8):
            cv2.rectangle(frame, (cx + offset - 3, top + 4), (cx + offset + 3, top + 10), color, -1)
    elif piece == "dama":
        crown = np.array(
            [[cx - 13, top + 21], [cx - 10, top + 5], [cx - 3, top + 17], [cx, top + 4], [cx + 3, top + 17], [cx + 10, top + 5], [cx + 13, top + 21]],
            dtype=np.int32,
        )
        cv2.polylines(frame, [crown], False, color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx - 10, top + 23), (cx + 10, top + 23), color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx - 7, top + 23), (cx - 7, bottom - 4), color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx + 7, top + 23), (cx + 7, bottom - 4), color, 2, cv2.LINE_AA)
    elif piece == "rei":
        cv2.line(frame, (cx, top + 2), (cx, top + 14), color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx - 6, top + 8), (cx + 6, top + 8), color, 2, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, top + 22), (10, 12), 0, 0, 360, color, 2, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Le tabuleiro de xadrez, identifica peca e aprende exemplos.")
    parser.add_argument("--camera", type=int, default=0, help="Indice da camera. Padrao: 0.")
    parser.add_argument("--image", help="Usa uma imagem em vez da camera.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = ChessPieceReader(camera_index=args.camera, image_path=args.image)
    app.run()


if __name__ == "__main__":
    main()
