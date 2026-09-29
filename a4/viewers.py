"""Optional text and Tk viewers for Jetan matches."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Mapping

from jetan import BOARD_SIZE, JetanBoard, Move, PieceType, Player


class TerminalViewer:
    def __init__(self, delay: float = 0.0, clear_screen: bool = True) -> None:
        self.delay = delay
        self.clear_screen = clear_screen
        self.names: Mapping[Player, str] = {}

    def start(self, board: JetanBoard, names: Mapping[Player, str]) -> None:
        self.names = names
        self._show(board, "Initial position")

    def update(self, before: JetanBoard, action: Move, after: JetanBoard) -> None:
        self._show(after, f"{self.names[before.turn]} ({before.turn.name}) played {action}")

    def close(self, board: JetanBoard) -> None:
        if board.winner is None:
            message = "Draw"
        else:
            message = f"Winner: {self.names[board.winner]} ({board.winner.name})"
        self._show(board, message, pause=False)

    def _show(self, board: JetanBoard, message: str, pause: bool = True) -> None:
        if self.clear_screen:
            print("\033[2J\033[H", end="")
        print(f"{message}\nPly: {board.ply}\n{board}", flush=True)
        if pause and self.delay:
            time.sleep(self.delay)


class TkViewer:
    """Standard-library graphical viewer loaded only when requested."""

    LIGHT = "#f1d7ad"
    DARK = "#9f563a"
    ORANGE = "#df6b2f"
    BLACK = "#232323"

    def __init__(self, delay: float = 0.25, cell_size: int = 64) -> None:
        if not os.environ.get("DISPLAY"):
            raise RuntimeError("Tk viewing requires a graphical DISPLAY")
        try:
            import tkinter as tk
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Tk viewing requires Tkinter; on Ubuntu, install it with "
                "'sudo apt install python3-tk'"
            ) from error

        self.tk = tk
        self.delay = delay
        self.cell_size = cell_size
        self.margin = 28
        self.root = tk.Tk()
        self.root.title("Jetan")
        width = BOARD_SIZE * cell_size + 2 * self.margin
        self.canvas = tk.Canvas(self.root, width=width, height=width)
        self.canvas.pack()
        self.status = tk.Label(self.root, font=("TkDefaultFont", 12))
        self.status.pack(fill="x", pady=6)
        self.names: Mapping[Player, str] = {}
        asset_dir = Path(__file__).with_name("assets")
        self.piece_images = {}
        for player in Player:
            for kind in PieceType:
                path = asset_dir / f"{player.name.lower()}-{kind.name.lower()}.png"
                try:
                    self.piece_images[player, kind] = tk.PhotoImage(file=str(path))
                except tk.TclError:
                    # Letter markers keep custom installations usable if an
                    # optional artwork file is absent or unsupported by Tk.
                    pass

    def start(self, board: JetanBoard, names: Mapping[Player, str]) -> None:
        self.names = names
        self._draw(board, "Initial position")

    def update(self, before: JetanBoard, action: Move, after: JetanBoard) -> None:
        self._draw(after, f"{self.names[before.turn]} played {action}")

    def close(self, board: JetanBoard) -> None:
        outcome = "Draw" if board.winner is None else f"Winner: {self.names[board.winner]}"
        self._draw(board, outcome, wait=False)

    def _draw(self, board: JetanBoard, message: str, wait: bool = True) -> None:
        self.canvas.delete("all")
        size = self.cell_size
        margin = self.margin
        board_end = margin + BOARD_SIZE * size
        label_font = ("TkDefaultFont", 10, "bold")
        for x in range(BOARD_SIZE):
            label = chr(ord("a") + x)
            center_x = margin + x * size + size / 2
            self.canvas.create_text(center_x, margin / 2, text=label, font=label_font)
            self.canvas.create_text(
                center_x, board_end + margin / 2, text=label, font=label_font
            )
        for y in range(BOARD_SIZE):
            screen_y = BOARD_SIZE - 1 - y
            center_y = margin + screen_y * size + size / 2
            self.canvas.create_text(margin / 2, center_y, text=str(y), font=label_font)
            self.canvas.create_text(
                board_end + margin / 2, center_y, text=str(y), font=label_font
            )
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                screen_y = BOARD_SIZE - 1 - y
                left = margin + x * size
                top = margin + screen_y * size
                fill = self.LIGHT if (x + y) % 2 == 0 else self.DARK
                self.canvas.create_rectangle(
                    left,
                    top,
                    left + size,
                    top + size,
                    fill=fill,
                    outline="#5b3326",
                )
                piece = board.at((x, y))
                if piece is not None:
                    center = (left + size / 2, top + size / 2)
                    image = self.piece_images.get((piece.player, piece.kind))
                    if image is not None:
                        self.canvas.create_image(*center, image=image)
                    else:
                        color = (
                            self.ORANGE
                            if piece.player is Player.ORANGE
                            else self.BLACK
                        )
                        self.canvas.create_oval(
                            left + 7,
                            top + 7,
                            left + size - 7,
                            top + size - 7,
                            fill=color,
                            outline="white",
                            width=2,
                        )
                        self.canvas.create_text(
                            *center,
                            text=piece.kind.value.upper(),
                            fill="white",
                            font=("TkDefaultFont", max(12, size // 3), "bold"),
                        )
        self.status.config(text=f"{message} | ply {board.ply}")
        self.root.update_idletasks()
        self.root.update()
        if wait and self.delay:
            time.sleep(self.delay)
