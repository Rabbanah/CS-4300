"""Extract transparent piece sprites from the Wikimedia Jetan board image.

Regeneration requires Pillow. From the starter directory, run:

    uv run --with pillow python assets/generate_piece_assets.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ASSET_DIR = Path(__file__).parent
SOURCE = ASSET_DIR / "Jetan_Board-960.png"
CELL_SIZE = 96
OUTPUT_SIZE = 56

BACK_RANK = {
    "warrior": 0,
    "padwar": 1,
    "dwar": 2,
    "flier": 3,
    "chief": 4,
    "princess": 5,
}
FRONT_RANK = {"thoat": 0, "panthan": 1}


def extract(source: Image.Image, player: str, kind: str, x: int, y: int) -> None:
    box = (
        x * CELL_SIZE,
        y * CELL_SIZE,
        (x + 1) * CELL_SIZE,
        (y + 1) * CELL_SIZE,
    )
    sprite = source.crop(box).convert("RGBA")
    for corner in ((0, 0), (95, 0), (0, 95), (95, 95)):
        ImageDraw.floodfill(sprite, corner, (0, 0, 0, 0), thresh=12)
    bounds = sprite.getbbox()
    if bounds is None:
        raise RuntimeError(f"no artwork found for {player} {kind}")
    sprite = sprite.crop(bounds)
    sprite.thumbnail((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS)
    sprite = ImageOps.pad(sprite, (OUTPUT_SIZE, OUTPUT_SIZE), color=(0, 0, 0, 0))
    sprite.save(ASSET_DIR / f"{player}-{kind}.png")


def main() -> None:
    source = Image.open(SOURCE)
    for player, back_y, front_y in (
        ("orange", 0, 1),
        ("black", 9, 8),
    ):
        for kind, x in BACK_RANK.items():
            extract(source, player, kind, x, back_y)
        for kind, x in FRONT_RANK.items():
            extract(source, player, kind, x, front_y)


if __name__ == "__main__":
    main()
