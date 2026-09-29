"""Checks for locally available, attributed graphical-viewer assets."""

import unittest
from pathlib import Path

from jetan import PieceType, Player


class PieceAssetTests(unittest.TestCase):
    def test_every_player_and_piece_type_has_a_png_sprite(self) -> None:
        asset_dir = Path(__file__).with_name("assets")

        for player in Player:
            for kind in PieceType:
                path = asset_dir / f"{player.name.lower()}-{kind.name.lower()}.png"
                with self.subTest(player=player, kind=kind):
                    self.assertTrue(path.is_file())
                    self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_artwork_attribution_is_included(self) -> None:
        license_text = Path(__file__).with_name("assets").joinpath("LICENSE.md")

        self.assertIn("Ninjatacoshell", license_text.read_text())
        self.assertIn("CC BY-SA 3.0", license_text.read_text())


if __name__ == "__main__":
    unittest.main()
