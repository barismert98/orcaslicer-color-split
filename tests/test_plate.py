"""The message that hands a file to OrcaSlicer must parse the way OrcaSlicer parses it."""
import pytest

from color_split import plate


@pytest.mark.parametrize("path", [
    r"C:\Users\x\AppData\Roaming\OrcaSlicer\plugin_data\out\cube_split.3mf",
    r"C:\a b\c;d\"e\ğü.3mf",
])
def test_command_line_parses_back_to_the_path(path, monkeypatch):
    monkeypatch.setattr(plate.Path, "resolve", lambda self: self)
    words = plate.unescape_strings_cstyle(plate.command_line([path]))
    assert words == ["orca-slicer", path]


def test_space_separated_words_are_rejected_like_orcaslicer_does():
    # the first version sent this, and OrcaSlicer logged "incorrectly formatted"
    assert plate.unescape_strings_cstyle('"orca-slicer" "C:/x.3mf"') is None
