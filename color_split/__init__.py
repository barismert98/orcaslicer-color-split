# SPDX-License-Identifier: GPL-3.0-or-later
"""Split a colour-painted OrcaSlicer object into per-colour inlays and a pocketed base."""
try:
    import orca  # noqa: F401  (present only inside OrcaSlicer)
except ImportError:
    orca = None
if orca is not None:
    from . import plugin  # noqa: F401  (registers @orca.plugin during import)
