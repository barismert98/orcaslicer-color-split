# Color Split

Turn a color-painted model into separate printable parts, one per color. Print each color
on its own and glue or press-fit the parts together. You need no multi-material unit, and
there is no purge waste.

## How to use
1. Paint the model with OrcaSlicer's color painting tool.
2. Open the **Color Split** tab, pick the object and press **Split**.
3. The parts land on the plate as one object, each part on its own filament. They are
   unsaved: save the project to keep them, or undo.

## What it does
- Painted features come off along the edge they stand on, such as a chimney on a roof or a
  raised logo. The cut follows the surface under the feature, so the parts meet exactly.
- A shell painted on the outside (a boat's cabin, its doors and windows left bare) comes off
  whole at its foot, inside walls and frames included.
- Paint that winds around the model (a ball sunk into a joint) is cut along a surface
  stretched across its border, so the part always fits back on.
- Colors that touch each other come off as one piece and are split between the colors.
- Sloppy paint is fine: paint up to 0.5 mm over or short of an edge snaps to it, and stray
  specks are cleaned up.
- Every part is a closed mesh, and "Split to objects" gives no loose fragments.

## Surface paint
Painted regions that wrap a shape (a sleeve, an eye) are cut along their paint border and the
openings closed. For paint on a smooth surface, you can choose:
- **Cut** it along its border (the default); paint that encloses no volume stays painted;
- **Raise** it as a layer on top, 0.2 to 5 mm thick;
- **Inlay** it into the base, flush. Paint inside an engraving fills it.

## Options
**Buried parts, take out whole** (off): a ball sunk past its middle comes out whole, and the
base is cut through it so the ball fits between the two halves of its socket.

## Notes
- Windows is the tested platform. Adding the parts to the plate is Windows only; elsewhere,
  import `<object>_split.3mf` from the plugin's folder with File → Import.
- The plugin installs numpy, manifold3d and scipy. It asks once for permission to start a
  background thread.
- License: GPL-3.0-or-later.
