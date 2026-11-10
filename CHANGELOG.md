# Changelog

## 1.3.1 (2026-11-10)
- American English throughout the tab, messages and documents. Nothing else changed.

## 1.3.0 (2026-11-09)
Published on Orca Cloud (review: Low risk). The user tried it in OrcaSlicer and liked it.

- New tab design: one row per color found on the model, with its real filament color and its
  measured share of the surface, next to what Color Split will try. After the split the same row
  says what came out: the part written, or that the color stayed painted.
- The tab reads the object's colors before the split, so you see them before pressing Split.
- The finished view leads with the next step; settings fold away after a split.
- The tab icon is the slid-disc logo.
- Fix: a failed read no longer leaves the temporary project locked on Windows.

## 1.2.0 (2026-11-08)
- New default: painted regions on smooth surfaces (sleeves, trousers, eyes) are cut along their
  paint border and the openings closed, instead of staying on the base as surface paint. Paint
  that encloses no volume (a flat patch) or whose border does not close stays on the base.
- A color too thin to be a part stays painted on the part around it, with a note.

## 1.1.1 (2026-11-08)
- Fix: paint that stays on the base (surface paint) was lost in the split project, so the copy
  added to the plate came out in one color. The base now keeps it, brush detail included.
- Fix: models with heavily subdivided paint ran out of memory while the mesh was prepared; that
  step now uses far less memory and is much faster.
- The wheel's METADATA has LF line endings, so OrcaSlicer stores a clean name and version.

## 1.1.0 (2026-11-08)
- New: a shell painted on the outside (a boat's cabin with bare door and window frames) comes
  off whole at its foot, with its inside walls and frames, instead of staying on the base as
  surface paint.
- A base-color part standing on a cut-off part (a chimney on the cabin roof) comes off as a
  part of its own.
- Paint on the inside of a cut-off part is no longer reported as surface paint left on the base.
- Known limit: when every edge of such a shell is sharp, a bare frame standing proud of its
  wall can stay on the base.

## 1.0.1 (2026-11-07)
- Fix: with Raise or Inlay, thin color layers that touch along an edge (a Benchy cabin painted
  outside) failed to export ("open or shared edges would remain"). The export now separates
  such copies until the STL is watertight.

## 1.0.0 (2026-11-07)
First public release.

- **Color Split tab**, next to Prepare and Preview. It has the object, surface paint and
  buried-part options, progress with cancel, and a result panel with each part's filament,
  pieces and size. The same split is in the Plugins menu as a dialog.
- **Parts added to the plate** as one object, each part on its own filament, unsaved
  (Windows). Elsewhere, or if that is not available, import `<object>_split.3mf` with Ctrl+I.
- **Feature cut**: a painted feature comes off along the sharp edge it stands on. The cap
  follows the base surface under it, however curved.
- **Film cut**: paint that winds around the model is cut along a surface stretched across its
  border; the part is never trapped.
- **Touching colors** come off as one feature and are split between the colors.
- **Separately modeled bodies** painted all over are taken as they are.
- **Surface paint**: keep painted, raise, or inlay, 0.2 to 5 mm. Painted engravings are filled
  flush.
- **Buried parts** (option): a ball sunk past its middle comes out whole, and the base is cut
  through it.
- **Robust to real paint and meshes**: paint snaps to edges within 0.5 mm, specks are cleaned,
  small mesh defects are repaired, and parts keep their own filaments.
- **Output**: watertight parts with no loose fragments, plus STLs and a project 3MF in the
  plugin's folder.
