---
name: shoe-try-on
description: The try-on domain itself — the two viewing scenarios and what each one breaks, the normalized shoe convention every layer agrees on, how a foot in the image becomes a shoe standing on the floor, sizing, and left/right. Use when placing or scaling a shoe, when adding or converting a shoe asset, when a shoe sits wrong on the foot, or when reasoning about what "correct" means for this product.
---

# Putting a shoe on a foot

The product is a virtual try-on: a real foot in the camera, a 3D sneaker on it, on-device. What
makes it hard is not rendering — it is that a foot gives away very little about its 3D pose.

## The two scenarios, and what each one breaks

| Scenario | Camera | What breaks |
|---|---|---|
| **Mirror** | 1–2 m away, chest height, pointed at a mirror | Foot is small in frame; solving depth from heel-to-toe length is ill-conditioned; left/right in the image is the opposite of the person's own left/right |
| **On your feet** | 20–40 cm, looking down at your own shoes | The **heel is hidden behind the foot**. The body model puts it on the ankle, and a shoe anchored to a wrong heel lands in the wrong place |

Both are equally in scope. A fix that helps one and breaks the other is not a fix. When judging a
change, say which scenario the evidence came from.

## The floor-plane trick

Do not try to solve depth from the foot's apparent size — the mirror scenario makes that
ill-conditioned, and it was measured as such.

Instead: **a standing foot has every point at a known height above the floor.** With gravity (from
CoreMotion) and an assumed camera height, each image point back-projects onto its own horizontal
plane and becomes a real 3D point. That works from any view, mirror included. `shoePose.ts` does
exactly this.

Consequences worth keeping in mind:

- The **toes anchor** the shoe — they are what must line up in the image — and the ankle, or failing
  that the heel, gives the direction. Not the other way round.
- Camera height is currently **assumed at 1.3 m**, not measured. A wrong guess scales the shoe
  slightly but keeps it on the feet in the image, which is why it is tolerable in a prototype. It
  is an assumption, and any doc or estimate that depends on it must say so.
- Rays near the horizon meet the floor too far away to trust, and are rejected.

## The normalized shoe, which every layer depends on

A shoe model is normalized once, offline, and **every** layer assumes it:

> heel at the origin, sole on `y = 0`, toe along `+Z`, sole length exactly `1`.

The app then scales it to the real size in millimetres. `tools/asset-pipeline/normalize.mjs`
produces this; `modules/shoe-stage` consumes it. If a shoe appears rotated, backwards or lying
down, the fault is almost always at normalization time (`--flip`, `--pre-rotate-x`), not in the
app's pose maths.

- **Left vs right**: one model, mirrored. `glb2usdz.py --mirror` flips `x` and the triangle
  winding. On a normalized *right* shoe the arch cutout and the big-toe bulge are on `+x`.
- **Sizing**: the shoe length in metres is the contract. `SHOE_LENGTH_M = 0.29` (about EU 43) is a
  placeholder until catalog sole lengths land. Size recommendation is an optional, flagged feature,
  not part of placement.
- **Anatomical landmark heights** on the shoe (ankle ≈ 28 % of length up, little toe ≈ 80 % along)
  are **estimates, not measurements**. They are in `shoePose.ts` and labelled there. Treat them as
  open numbers, not constants handed down.

## Naming, which has bitten before

Point names follow **how the foot looks in the image**, not the person's anatomy. In a mirror those
are opposite. Never silently "correct" a side; if a side looks wrong, establish which convention
the evidence is in first.

## Assets

Product photo → cutout → single-image 3D → normalize → USDZ, all offline on the developer's Mac,
documented in `tools/asset-pipeline/README.md`. Two things that matter beyond that README:

- The 3D generator takes **one** image and invents everything it cannot see. Choose the view that
  shows the most of one shoe.
- Catalog photos are retailer images and stay local — `input/` and `work/` are git-ignored. Do not
  commit them, and do not put them anywhere shared.

## What "good" looks like

Currently: the FIND foot template fits real frames to 0.3–2 px in the mirror and 1–3 px from above,
bare or in socks. **Shoes fail** — neither training dataset contains footwear, which is the honest
limit of the current model. A shod-foot failure is a data gap, not a placement bug.
