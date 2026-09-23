# Shoe model licenses

| File | Source | Author | License |
|---|---|---|---|
| `placeholder-shoe-right.usdz`, `placeholder-shoe-left.usdz` | [MaterialsVariantsShoe](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/MaterialsVariantsShoe), glTF Sample Assets | Shopify, for the Khronos Group | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Changes: the default "Street" variant only. The model is normalized by `tools/asset-pipeline/normalize.mjs` (heel at the origin, sole on `y = 0`, toe along `+Z`, length 1) and converted to USDZ by `tools/asset-pipeline/glb2usdz.py`. The left shoe is the right one mirrored across `x`.
