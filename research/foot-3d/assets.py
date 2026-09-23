"""Where the big files live.

Datasets, checkpoints and captured footage are not in git — tens of gigabytes, each restorable from
its source or from a backup. They all sit under one `assets/` directory at the repository root, one
self-contained folder per dataset so a folder can be zipped and backed up on its own. See
`assets/README.md` for what each one is.

Set `SHOE_ASSETS` to work off an external disk without touching any code.
"""

import os
from pathlib import Path

ASSETS = Path(os.environ.get("SHOE_ASSETS") or Path(__file__).resolve().parents[2] / "assets")
