# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from __future__ import annotations

from pathlib import Path
from typing import List

import array_record as ar


def load_brain_ar(path: Path | str) -> List[dict]:
    """Load brain MRI data from an Array Record file.

    Args:
        path: Path to the brain .ar file

    Returns:
        List of dictionaries containing brain MRI data
    """
    records = []
    reader = ar.ArrayRecordReader(str(path))

    for i in range(len(reader)):
        record = reader[i]
        # Parse the record - adjust based on actual structure
        records.append({"data": record, "index": i})

    return records


__all__ = ["load_brain_ar"]
