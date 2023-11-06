# see https://stackoverflow.com/a/63831344

import functools
import shutil
from pathlib import Path
import requests
from tqdm.auto import tqdm
import gzip


def gunzip(path):
    path = Path(path)

    assert path.suffix == ".gz"

    new_path = path.with_suffix("")

    with gzip.open(path, "rb") as f_in:
        with open(new_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    path.unlink()
    return new_path


def download(url, path, verbose=False):
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get("Content-Length", 0))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    r.raw.read = functools.partial(
        r.raw.read, decode_content=True
    )  # Decompress if needed

    with tqdm.wrapattr(
        r.raw,
        "read",
        total=file_size,
        disable=not verbose,
        desc=f"Downloading {url}",
    ) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return path