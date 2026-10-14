"""Create an isolated venv, including systems without Debian's ensurepip package."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.request import urlopen
import venv


def create(destination):
    destination = Path(destination).absolute()
    venv.EnvBuilder(with_pip=False, symlinks=True).create(destination)
    python = destination / "bin/python"
    if subprocess.run([str(python), "-m", "pip", "--version"], capture_output=True).returncode == 0:
        return
    # PyPI's release JSON supplies the exact wheel's hash, over authenticated HTTPS.
    with urlopen("https://pypi.org/pypi/pip/25.3/json", timeout=60) as response:
        release = json.load(response)
    wheel = next(row for row in release["urls"] if row["filename"].endswith("py3-none-any.whl"))
    with tempfile.TemporaryDirectory(prefix="annoloop-pip-") as temp:
        target = Path(temp) / wheel["filename"]
        with urlopen(wheel["url"], timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != wheel["digests"]["sha256"]:
            raise ValueError("Pip bootstrap checksum mismatch")
        target.write_bytes(content)
        subprocess.run([str(python), "-c",
            "import sys; sys.path.insert(0,sys.argv.pop(1)); from pip._internal.cli.main import main; raise SystemExit(main())",
            str(target), "install", "pip==25.3"], check=True)


if __name__ == "__main__":
    create(sys.argv[1])
