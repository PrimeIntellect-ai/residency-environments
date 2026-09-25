"""Verify the installed stock Verifiers bash harness can start without network access."""

import argparse
import hashlib
import json
import subprocess
import time
import uuid

from verifiers.v1.harnesses.bash.harness import PROGRAM_SOURCE
from verifiers.v1.runtimes.base import _ENSURE_UV


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="physim-agent:0.12.2")
    args = parser.parse_args()
    program_hash = hashlib.sha256(PROGRAM_SOURCE.encode()).hexdigest()
    path = f"/tmp/vf-scripts/{program_hash}.py"
    script = """import json,os,subprocess,sys
from pathlib import Path
data=json.load(sys.stdin)
assert os.environ.get('UV_OFFLINE') == 'true'
assert os.environ.get('PIP_NO_INDEX') == '1'
path=Path(data['path']);path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(data['program'])
subprocess.run(['sh','-c','set -e; '+data['ensure']+'; uv sync --script '+str(path)+' --no-config'],check=True)
python=subprocess.check_output(['uv','python','find','--script',str(path),'--no-config'],text=True).strip()
subprocess.run([python,str(path),'--help'],check=True,stdout=subprocess.DEVNULL)
"""
    start = time.monotonic()
    name = "physim-harness-check-" + uuid.uuid4().hex[:12]
    try:
        subprocess.run(
            ["docker", "run", "--rm", "--name", name, "--network", "none", "-i", args.image, "python", "-c", script],
            input=json.dumps(dict(program=PROGRAM_SOURCE, path=path, ensure=_ENSURE_UV)),
            text=True,
            check=True,
            timeout=60,
        )
    finally:
        subprocess.run(["docker", "rm", "--force", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(
        json.dumps(
            dict(
                ok=True, image=args.image, network="none", program_sha256=program_hash, seconds=time.monotonic() - start
            )
        )
    )


if __name__ == "__main__":
    main()
