"""Probe outbound restrictions in a Prime sandbox against an open control.

    uv run --no-sync python scripts/midog-mitosis/egress_probe.py

Two VM sandboxes (a fraction of a cent) run the same probes from inside. The restricted one gets exactly the policy a
rollout gets (``allow = []``, applied after setup); the open one is the control that shows
each probe can succeed. A probe *leaks* when it gets data back from the internet in the restricted sandbox. A TCP
handshake is not data: a transparent proxy in front of the sandbox accepts connections to any address and then drops
what it does not allow, so every probe here sends bytes and waits for a reply.

Exit code 1 if a non-exempt probe leaks or the model route is not reachable. Resolver DNS is reported separately
and excluded from the exit status. DNS can carry data, so a successful run does not establish complete isolation.
Needs the package installed and a Prime API key (``PRIME_API_KEY`` or ``prime login``).
"""

import asyncio
import sys

from verifiers.v1.runtimes import PrimeConfig, provision_runtime

ROUTE = "https://api.pinference.ai/v1"
"""Stands in for the interception tunnel that a rollout leaves open."""

TLS = """
import socket, ssl
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
def get(host, port, sni, label):
    try:
        s = socket.create_connection((host, port), 6)
        t = ctx.wrap_socket(s, server_hostname=sni)
        t.sendall(('GET / HTTP/1.1\\r\\nHost: %s\\r\\nConnection: close\\r\\n\\r\\n' % (sni or host)).encode())
        print('DATA', label, t.recv(40))
    except Exception as e:
        print('blocked', label, type(e).__name__)
"""

UDP = """
import socket
def udp(host, port, payload, label):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(4)
    try:
        s.sendto(payload, (host, port)); data, _ = s.recvfrom(1024); print('DATA', label, len(data), 'bytes')
    except Exception as e:
        print('blocked', label, type(e).__name__)
QUERY = b'\\x12\\x34\\x01\\x00\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x07example\\x03com\\x00\\x00\\x01\\x00\\x01'
"""

# name -> (kind, code): "block" probes must not get data in the restricted sandbox, "open" probes must.
PROBES = {
    "https to huggingface.co, figshare.com, example.com": (
        "block",
        TLS + "for h in ('huggingface.co', 'figshare.com', 'example.com'): get(h, 443, h, h)",
    ),
    "http (port 80) to huggingface.co, example.com": (
        "block",
        """
import urllib.request as u
for url in ('http://huggingface.co', 'http://example.com'):
    try:
        print('DATA', url, u.urlopen(url, timeout=6).status)
    except Exception as e:
        print('blocked', url, type(e).__name__)
""",
    ),
    "tls without SNI, to an IP": ("block", TLS + "get('1.1.1.1', 443, None, 'no SNI')"),
    "tls with an unapproved SNI, to an IP": (
        "block",
        TLS + "get('1.1.1.1', 443, 'one.one.one.one', 'SNI one.one.one.one')",
    ),
    "raw http to an IP (ports 80, 8080)": (
        "block",
        """
import socket
for port in (80, 8080):
    try:
        s = socket.create_connection(('1.1.1.1', port), 5); s.settimeout(5)
        s.sendall(b'GET / HTTP/1.1\\r\\nHost: one.one.one.one\\r\\nConnection: close\\r\\n\\r\\n')
        data = s.recv(40)
        print('DATA' if data else 'blocked', port, data)
    except Exception as e:
        print('blocked', port, type(e).__name__)
""",
    ),
    "service banners (github.com:22 ssh)": (
        "block",
        """
import socket
try:
    s = socket.create_connection(('github.com', 22), 6); s.settimeout(6); data = s.recv(40)
    print('DATA' if data else 'blocked', data)
except Exception as e:
    print('blocked', type(e).__name__)
""",
    ),
    "dns over tcp (8.8.8.8:53)": (
        "block",
        """
import socket, struct
q = b'\\x12\\x34\\x01\\x00\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x07example\\x03com\\x00\\x00\\x01\\x00\\x01'
try:
    s = socket.create_connection(('8.8.8.8', 53), 5); s.settimeout(5); s.sendall(struct.pack('!H', len(q)) + q)
    data = s.recv(512); print('DATA' if data else 'blocked', len(data), 'bytes')
except Exception as e:
    print('blocked', type(e).__name__)
""",
    ),
    "udp: ntp, dns to other resolvers": (
        "block",
        UDP + "udp('time.google.com', 123, b'\\x1b' + 47 * b'\\0', 'ntp')\n"
        "for server in ('9.9.9.9', '1.1.1.1', '8.8.4.4'): udp(server, 53, QUERY, 'dns ' + server)",
    ),
    "ipv6": (
        "block",
        """
import socket
try:
    socket.create_connection(('2606:4700:4700::1111', 443), 5).close(); print('DATA connected')
except Exception as e:
    print('blocked', type(e).__name__)
""",
    ),
    "pip install": (
        "block",
        """
import subprocess
r = subprocess.run(['python3', '-m', 'pip', 'download', '--no-deps', '-d', '/tmp/p', 'six'], capture_output=True, text=True, timeout=60)
print('DATA' if r.returncode == 0 else 'blocked', 'pip exit', r.returncode)
""",
    ),
    "model route (must stay open)": (
        "open",
        """
import urllib.error, urllib.request as u
try:
    u.urlopen('https://api.pinference.ai/api/v1/models', timeout=8); print('DATA reachable')
except urllib.error.HTTPError as e:
    print('DATA reachable, HTTP', e.code)
except Exception as e:
    print('blocked', type(e).__name__)
""",
    ),
    "udp dns via the sandbox resolver (known open)": (
        "known",
        UDP + "udp(open('/etc/resolv.conf').read().split()[1], 53, QUERY, 'resolver')",
    ),
}


async def run_probes(config: PrimeConfig, seal: bool) -> dict[str, str]:
    results = {}
    async with provision_runtime(config) as box:
        await box.run(["python3", "-c", "print('up')"], {})
        if seal:
            await box.prepare_execution([ROUTE])
        for name, (_, code) in PROBES.items():
            done = await box.run(["python3", "-c", code], {})
            results[name] = (done.stdout + done.stderr).strip()
    return results


def state(output: str) -> str:
    return "data" if any(line.startswith("DATA") for line in output.splitlines()) else "blocked"


async def main() -> int:
    sealed, control = await asyncio.gather(
        run_probes(PrimeConfig(allow=[], block=["*"]), seal=True), run_probes(PrimeConfig(), seal=False)
    )
    failures = 0
    print(f"{'probe':52s} {'restricted':11s} {'open control':12s} verdict")
    for name, (kind, _) in PROBES.items():
        inside, outside = state(sealed[name]), state(control[name])
        if kind == "block":
            verdict = (
                "LEAK" if inside == "data" else ("ok" if outside == "data" else "inconclusive (control also blocked)")
            )
        elif kind == "open":
            verdict = "ok" if inside == "data" else "BROKEN (the model route is unreachable)"
        else:
            verdict = "open; DNS excluded from pass/fail" if inside == "data" else "closed"
        failures += verdict.startswith(("LEAK", "BROKEN"))
        print(f"{name:52s} {inside:11s} {outside:12s} {verdict}")
        if verdict.startswith(("LEAK", "BROKEN")):
            print("   ", sealed[name].replace("\n", "\n    "))
    print(
        f"\n{failures} leaks"
        if failures
        else "\nno non-exempt probe fetched data; resolver DNS is excluded from this pass/fail result"
    )
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
