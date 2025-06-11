import dataclasses
import logging
import socket

import tyro

from hume.models import HumePolicy
from hume.serving import websocket_policy_server


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Port to serve the policy on.
    port: int = 8000

    ckpt_path: str | None = None


def main(args: Args) -> None:
    hume = HumePolicy.from_pretrained(args.ckpt_path).to("cuda")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=hume,
        host="0.0.0.0",
        port=args.port,
        metadata={},
    )
    logging.info("server running")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
