import argparse
import base64
import os
import stat
from pathlib import Path

import paramiko


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?")
    parser.add_argument("--base64", action="store_true")
    parser.add_argument("--upload")
    parser.add_argument("--remote-path")
    parser.add_argument("--download")
    parser.add_argument("--local-path")
    args = parser.parse_args()
    if args.upload and args.download:
        parser.error("--upload and --download are mutually exclusive")
    if not args.command and not args.upload and not args.download:
        parser.error("command, --upload, or --download is required")
    command = base64.b64decode(args.command).decode() if args.base64 and args.command else args.command

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=os.environ["REMOTE_HOST"],
        port=int(os.environ["REMOTE_PORT"]),
        username=os.environ["REMOTE_USER"],
        password=os.environ["REMOTE_PASSWORD"],
        timeout=15,
    )
    if args.upload:
        if not args.remote_path:
            parser.error("--remote-path is required with --upload")
        sftp = client.open_sftp()
        sftp.put(args.upload, args.remote_path)
        sftp.close()
        client.close()
        return
    if args.download:
        if not args.local_path:
            parser.error("--local-path is required with --download")
        destination = Path(args.local_path)
        if destination.exists():
            raise FileExistsError(destination)
        sftp = client.open_sftp()

        def download(remote: str, local: Path) -> None:
            attributes = sftp.stat(remote)
            if stat.S_ISDIR(attributes.st_mode):
                local.mkdir(parents=True)
                for name in sftp.listdir(remote):
                    download(f"{remote.rstrip('/')}/{name}", local / name)
            else:
                local.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote, str(local))

        download(args.download, destination)
        sftp.close()
        client.close()
        return

    _, stdout, stderr = client.exec_command(command)
    stdout.channel.settimeout(None)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    if out:
        print(out, end="")
    if err:
        print(err, end="", file=os.sys.stderr)
    status = stdout.channel.recv_exit_status()
    client.close()
    raise SystemExit(status)


if __name__ == "__main__":
    main()
