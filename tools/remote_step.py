import argparse
import base64
import os

import paramiko


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--base64", action="store_true")
    parser.add_argument("--upload")
    parser.add_argument("--remote-path")
    args = parser.parse_args()
    command = base64.b64decode(args.command).decode() if args.base64 else args.command

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
