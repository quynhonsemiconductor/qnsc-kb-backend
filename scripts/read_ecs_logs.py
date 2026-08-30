"""Read the ECS task logs out of CloudWatch, through an assumed role.

WHY A SCRIPT AND NOT A PASTED KEY. Credentials pasted into a chat, a ticket or a commit
message live there forever; that is how this project ended up with a key on a rotation
list. This script never accepts a secret. It takes a PROFILE NAME and lets botocore read
`~/.aws/config` itself, so the only place the material exists is the file AWS's own tools
already own.

Role switching is botocore's job, not ours. A profile declaring `role_arn` +
`source_profile` is assumed automatically on first call, the temporary credentials are
cached under `~/.aws/cli/cache`, and `mfa_serial` prompts on stdin when required. So the
whole "I have to switch role to see CloudWatch" problem is solved by configuration:

    # ~/.aws/credentials  — the long-lived user, nothing else
    [sinhhpt]
    aws_access_key_id = ...
    aws_secret_access_key = ...

    # ~/.aws/config  — the role to switch INTO
    [profile qnsc-logs]
    source_profile = sinhhpt
    role_arn = arn:aws:iam::<ACCOUNT_ID>:role/<ROLE_YOU_SWITCH_TO>
    region = ap-southeast-1
    # mfa_serial = arn:aws:iam::<ACCOUNT_ID>:mfa/sinhhpt   # only if the role demands MFA

Then: `python scripts/read_ecs_logs.py --profile qnsc-logs --env develop --since 30m`

Log group names are derived the same way the Terraform does it — `/ecs/${product}-${env}-api`
from infra/modules/stack/main.tf — so they cannot drift from what is actually deployed.

READ-ONLY. Every call is Describe/Filter/Get; nothing here mutates AWS.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PRODUCT = "qnsc-kb"
DEFAULT_REGION = "ap-southeast-1"

#: Mirrors infra/modules/stack/main.tf:64-65 (`api_log_group` / `worker_log_group`).
SERVICES = ("api", "worker")

DURATION_RE = re.compile(r"^(\d+)([smhd])$")
DURATION_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}


def parse_since(value: str) -> datetime:
    """Turn `30m` / `2h` / `1d` into an absolute UTC start time."""
    match = DURATION_RE.match(value.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError(
            f"--since expects a duration such as 30m, 2h or 1d (got {value!r})"
        )
    amount, unit = int(match.group(1)), match.group(2)
    delta = timedelta(**{DURATION_UNITS[unit]: amount})
    return datetime.now(timezone.utc) - delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read qnsc-kb ECS logs from CloudWatch via an assumed role",
        epilog=(
            "Credentials are never passed on the command line. Configure a profile with "
            "role_arn + source_profile in ~/.aws/config and pass its name to --profile."
        ),
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="AWS profile name from ~/.aws/config (the one declaring role_arn)",
    )
    parser.add_argument(
        "--env",
        default="develop",
        choices=("develop", "prod"),
        help="Which deployment's log group to read (default: develop)",
    )
    parser.add_argument(
        "--service",
        default="api",
        choices=SERVICES + ("both",),
        help="api, worker, or both (default: api)",
    )
    parser.add_argument(
        "--since",
        type=parse_since,
        default="30m",
        help="How far back to read, e.g. 30m, 2h, 1d (default: 30m)",
    )
    parser.add_argument(
        "--grep",
        default=None,
        help=(
            "CloudWatch filter pattern. Quote a plain string for substring match, e.g. "
            '--grep \'"API request failed"\''
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=200, help="Max events to print (default: 200)"
    )
    parser.add_argument(
        "--region", default=DEFAULT_REGION, help=f"AWS region (default: {DEFAULT_REGION})"
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Keep polling for new events until interrupted",
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="List the log groups the assumed role can see, then exit",
    )
    return parser.parse_args()


def log_group(env: str, service: str) -> str:
    return f"/ecs/{PRODUCT}-{env}-{service}"


def build_client(profile: str, region: str):
    """Create a CloudWatch Logs client, assuming the profile's role if it declares one."""
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError, ProfileNotFound
    except ModuleNotFoundError:  # pragma: no cover - boto3 is a declared dependency
        sys.exit("boto3 is not installed. Run: pip install boto3")

    try:
        session = boto3.Session(profile_name=profile, region_name=region)
    except ProfileNotFound:
        sys.exit(
            f"No AWS profile named {profile!r}.\n"
            "Add one to ~/.aws/config, for example:\n\n"
            f"  [profile {profile}]\n"
            "  source_profile = sinhhpt\n"
            "  role_arn = arn:aws:iam::<ACCOUNT_ID>:role/<ROLE>\n"
            f"  region = {region}\n"
        )

    client = session.client("logs")
    # Prove the role actually assumed before streaming, so a permissions problem reports
    # itself as a permissions problem rather than as an empty log.
    try:
        identity = session.client("sts").get_caller_identity()
    except (BotoCoreError, ClientError) as exc:
        sys.exit(
            f"Could not authenticate with profile {profile!r}: {exc}\n\n"
            "If the role needs MFA, add mfa_serial to the profile and re-run; botocore "
            "will prompt for the code."
        )
    print(
        f"# assumed: {identity.get('Arn', '?')}\n"
        f"# account: {identity.get('Account', '?')}  region: {region}",
        file=sys.stderr,
    )
    return client


def list_groups(client, region: str) -> int:
    paginator = client.get_paginator("describe_log_groups")
    found = []
    for page in paginator.paginate(logGroupNamePrefix="/ecs/"):
        found.extend(group["logGroupName"] for group in page.get("logGroups", []))
    if not found:
        print("No /ecs/ log groups visible to this role.", file=sys.stderr)
        return 1
    for name in sorted(found):
        print(name)
    return 0


def fetch(client, group: str, start_ms: int, pattern: str | None, limit: int) -> tuple[list, int]:
    """Page through filter_log_events, newest cursor returned for --follow."""
    from botocore.exceptions import ClientError

    kwargs: dict = {"logGroupName": group, "startTime": start_ms}
    if pattern:
        kwargs["filterPattern"] = pattern

    events: list = []
    token: str | None = None
    while len(events) < limit:
        if token:
            kwargs["nextToken"] = token
        try:
            response = client.filter_log_events(**kwargs)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code == "ResourceNotFoundException":
                sys.exit(
                    f"Log group {group!r} does not exist in this account/region.\n"
                    "Run with --list-groups to see what this role can actually read."
                )
            if code in ("AccessDeniedException", "AccessDenied"):
                sys.exit(
                    f"The assumed role cannot read {group!r}.\n"
                    "It needs logs:FilterLogEvents and logs:DescribeLogGroups on it."
                )
            raise
        events.extend(response.get("events", []))
        token = response.get("nextToken")
        if not token:
            break

    events.sort(key=lambda item: item.get("timestamp", 0))
    events = events[:limit]
    cursor = max((item.get("timestamp", 0) for item in events), default=start_ms)
    return events, cursor


def render(events: list, group: str) -> None:
    label = group.rsplit("-", 1)[-1]
    for event in events:
        stamp = datetime.fromtimestamp(
            event.get("timestamp", 0) / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{stamp}  [{label}]  {event.get('message', '').rstrip()}")


def main() -> int:
    args = parse_args()
    client = build_client(args.profile, args.region)

    if args.list_groups:
        return list_groups(client, args.region)

    services = SERVICES if args.service == "both" else (args.service,)
    groups = [log_group(args.env, service) for service in services]
    cursors = {group: int(args.since.timestamp() * 1000) for group in groups}

    total = 0
    for group in groups:
        events, cursors[group] = fetch(
            client, group, cursors[group], args.grep, args.limit
        )
        render(events, group)
        total += len(events)

    if not total:
        print(
            f"# no events in {', '.join(groups)} since "
            f"{args.since.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            + (f" matching {args.grep}" if args.grep else ""),
            file=sys.stderr,
        )

    if not args.follow:
        return 0

    print("# following; Ctrl-C to stop", file=sys.stderr)
    try:
        while True:
            time.sleep(5)
            for group in groups:
                # +1ms so the last printed event is not repeated.
                events, latest = fetch(
                    client, group, cursors[group] + 1, args.grep, args.limit
                )
                if events:
                    render(events, group)
                    cursors[group] = latest
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
