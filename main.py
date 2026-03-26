"""Run the SSOTSyncDevices (Sync Network Devices) SSoT job via pynautobot.

Reads device data from network_devices.csv. Rows sharing the same non-IP
fields are grouped into job runs with comma-separated IP addresses.  Large
groups are further split into chunks of BATCH_SIZE (default 100) to handle
inventories of thousands of devices.

After each successful batch the script applies post-processing to every
onboarded device:
- Applies the "Weekly Backup" tag (creates it when absent).
- Copies the location's tenant to the device (when the location has one).
- Copies location tags that carry the ``dcim.device`` content-type to the
  device (skips tags already present on the device).

Required env vars: NAUTOBOT_URL, NAUTOBOT_TOKEN
"""

import csv
import logging
import os
import time
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pynautobot
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

NAUTOBOT_URL = os.environ["NAUTOBOT_URL"]
NAUTOBOT_TOKEN = os.environ["NAUTOBOT_TOKEN"]
CSV_PATH = os.environ.get("CSV_PATH", "network_devices.csv")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "100"))

JOB_NAME = "SSOTSyncDevices"
WEEKLY_BACKUP_TAG_NAME = "Weekly Backup"
DEVICE_CONTENT_TYPE = "dcim.device"


@dataclass(frozen=True)
class JobKey:
    """Fields that must be identical for IPs to be batched into one job run.

    Rows in the CSV that share the same values for all these fields will be
    combined into a single SSOTSyncDevices job call with multiple IP addresses.
    """

    location_name: str
    location_parent_name: str
    namespace: str
    device_role_name: str
    device_status_name: str
    interface_status_name: str
    ip_address_status_name: str
    secrets_group_name: str
    platform_name: str
    port: int
    timeout: int
    set_mgmt_only: bool
    update_devices_without_primary_ip: bool


def _parse_bool(value: str) -> bool:
    """Return True if value is a truthy string ("true", "1", or "yes").

    Args:
        value: Raw string from CSV cell.

    Returns:
        bool: Parsed boolean value.
    """
    return value.strip().lower() in {"true", "1", "yes"}


def _chunks(items: list[str], size: int) -> Generator[list[str], None, None]:
    """Yield successive fixed-size chunks from a list.

    Args:
        items: List to split.
        size: Maximum number of elements per chunk.

    Yields:
        list[str]: Successive slices of ``items`` of at most ``size`` elements.
    """
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _get_uuid(obj: object, label: str) -> str:
    """Extract the UUID string from a pynautobot Record.

    Args:
        obj: The result of a pynautobot ``.get()`` call.
        label: Human-readable description used in error messages.

    Returns:
        str: UUID of the matched Nautobot object.

    Raises:
        RuntimeError: If the lookup returned no results.
        TypeError: If the lookup returned multiple results.
    """
    if obj is None:
        msg = f"Nautobot returned no match for: {label}"
        raise RuntimeError(msg)
    if isinstance(obj, list):
        msg = f"Nautobot returned multiple matches for: {label} — use a UUID directly"
        raise TypeError(msg)
    return str(obj.id)  # type: ignore[union-attr]


def _resolve_uuids(nb: pynautobot.api, key: JobKey) -> dict[str, object]:  # type: ignore[valid-type]
    """Resolve all name-based CSV fields to Nautobot UUIDs.

    Args:
        nb: Authenticated pynautobot API client.
        key: JobKey containing the human-readable names from the CSV row.

    Returns:
        dict[str, object]: Mapping of job data field names to resolved UUIDs
        and scalar values ready to pass to the job's ``data`` payload.
    """
    location_filter: dict[str, str] = {"name": key.location_name}
    if key.location_parent_name:
        location_filter["parent__name"] = key.location_parent_name
    location_uuid = _get_uuid(
        nb.dcim.locations.get(**location_filter),
        f"location '{key.location_name}' (parent: '{key.location_parent_name}')",
    )

    namespace_uuid = _get_uuid(
        nb.ipam.namespaces.get(name=key.namespace),
        f"namespace '{key.namespace}'",
    )

    device_role_uuid = _get_uuid(
        nb.extras.roles.get(name=key.device_role_name),
        f"device_role '{key.device_role_name}'",
    )

    device_status_uuid = _get_uuid(
        nb.extras.statuses.get(name=key.device_status_name),
        f"device_status '{key.device_status_name}'",
    )

    interface_status_uuid = _get_uuid(
        nb.extras.statuses.get(name=key.interface_status_name),
        f"interface_status '{key.interface_status_name}'",
    )

    ip_address_status_uuid = _get_uuid(
        nb.extras.statuses.get(name=key.ip_address_status_name),
        f"ip_address_status '{key.ip_address_status_name}'",
    )

    secrets_group_uuid = _get_uuid(
        nb.extras.secrets_groups.get(name=key.secrets_group_name),
        f"secrets_group '{key.secrets_group_name}'",
    )

    resolved: dict[str, object] = {
        "location": location_uuid,
        "namespace": namespace_uuid,
        "device_role": device_role_uuid,
        "device_status": device_status_uuid,
        "interface_status": interface_status_uuid,
        "ip_address_status": ip_address_status_uuid,
        "secrets_group": secrets_group_uuid,
        "port": key.port,
        "timeout": key.timeout,
        "set_mgmt_only": key.set_mgmt_only,
        "update_devices_without_primary_ip": key.update_devices_without_primary_ip,
    }

    if key.platform_name:
        resolved["platform"] = _get_uuid(
            nb.dcim.platforms.get(name=key.platform_name),
            f"platform '{key.platform_name}'",
        )

    return resolved


def _wait_for_job(nb: pynautobot.api, job_result_id: str) -> str:  # type: ignore[valid-type]
    """Poll a job result until it reaches a terminal status.

    Args:
        nb: Authenticated pynautobot API client.
        job_result_id: UUID string of the job result to poll.

    Returns:
        str: Final status string (e.g. ``"completed"``, ``"failed"``).

    Raises:
        RuntimeError: If the job result cannot be retrieved.
    """
    poll_interval = 5
    terminal_statuses = {"completed", "failed", "errored"}
    status = "pending"

    while True:
        job_result = nb.extras.job_results.get(job_result_id)
        if job_result is None or isinstance(job_result, list):
            msg = f"Could not retrieve job result id={job_result_id}"
            raise RuntimeError(msg)
        raw = job_result.status  # type: ignore[union-attr]
        status = raw.value if hasattr(raw, "value") else str(raw)
        log.info("    status: %s", status)
        if status.lower() in terminal_statuses:
            break
        time.sleep(poll_interval)

    return status


def _ensure_weekly_backup_tag(nb: pynautobot.api) -> str:  # type: ignore[valid-type]
    """Return the UUID of the 'Weekly Backup' tag, creating it when absent.

    The tag is created with the ``dcim.device`` content-type so it can be
    applied to device objects.

    Args:
        nb: Authenticated pynautobot API client.

    Returns:
        str: UUID of the 'Weekly Backup' tag.
    """
    tag = nb.extras.tags.get(name=WEEKLY_BACKUP_TAG_NAME)
    if tag is None:
        log.info("Tag '%s' not found — creating it.", WEEKLY_BACKUP_TAG_NAME)
        tag = nb.extras.tags.create(
            name=WEEKLY_BACKUP_TAG_NAME,
            content_types=[DEVICE_CONTENT_TYPE],
        )
    return str(tag.id)  # type: ignore[union-attr]


def _fetch_device_tag_ids(nb: pynautobot.api) -> set[str]:  # type: ignore[valid-type]
    """Return the set of tag UUIDs that carry the ``dcim.device`` content-type.

    Fetched once at startup and passed down to post-processing so that
    per-device logic can check content-types with a simple set lookup instead
    of issuing extra API calls.

    Args:
        nb: Authenticated pynautobot API client.

    Returns:
        set[str]: UUIDs of all tags applicable to device objects.
    """
    return {str(t.id) for t in nb.extras.tags.filter(content_types=DEVICE_CONTENT_TYPE)}


def _get_location_context(
    nb: pynautobot.api,  # type: ignore[valid-type]
    location_uuid: str,
    device_tag_ids: set[str],
) -> tuple[str | None, list[str]]:
    """Return the tenant UUID and device-applicable tag UUIDs for a location.

    Args:
        nb: Authenticated pynautobot API client.
        location_uuid: UUID of the location to inspect.
        device_tag_ids: Pre-fetched set of tag UUIDs with the ``dcim.device``
            content-type (used to filter location tags efficiently).

    Returns:
        tuple[str | None, list[str]]: A 2-tuple of ``(tenant_uuid, tag_uuids)``
        where ``tenant_uuid`` is ``None`` when the location has no tenant and
        ``tag_uuids`` contains only tags valid for device objects.
    """
    location = nb.dcim.locations.get(location_uuid)
    if location is None or isinstance(location, list):
        return None, []

    tenant_id: str | None = None
    if getattr(location, "tenant", None):
        tenant_id = str(location.tenant.id)  # type: ignore[union-attr]

    location_tags: list[str] = []
    for tag in getattr(location, "tags", None) or []:
        tag_id = str(tag.id)
        if tag_id in device_tag_ids:
            location_tags.append(tag_id)

    return tenant_id, location_tags


def _post_process_devices(
    nb: pynautobot.api,  # type: ignore[valid-type]
    ips: list[str],
    location_uuid: str,
    weekly_backup_tag_id: str,
    device_tag_ids: set[str],
) -> None:
    """Apply tags and tenant to devices that were just onboarded.

    For each device found by primary IP the function:

    - Adds the "Weekly Backup" tag.
    - Sets the location's tenant when the device has none and the location
      has a tenant.
    - Adds location tags that carry the ``dcim.device`` content-type and are
      not already present on the device.

    Args:
        nb: Authenticated pynautobot API client.
        ips: IP addresses of the newly onboarded devices.
        location_uuid: UUID of the shared location for this batch.
        weekly_backup_tag_id: UUID of the 'Weekly Backup' tag.
        device_tag_ids: Pre-fetched set of tag UUIDs applicable to devices.
    """
    tenant_id, location_tag_ids = _get_location_context(nb, location_uuid, device_tag_ids)

    if tenant_id:
        log.info("  location tenant: %s", tenant_id)
    else:
        log.info("  location has no tenant — skipping tenant assignment")

    if location_tag_ids:
        log.info("  location device-applicable tags: %s", location_tag_ids)
    else:
        log.info("  location has no device-applicable tags — skipping tag copy")

    # Fetch devices in sub-chunks to avoid overly long query strings
    for sub_chunk in _chunks(ips, 50):
        devices = list(nb.dcim.devices.filter(primary_ip4__address=sub_chunk))
        for device in devices:
            current_tag_ids = {str(t.id) for t in (getattr(device, "tags", None) or [])}

            tags_to_add: set[str] = {weekly_backup_tag_id}

            for loc_tag_id in location_tag_ids:
                if loc_tag_id not in current_tag_ids:
                    tags_to_add.add(loc_tag_id)

            new_tag_ids = list(current_tag_ids | tags_to_add)
            update_payload: dict[str, object] = {"tags": new_tag_ids}

            if tenant_id and not getattr(device, "tenant", None):
                update_payload["tenant"] = tenant_id

            device.update(update_payload)  # type: ignore[union-attr]
            log.info(
                "    updated device %s: +tags=%s tenant=%s",
                getattr(device, "name", device),
                tags_to_add,
                update_payload.get("tenant", "(unchanged)"),
            )


def read_csv(path: str) -> dict[JobKey, list[str]]:
    """Parse the device CSV into groups keyed by shared job parameters.

    Rows that share identical values for all non-IP fields are merged into
    one group so their IP addresses can be submitted in a single job run.

    Args:
        path: Filesystem path to the CSV file.

    Returns:
        dict[JobKey, list[str]]: Mapping of job parameter sets to lists of
        IP address strings.
    """
    groups: dict[JobKey, list[str]] = defaultdict(list)

    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = JobKey(
                location_name=row["location_name"].strip(),
                location_parent_name=row.get("location_parent_name", "").strip(),
                namespace=row["namespace"].strip(),
                device_role_name=row["device_role_name"].strip(),
                device_status_name=row["device_status_name"].strip(),
                interface_status_name=row["interface_status_name"].strip(),
                ip_address_status_name=row["ip_address_status_name"].strip(),
                secrets_group_name=row["secrets_group_name"].strip(),
                platform_name=row.get("platform_name", "").strip(),
                port=int(row.get("port", "22").strip()),
                timeout=int(row.get("timeout", "30").strip()),
                set_mgmt_only=_parse_bool(row.get("set_mgmt_only", "true")),
                update_devices_without_primary_ip=_parse_bool(
                    row.get("update_devices_without_primary_ip", "false")
                ),
            )
            groups[key].append(row["ip_address_host"].strip())

    return groups


def run_sync_devices_job() -> None:
    """Load the CSV and trigger one SSOTSyncDevices job per unique parameter group.

    Large IP groups are split into batches of ``BATCH_SIZE`` (default 100).
    After each successful batch the onboarded devices are post-processed:
    the "Weekly Backup" tag is applied, the location's tenant is copied to the
    device (if available), and the location's device-applicable tags are added.

    Raises:
        RuntimeError: If the SSOTSyncDevices job is not found in Nautobot.
    """
    nb = pynautobot.api(NAUTOBOT_URL, token=NAUTOBOT_TOKEN)

    groups = read_csv(CSV_PATH)
    total_ips = sum(len(v) for v in groups.values())
    log.info("Loaded %s: %d IPs in %d job group(s)\n", CSV_PATH, total_ips, len(groups))

    weekly_backup_tag_id = _ensure_weekly_backup_tag(nb)
    device_tag_ids = _fetch_device_tag_ids(nb)
    log.info("Weekly Backup tag id: %s", weekly_backup_tag_id)
    log.info("Device-applicable tags in Nautobot: %d\n", len(device_tag_ids))

    job = nb.extras.jobs.get(name=JOB_NAME)
    if job is None or isinstance(job, list):
        msg = (
            f"Job '{JOB_NAME}' not found. "
            "Ensure nautobot-app-device-onboarding is installed and the job is enabled."
        )
        raise RuntimeError(msg)

    for group_idx, (key, ips) in enumerate(groups.items(), start=1):
        uuids = _resolve_uuids(nb, key)
        location_uuid = str(uuids["location"])
        batches = list(_chunks(ips, BATCH_SIZE))

        log.info(
            "[group %d/%d] location=%s  IPs=%d  batches=%d",
            group_idx,
            len(groups),
            key.location_name,
            len(ips),
            len(batches),
        )

        for batch_idx, chunk in enumerate(batches, start=1):
            ip_str = ",".join(chunk)
            log.info("  [batch %d/%d] IPs: %s", batch_idx, len(batches), ip_str)

            job_data: dict[str, object] = {"ip_addresses": ip_str, **uuids}
            result = job.run(data=job_data)  # type: ignore[union-attr]
            job_result_id = result.job_result.id  # type: ignore[union-attr]
            log.info("  submitted — job_result id: %s", job_result_id)

            status = _wait_for_job(nb, str(job_result_id))
            log.info("  done: %s", status)
            log.info("  view: %s/extras/job-results/%s/", NAUTOBOT_URL, job_result_id)

            if status.lower() == "completed":
                log.info("  post-processing %d devices...", len(chunk))
                _post_process_devices(nb, chunk, location_uuid, weekly_backup_tag_id, device_tag_ids)
            else:
                log.info("  skipping post-processing (job did not complete successfully)")

        log.info("")


if __name__ == "__main__":
    run_sync_devices_job()
