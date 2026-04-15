"""Pytest hooks for ``tests/io/remote_storage/`` only (not loaded from repo-root ``tests/conftest.py``).

Creates buckets/containers when remote emulators are running. Assumes emulators are already up
(e.g. Docker: ``docker run -p 5000:5000 -p 10000:10000 -p 4443:4443 spatialdata-emulators``).
Ports: S3/moto 5000, Azure/Azurite 10000, GCS/fake-gcs-server 4443.

``pytest_configure`` here patches ``fsspec.asyn.sync`` and ``gcsfs`` session teardown for this subtree
only; the library package itself does not apply those patches globally.
"""

from __future__ import annotations

import os

os.environ.setdefault("GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT", "false")

import socket
import time

import pytest



def _ensure_gcs_emulator_env() -> None:
    """Point google-cloud-storage / gcsfs defaults at fake-gcs-server (not production)."""
    raw = os.environ.get("STORAGE_EMULATOR_HOST", "").strip()
    if raw in ("", "default"):
        os.environ["STORAGE_EMULATOR_HOST"] = "http://127.0.0.1:4443"
    elif not raw.startswith(("http://", "https://")):
        os.environ["STORAGE_EMULATOR_HOST"] = f"http://{raw}"


# Error messages from asyncio when closing sessions after the event loop is gone (e.g. at process exit)
_LOOP_GONE_ERRORS = ("different loop", "Loop is not running")


def _patch_fsspec_sync_for_shutdown() -> None:
    """If fsspec.asyn.sync() runs at exit when the loop is gone, return None instead of raising.

    SpatialData does not patch ``fsspec.asyn.sync`` at import time (too broad for a library); this
    hook runs only for pytest sessions that load this conftest (remote emulator tests).
    """
    import fsspec.asyn as asyn_mod

    _orig = asyn_mod.sync

    def _wrapped(loop, func, *args, timeout=None, **kwargs):
        try:
            return _orig(loop, func, *args, timeout=timeout, **kwargs)
        except RuntimeError as e:
            if any(msg in str(e) for msg in _LOOP_GONE_ERRORS):
                return None
            raise

    asyn_mod.sync = _wrapped


def _patch_gcsfs_close_session_for_shutdown() -> None:
    """If gcsfs close_session fails (loop gone), close the connector synchronously instead of raising."""
    import asyncio

    import fsspec
    import fsspec.asyn as asyn_mod
    import gcsfs.core

    @staticmethod
    def _close_session(loop, session, asynchronous=False):
        if session.closed:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        use_force_close = False
        if loop and loop.is_running():
            loop.create_task(session.close())
        elif running and running.is_running() and asynchronous:
            running.create_task(session.close())
        elif asyn_mod.loop[0] is not None and asyn_mod.loop[0].is_running():
            try:
                asyn_mod.sync(asyn_mod.loop[0], session.close, timeout=0.1)
            except (RuntimeError, fsspec.FSTimeoutError):
                use_force_close = True
        else:
            use_force_close = True

        if use_force_close:
            connector = getattr(session, "_connector", None)
            if connector is not None:
                connector._close()

    gcsfs.core.GCSFileSystem.close_session = _close_session


def _apply_resilient_async_close_patches() -> None:
    """Avoid RuntimeError tracebacks when aiohttp sessions are closed at process exit (loop already gone)."""
    _patch_fsspec_sync_for_shutdown()
    _patch_gcsfs_close_session_for_shutdown()


def pytest_configure(config: pytest.Config) -> None:
    """Apply patches for remote storage tests (resilient async close at shutdown)."""
    _apply_resilient_async_close_patches()


EMULATOR_PORTS = {"s3": 5000, "azure": 10000, "gcs": 4443}
S3_BUCKETS = ("bucket", "test-azure", "test-s3", "test-gcs")
AZURE_CONTAINERS = ("test-container", "test-azure", "test-s3", "test-gcs")
GCS_BUCKETS = ("bucket", "test-azure", "test-s3", "test-gcs")

AZURITE_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=http;"
    "AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)


def _port_open(host: str = "127.0.0.1", port: int | None = None, timeout: float = 2.0) -> bool:
    if port is None:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, TimeoutError):
        return False


def _ensure_s3_buckets(host: str) -> None:
    if not _port_open(host, EMULATOR_PORTS["s3"]):
        return
    os.environ.setdefault("AWS_ENDPOINT_URL", "http://127.0.0.1:5000")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
        config=Config(signature_version="s3v4"),
    )
    existing = {b["Name"] for b in client.list_buckets().get("Buckets", [])}
    for name in S3_BUCKETS:
        if name not in existing:
            client.create_bucket(Bucket=name)


def _ensure_azure_containers(host: str) -> None:
    if not _port_open(host, EMULATOR_PORTS["azure"]):
        return
    from azure.storage.blob import BlobServiceClient

    client = BlobServiceClient.from_connection_string(AZURITE_CONNECTION_STRING)
    existing = {c.name for c in client.list_containers()}
    for name in AZURE_CONTAINERS:
        if name not in existing:
            client.create_container(name)


def _ensure_gcs_buckets(host: str) -> None:
    if not _port_open(host, EMULATOR_PORTS["gcs"]):
        return
    os.environ.setdefault("STORAGE_EMULATOR_HOST", "http://127.0.0.1:4443")
    from google.auth.credentials import AnonymousCredentials
    from google.cloud import storage

    client = storage.Client(credentials=AnonymousCredentials(), project="test")
    existing = {b.name for b in client.list_buckets()}
    for name in GCS_BUCKETS:
        if name not in existing:
            client.create_bucket(name)


def _wait_for_emulator_ports(host: str = "127.0.0.1", timeout: float = 10.0, check_interval: float = 2.0) -> None:
    """Wait until all three emulator ports accept connections (e.g. after docker run)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(_port_open(host, EMULATOR_PORTS[p]) for p in ("s3", "azure", "gcs")):
            return
        time.sleep(check_interval)
    raise RuntimeError(
        f"Emulators did not become ready within {timeout}s. "
        "Ensure the container is running: docker run --rm -d -p 5000:5000 "
        "-p 10000:10000 -p 4443:4443 spatialdata-emulators"
    )


@pytest.fixture(scope="session")
def _remote_storage_buckets_containers():
    """Create buckets/containers on running emulators so remote storage tests can run.

    Run with emulators up, e.g.:
      docker run --rm -d -p 5000:5000 -p 10000:10000 -p 4443:4443 spatialdata-emulators
    Then: pytest tests/io/test_remote_storage.py -v
    """
    host = "127.0.0.1"
    _wait_for_emulator_ports(host)
    _ensure_s3_buckets(host)
    _ensure_azure_containers(host)
    _ensure_gcs_buckets(host)
    yield


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Inject bucket/container creation for test_remote_storage.py."""
    if any("remote_storage" in str(getattr(item, "path", None) or getattr(item, "fspath", "")) for item in items):
        _ensure_gcs_emulator_env()
    for item in items:
        path = getattr(item, "path", None) or getattr(item, "fspath", None)
        if path and "test_remote_storage" in str(path):
            item.add_marker(pytest.mark.usefixtures("_remote_storage_buckets_containers"))
