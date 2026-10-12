from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data"
ENERGIES = ("124", "197", "360", "861")


def benzene_files(channel="inc"):
    return [DATA / f"benzene_290_{e}_{channel}.nxspe" for e in ENERGIES]


@pytest.fixture(scope="session")
def data_dir():
    if not all(p.exists() for p in benzene_files()):
        pytest.skip("benzene LET files not found in data/")
    return DATA


@pytest.fixture(scope="session")
def benzene(data_dir):
    """The four incident energies of the 290 K run, incoherent channel, as fitted in the benchmark."""
    from uqens import let

    return let.load(benzene_files())
