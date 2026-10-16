"""Training entry adapter for the prepared controller/human-GT fixtures."""
from pathlib import Path
import sys
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from openwam.dataloader.curated_fixture import CuratedFixtureDataset

class CuratedDataset(CuratedFixtureDataset):
    def __init__(self, augmentation=None):
        super().__init__(OUT / "data_manifest.json", augmentation=augmentation)
