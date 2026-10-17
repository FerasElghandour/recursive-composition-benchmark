"""Offline example using the public implementation; see README for evidence scope."""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from experiment import dataset,target,forbidden
d=dataset(0)
print({'train_3':list(d['train_3'].shape),'ood_3':list(d['ood_3'].shape),'long_8':list(d['long_8'].shape),'train_has_forbidden':any(forbidden(p) for p in d['train_3']),'ood_has_forbidden':all(forbidden(p) for p in d['ood_3'])})
print({'target_first_train':target(d['train_3'][:1]).tolist()})
