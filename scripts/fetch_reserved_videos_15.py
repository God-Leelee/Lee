"""New metadata-selected100 videos, excluding every prior source boundary."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import fetch_independent_videos_13 as base

if __name__=='__main__':
    base.OUTPUT=base.ROOT/'data/public_genvideo_reserved_v1.5';base.SEED=10241504;base.TOTAL_ROWS=100
    base.SCOPE='New1.5 reserved public source videos, never fitted; source labels are not insurance business truth'
    base.PROVENANCE_INPUTS+=('data/public_genvideo_independent_v1/manifest.json','data/complete_suite_v1.3/manifest.json',
      'data/public_genvideo_v1.4/manifest.json','data/complete_suite_v1.4/manifest.json')
    base.main()
