# Third-party notices

Masht incorporates modified source from HYDRA and MultiRocket. Both projects
are licensed under the GNU General Public License, version 3. Consequently,
Masht is distributed under GPL-3.0-only as a whole. The complete license is in
[`LICENSE`](LICENSE), and preserved copies are in [`LICENSES/`](LICENSES/).

## HYDRA

- Upstream: <https://github.com/angus924/hydra>
- Authors named in the source: Angus Dempster, Daniel F. Schmidt, and
  Geoffrey I. Webb
- License: GNU GPL version 3
- Included files: `masht_vendor/hydra/univariate.py` and
  `masht_vendor/hydra/multivariate.py`
- Changes made for Masht on 2026-07-17: files were renamed and repackaged;
  algorithm code was not changed.

## MultiRocket

- Upstream: <https://github.com/ChangWeiTan/MultiRocket>
- Authors named in the source: Chang Wei Tan, Angus Dempster, Christoph
  Bergmeir, and Geoffrey I. Webb
- License: GNU GPL version 3
- Included files: `masht_vendor/multirocket/univariate.py` and
  `masht_vendor/multirocket/multivariate.py`
- Changes made for Masht on 2026-07-17: files were renamed and repackaged, and
  the unused standalone MultiRocket classifier wrappers and imports were
  removed. The `fit` and `transform` kernel implementations used by Masht were
  retained.

## TabPFN

TabPFN is installed as an external dependency and is not redistributed here.
Its code and model checkpoints have terms separate from Masht. In particular,
the current default TabPFN model weights may be restricted to non-commercial
use and require acceptance of Prior Labs' terms. Users must review and comply
with the terms published at <https://github.com/PriorLabs/TabPFN> before a
checkpoint is downloaded or used. Masht's GPL license does not grant rights to
TabPFN code or model weights.
