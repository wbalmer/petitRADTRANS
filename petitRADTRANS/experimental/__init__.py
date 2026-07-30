"""Experimental petitRADTRANS features.

Everything under :mod:`petitRADTRANS.experimental` is **under active
development**. Unlike the rest of petitRADTRANS, these subpackages are not
covered by the usual deprecation policy: module layout, public names, call
signatures, on-disk artifact formats (datasets, saved posteriors, run
manifests), and numerical behaviour may change -- or be removed -- in any
release, including patch releases.

Pin an exact petitRADTRANS version if you depend on anything here, and expect
to re-generate cached artifacts after upgrading.

Current contents
----------------
* :mod:`petitRADTRANS.experimental.sbi` -- amortized simulation-based inference
  (neural posterior estimation) built on the retrieval runtime.
"""

import warnings

warnings.warn(
    "petitRADTRANS.experimental contains features under active development. "
    "Their public API, saved-artifact formats, and numerical behaviour may "
    "change without a deprecation cycle; pin a petitRADTRANS version if you "
    "depend on them.",
    UserWarning,
    stacklevel=2,
)
