"""Stable separation interface.

The implementation lives in :mod:`audio2gp.separate` (kept for backwards
compatibility with the original MVP). New code may import either name; this
module re-exports the supported surface so callers do not depend on private
helpers.
"""

from .separate import CACHE_VERSION, STEM_ORDER, separate, stem_cache_dir

__all__ = ["CACHE_VERSION", "STEM_ORDER", "separate", "stem_cache_dir"]
