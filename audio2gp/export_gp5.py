"""Stable GP5 export interface.

The implementation lives in :mod:`audio2gp.gp_writer`. Both the batch
pipeline and the editor rebuild path go through these functions, so a score
edit is exported by exactly the same writer as the original job.
"""

from .gp_writer import parse_gp5, verify_gp5, write_gp5

__all__ = ["parse_gp5", "verify_gp5", "write_gp5"]
