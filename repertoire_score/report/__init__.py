"""Readable summary, report, chapter and opening pages from matching saved analysis JSONs.

Rendering never reads Explorer tables, recalculates metrics, or uses the network.
Companion hashes prevent mixing results from different repertoire snapshots.
"""

from .generate import generate, page_names

__all__ = ['generate', 'page_names']
