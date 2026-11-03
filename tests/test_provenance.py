# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/network-intrusion-detection · MIT licence · sp2532-d3352b55697b
"""Keeps the authorship metadata intact (see LICENSE)."""
import nids


def test_authorship_metadata():
    assert nids.__author__ == "Shubhan Punde"
    assert nids.__provenance__ == "sp2532-d3352b55697b"
