"""
Output generation modules.

This package handles:
- MusicXML generation
- Debug visualization output
"""

from homr.output.music_xml_generator import (
    generate_xml,
    XmlGeneratorArguments,
)
from homr.output.visualization_output import VisualizationOutput

__all__ = [
    "generate_xml",
    "XmlGeneratorArguments",
    "VisualizationOutput",
]
