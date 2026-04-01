"""Signal processing for extracting vibration vectors from time-domain data.

Typical workflow
----------------
1. Determine the rotation frequency — either from machine specifications
   or by calling :func:`detect_rotation_freq` on the trigger signal.
2. Create a :class:`VibrationExtractor` with the sample rate and
   rotation frequency.
3. Call :meth:`~VibrationExtractor.get_vibration_vector` for each
   measurement run to obtain a :class:`~pyPRB.core.VibrationVector`.

The default extraction method is :class:`SynchronousAveraging`, which
averages revolution-locked signal segments and extracts the fundamental
(once-per-revolution) component.  Custom methods can be provided via the
:class:`ExtractionMethod` protocol.
"""

from ._vibration_extractor import ExtractionMethod, VibrationExtractor, detect_rotation_freq
from .fft_extraction import FFTExtraction
from .iq_demodulation import IQDemodulation
from .synchronous_averaging import SynchronousAveraging

__all__ = [
    "ExtractionMethod",
    "VibrationExtractor",
    "detect_rotation_freq",
    "FFTExtraction",
    "IQDemodulation",
    "SynchronousAveraging",
]
