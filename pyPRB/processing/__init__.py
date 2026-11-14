"""Signal processing for extracting vibration vectors from time-domain data.

Typical workflow
----------------
1. Determine the rotation frequency — either from machine specifications
   or by calling :func:`detect_rotation_freq` on the trigger signal.
2. Create a :class:`VibrationExtractor` with the sample rate and
   rotation frequency.
3. Call :meth:`~VibrationExtractor.get_vibration_vector` for each
   measurement run to obtain a :class:`~pyPRB.core.VibrationVector`.

The default extraction method is :class:`IQDemodulation`, which
multiplies the signal by cosine/sine references at the rotation frequency
and averages to extract the fundamental component.  Custom methods can be
provided via the :class:`ExtractionMethod` protocol.

All built-in methods measure the phase from the trigger edge (rising by
default): a response ``A·cos(θ + φ)``, where ``θ`` is the rotation angle
counted from the edge, gives the phase ``φ``.  The edges are located with
sub-sample resolution and hysteresis, so the phase does not depend on the
pulse width of the trigger.
"""

from ._vibration_extractor import ExtractionMethod, VibrationExtractor, detect_rotation_freq
from ._vibration_extractor import (HarmonicContent, SpeedVariation, VectorScatter,
                                   rotation_freq_variation)
from .iq_demodulation import IQDemodulation
from .least_squares_fit import LeastSquaresFit
from .synchronous_averaging import SynchronousAveraging
from .fft_extraction import FFTExtraction

__all__ = [
    "ExtractionMethod",
    "VibrationExtractor",
    "detect_rotation_freq",
    "rotation_freq_variation",
    "SpeedVariation",
    "VectorScatter",
    "HarmonicContent",
    "IQDemodulation",
    "LeastSquaresFit",
    "SynchronousAveraging",
    "FFTExtraction",
]
