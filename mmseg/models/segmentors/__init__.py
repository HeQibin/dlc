# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add HRDAEncoderDecoder

from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder
from .hrda_encoder_decoder import HRDAEncoderDecoder
from .ntmcor_encoder_decoder import NTMCorrectEncoderDecoder

__all__ = ['BaseSegmentor', 'EncoderDecoder', 'HRDAEncoderDecoder',
           'NTMCorrectEncoderDecoder']
