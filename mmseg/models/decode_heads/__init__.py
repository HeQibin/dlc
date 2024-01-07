# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional decode_heads

from .aspp_head import ASPPHead
from .da_head import DAHead
from .daformer_head import DAFormerHead
from .dlv2_head import DLV2Head
from .fcn_head import FCNHead
from .hrda_head import HRDAHead
from .isa_head import ISAHead
from .psp_head import PSPHead
from .segformer_head import SegFormerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .uper_head import UPerHead

from .deformable_head_with_time import DeformableHeadWithTime
from .segformer_head_unet_fc_head_multi_step import SegformerHeadUnetFCHeadMultiStep
from .segformer_head_unet_fc_head_single_step import SegformerHeadUnetFCHeadSingleStep
from .daformer_ntm_head import DAFormerHeadWithNTM
from .hrda_ntm_head import HRDAHeadWithNTM
from .hrda_ntmcor_head import HRDAHeadWithNTMCorrect

__all__ = [
    'FCNHead',
    'PSPHead',
    'ASPPHead',
    'UPerHead',
    'DepthwiseSeparableASPPHead',
    'DAHead',
    'DLV2Head',
    'SegFormerHead',
    'DAFormerHead',
    'ISAHead',
    'HRDAHead',
    'DeformableHeadWithTime',
    'SegformerHeadUnetFCHeadMultiStep',
    'SegformerHeadUnetFCHeadSingleStep',
    'DAFormerHeadWithNTM',
    'HRDAHeadWithNTM',
    'HRDAHeadWithNTMCorrect'
]
