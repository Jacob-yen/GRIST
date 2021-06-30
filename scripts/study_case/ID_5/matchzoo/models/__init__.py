from .dense_baseline import DenseBaseline
from .dssm import DSSM
from .cdssm import CDSSM
from .drmm import DRMM
from .drmmtks import DRMMTKS
from .esim import ESIM
from .knrm import KNRM
from .conv_knrm import ConvKNRM
from .bimpm import BiMPM
from .matchlstm import MatchLSTM
from .arci import ArcI
from .arcii import ArcII
from .bert import Bert
from .mvlstm import MVLSTM
from .match_pyramid import MatchPyramid


def list_available() -> list:
    from scripts.study_case.ID_5.matchzoo.engine.base_model import BaseModel
    from scripts.study_case.ID_5.matchzoo.utils import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(BaseModel)
