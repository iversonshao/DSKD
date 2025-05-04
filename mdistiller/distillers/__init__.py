from ._base import Vanilla
from .KD import KD
from .SDD_KD import SDD_KD
from .MLKD import MLKD
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .Sonly import Sonly
from .VID import VID
from .ReviewKD import ReviewKD
from .DKD import DKD
from .SDD_DKD import SDD_DKD
from .RLD import RLD
from .SDD_RLD import SDD_RLD
from .SWAP import SWAP
from .REVISION import REVISION
from .RC import RC
from .MLKD_NOAUG import MLKD_NOAUG

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "SDD_KD" : SDD_KD,
    "MLKD": MLKD,
    "MLKD_NOAUG": MLKD_NOAUG,
    "AT": AT,
    "OFD": OFD,
    "RKD": RKD,
    "FITNET": FitNet,
    "KDSVD": KDSVD,
    "CRD": CRD,
    "NST": NST,
    "PKT": PKT,
    "SP": SP,
    "Sonly": Sonly,
    "VID": VID,
    "REVIEWKD": ReviewKD,
    "DKD": DKD,
    "SDD_DKD": SDD_DKD,
    "RLD": RLD,
    "SDD_RLD": SDD_RLD,    
    "SWAP": SWAP,
    "REVISION": REVISION,
    "RC": RC,
}
