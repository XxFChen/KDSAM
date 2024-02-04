from ._base import Vanilla
from .ReviewKD import ReviewKD
from .DKD import DKD

distiller_dict = {
    "NONE": Vanilla,
    "REVIEWKD": ReviewKD,
    "DKD": DKD,
}