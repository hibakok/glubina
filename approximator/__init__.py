"""
Эволюционирующий универсальный аппроксиматор вычислимых функций.
Пакет для аппроксимации функций с помощью генетического программирования.
"""

from .data import DataPair, Dataset
from .tree import ExpressionTree, Node, NodeType, Operation
from .population import Population
from .core import UniversalApproximator

__version__ = "1.0.0"
__all__ = [
    "DataPair",
    "Dataset", 
    "ExpressionTree",
    "Node",
    "NodeType",
    "Operation",
    "Population",
    "UniversalApproximator",
]
