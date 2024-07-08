"""Package containing different replay strategies.

Module: __init__.py
"""
# flake8: noqa

from openadapt.strategies.base import BaseReplayStrategy

# disabled because importing is expensive
from openadapt.strategies.cursor import CursorReplayStrategy
# from openadapt.strategies.demo import DemoReplayStrategy
from openadapt.strategies.naive import NaiveReplayStrategy
from openadapt.strategies.segment import SegmentReplayStrategy
from openadapt.strategies.stateful import StatefulReplayStrategy
from openadapt.strategies.vanilla import VanillaReplayStrategy
from openadapt.strategies.visual import VisualReplayStrategy

# add more strategies here
