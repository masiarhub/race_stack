import os
import sys

NODES = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path[:0] = [os.path.join(NODES, 'datagen'), os.path.join(NODES, 'datagen', 'test'),
                os.path.join(NODES, '..', '..', 'base_system', 'f110_simulator', 'f1tenth_gym', 'gym')]
