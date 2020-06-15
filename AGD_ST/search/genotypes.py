from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'skip',
    'conv3x3',
    #'conv3x3_d2',
    # 'conv3x3_d4',
    'residual',
    'dwsblock',
]
