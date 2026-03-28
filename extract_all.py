from src.features.basic import basic_features
from src.features.overhang import overhang_feature
from src.features.thickness import thickness_feature
from src.features.curvature import curvature_feature


def extract_features(mesh):

    features = {}

    features.update(basic_features(mesh))
    features.update(overhang_feature(mesh))
    features.update(thickness_feature(mesh))
    features.update(curvature_feature(mesh))

    return features