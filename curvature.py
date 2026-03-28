import numpy as np


def curvature_feature(mesh):

    normals = mesh.face_normals

    curvature = np.var(normals)

    return {"curvature": curvature}