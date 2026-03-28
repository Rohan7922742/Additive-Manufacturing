import numpy as np

#angles are being considered in degree

CRITICAL_ANGLE = 45  


def overhang_feature(mesh):

    normals = mesh.face_normals

    z_axis = np.array([0, 0, 1])

    angles = np.degrees(np.arccos(np.clip(normals @ z_axis, -1, 1)))

    overhang_faces = angles > CRITICAL_ANGLE

    ratio = np.sum(overhang_faces) / len(angles)

    max_overhang = np.max(angles)

    return {
        "max_overhang": max_overhang,
        "overhang_ratio": ratio
    }