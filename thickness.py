def thickness_feature(mesh):

    extents = mesh.bounding_box.extents

    min_dim = min(extents)

    return {"min_thickness": min_dim}