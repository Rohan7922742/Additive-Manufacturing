def basic_features(mesh):

    return {

        "volume": mesh.volume,
        "surface_area": mesh.area,
        "triangle_count": len(mesh.faces),
        "bbox_x": mesh.bounding_box.extents[0],
        "bbox_y": mesh.bounding_box.extents[1],
        "bbox_z": mesh.bounding_box.extents[2]
    }