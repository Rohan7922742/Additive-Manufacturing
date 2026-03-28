import trimesh 
def load_mesh(path):
    mesh = trimesh.load(path)
    if mesh is None:
        raise ValueError("Failed to load mesh")
    
    if not mesh.is_watertight:
        print(f"Warning: Non-watertight mesh: {path}")
        mesh.is_watertight_flag = False
    else:
        mesh.is_watertight_flag = True
    
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    
    return mesh
