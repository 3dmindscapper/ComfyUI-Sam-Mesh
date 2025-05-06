import numpy as np
import trimesh
from trimesh.base import Trimesh, Scene # Ensure Scene is imported here
from samesh.data.common import NumpyTensor, TorchTensor # Uncommented this line

def duplicate_verts(mesh: Trimesh) -> Trimesh:
    """
    Call before coloring mesh to avoid face interpolation since openGL stores color attributes per vertex.

        ...
        mesh = duplicate_verts(mesh)
        mesh.visual.face_colors = colors
        ...

    NOTE: removes visuals for verticies, but preserves for faces.
    """
    verts = mesh.vertices[mesh.faces.reshape(-1), :]
    faces = np.arange(0, verts.shape[0])
    faces = faces.reshape(-1, 3)

    kwargs = {'vertices': verts, 'faces': faces, 'process': False}
    
    # Check if the mesh has visual information.
    if hasattr(mesh, 'visual') and mesh.visual is not None:
        # Only try to use face_colors if mesh.visual is not TextureVisuals
        # and if it actually has a face_colors attribute that is not None.
        if not isinstance(mesh.visual, trimesh.visual.TextureVisuals) and \
           hasattr(mesh.visual, 'face_colors') and \
           mesh.visual.face_colors is not None:
            kwargs['face_colors'] = mesh.visual.face_colors
        # If it is TextureVisuals, or no face_colors are present or they are None,
        # we don't pass face_colors to the new Trimesh.
        # The new mesh will be created without explicit face/vertex colors derived from original face_colors.
        # Textures (UVs, material) would need separate handling if they are to be preserved
        # on this new geometry, as this function primarily modifies vertex/face structure.

    return Trimesh(**kwargs)


def handle_pose(pose: NumpyTensor['4 4']) -> NumpyTensor['4 4']:
    """
    Handles common case that results in numerical instability in rendering faceids:

        ...
        pose, _ = scene.graph[name]
        pose = handle_pose(pose)
        ...
    """
    identity = np.eye(4)
    if np.allclose(pose, identity, atol=1e-6):
        return identity
    return pose


def transform(pose: NumpyTensor['4 4'], vertices: NumpyTensor['nv 3']) -> NumpyTensor['nv 3']:
    """
    """
    homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
    return (pose @ homogeneous.T).T[:, :3]


def concat_scene_vertices(scene: Scene) -> NumpyTensor['nv 3']:
    """
    """
    verts = []
    for name, geom in scene.geometry.items():
        if name in scene.graph:
            pose, _ = scene.graph[name]
            pose = handle_pose(pose)
            geom.vertices = transform(pose, geom.vertices)
        verts.append(geom.vertices)
    return np.concatenate(verts)


def bounding_box(vertices: NumpyTensor['n 3']) -> NumpyTensor['2 3']:
    """
    Compute bounding box from vertices.
    """
    return np.array([vertices.min(axis=0), vertices.max(axis=0)])


def bounding_box_centroid(vertices: NumpyTensor['n 3']) -> NumpyTensor['3']:
    """
    Compute bounding box centroid from vertices.
    """
    return bounding_box(vertices).mean(axis=0)


def norm_mesh(mesh: Trimesh) -> Trimesh:
    """
    Normalize mesh vertices to bounding box [-1, 1]. 
    
    NOTE:: In place operation that consumes mesh.
    """
    centroid = bounding_box_centroid(mesh.vertices)
    mesh.vertices -= centroid
    mesh.vertices /= np.abs(mesh.vertices).max()
    mesh.vertices *= (1 - 1e-3)
    return mesh


def norm_scene(scene: Scene) -> Scene:
    """
    Normalize scene vertices to bounding box [-1, 1]. 
    
    NOTE:: In place operation that consumes scene.
    """
    centroid = bounding_box_centroid(concat_scene_vertices(scene))
    for geom in scene.geometry.values():
        geom.vertices -= centroid
    extent = np.abs(concat_scene_vertices(scene)).max()
    for geom in scene.geometry.values():
        geom.vertices /= extent
        geom.vertices *= (1 - 1e-3)
    return scene


if __name__ == "__main__":
    from samesh.data.loaders import read_mesh
    mesh = read_mesh('/home/ubuntu/meshseg/tests/examples/0ba4ae3aa97b4298866a2903de4fd1e7.glb')
    
    mesh.export('/home/ubuntu/meshseg/tests/examples/0ba4ae3aa97b4298866a2903de4fd1e7.obj')
    print(mesh.faces)
    print(mesh.vertices[mesh.faces[:, 0]])
    mesh = order_faces(mesh)
    print(mesh.faces)
    print(mesh.vertices[mesh.faces[:, 0]])
    
    mesh.export('/home/ubuntu/meshseg/tests/examples/0ba4ae3aa97b4298866a2903de4fd1e7_sorted.obj')
    print(mesh.vertices.max(), mesh.vertices.min())
    mesh = norm_mesh(mesh)
    print(mesh.vertices.max(), mesh.vertices.min())
    mesh.export('/home/ubuntu/meshseg/tests/examples/0ba4ae3aa97b4298866a2903de4fd1e7_norm.obj')

    print(mesh.vertices.shape)
    print(mesh.faces.shape)
    print(mesh.vertices[mesh.faces].shape)