from enum import Enum

import numpy as np
import torch
import transforms3d
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix


class PoseType(Enum):
    """Enum class for different types of poses"""

    MAT: str = "mat"  # (4, 4) matrix
    POS_QUAT: str = "pos_quat"  # (7,) position and quaternion
    ROT_6D: str = "rot_6d"  # (9,) position and 6D rotation
    EULER: str = "euler"  # (6,) position and euler angles


def rot_6d_to_mat(rot_6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to rotation matrix."""
    assert rot_6d.shape[1] == 9, f"Invalid rot_6d shape: {rot_6d.shape}"
    pos = rot_6d[:, :3]
    rot_6d = rot_6d[:, 3:]
    rot_mat = rotation_6d_to_matrix(torch.from_numpy(rot_6d))
    rot_mat = rot_mat.numpy()
    mat = np.zeros((rot_6d.shape[0], 4, 4))
    mat[:, :3, :3] = rot_mat
    mat[:, :3, 3] = pos
    mat[:, 3, 3] = 1
    return mat


def mat_to_rot_6d(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to 6D rotation representation."""
    assert mat.shape[1:] == (4, 4), f"Invalid matrix shape: {mat.shape}"
    rot_mat = mat[:, :3, :3]
    rot_6d = matrix_to_rotation_6d(torch.from_numpy(rot_mat))
    pos = mat[:, :3, 3]
    return np.concatenate([pos, rot_6d.numpy()], axis=1)


def pos_quat_to_mat(pose_in_pos_quat: np.ndarray) -> np.ndarray:
    assert (
        pose_in_pos_quat.shape[1] == 7
    ), f"Invalid pose_in_pos_quat shape: {pose_in_pos_quat.shape}"
    pos = pose_in_pos_quat[:, :3]
    quat = pose_in_pos_quat[:, 3:]
    mat = np.zeros((pose_in_pos_quat.shape[0], 4, 4))
    for i in range(pose_in_pos_quat.shape[0]):
        mat[i, :3, :3] = transforms3d.quaternions.quat2mat(quat[i])
    mat[:, :3, 3] = pos
    mat[:, 3, 3] = 1
    return mat


def pose_convert(
    pose: np.ndarray, from_type: PoseType, to_type: PoseType, convention: str = "xyz"
) -> np.ndarray:
    """Convert pose from one type to another type

    Args:
        pose: input pose
        from_type: input pose type
        to_type: output pose type
        convention: euler angle convention
    Returns:
        output pose
    """
    if from_type == to_type:
        return pose

    if from_type == PoseType.ROT_6D and to_type == PoseType.MAT:
        return rot_6d_to_mat(pose)
    elif from_type == PoseType.MAT and to_type == PoseType.ROT_6D:
        return mat_to_rot_6d(pose)
    elif from_type == PoseType.POS_QUAT and to_type == PoseType.MAT:
        return pos_quat_to_mat(pose)
    else:
        raise NotImplementedError(
            f"Conversion from {from_type} to {to_type} is not implemented"
        )
