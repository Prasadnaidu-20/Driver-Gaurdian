import numpy as np

from app.pipeline.preprocess import crop_face


def make_frame() -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = (255, 0, 0)  # pure blue in BGR
    return frame


def test_crop_shape_dtype_and_rgb_order() -> None:
    lm = np.array([[0.4, 0.4], [0.6, 0.6]])
    crop = crop_face(make_frame(), lm, size=224, margin=0.25)
    assert crop.shape == (224, 224, 3)
    assert crop.dtype == np.uint8
    assert tuple(crop[112, 112]) == (0, 0, 255)  # BGR blue became RGB blue


def test_face_at_edge_is_padded_with_black() -> None:
    lm = np.array([[0.0, 0.4], [0.2, 0.6]])  # bbox touches the left edge
    crop = crop_face(make_frame(), lm, size=224, margin=0.5)
    assert crop.shape == (224, 224, 3)
    assert crop[112, 0].sum() == 0  # left side is padding
    assert tuple(crop[112, 200]) == (0, 0, 255)


def test_face_partly_outside_frame_does_not_raise() -> None:
    lm = np.array([[-0.2, -0.1], [0.1, 0.2], [1.1, 1.2]])
    assert crop_face(make_frame(), lm, size=64, margin=0.25).shape == (64, 64, 3)
