"""A fake YOLO model so tests run without Ultralytics, torch or a trained model."""
import numpy as np


class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, xyxy, conf, cls, ids=None):
        self.xyxy, self.conf, self.cls = _T(xyxy), _T(conf), _T(cls)
        self.id = _T(ids) if ids is not None else None

    def __len__(self):
        return len(self.xyxy.a)


class _Result:
    def __init__(self, boxes, names):
        self.boxes, self.names = boxes, names


class FakeModel:
    """Always 'detects' two boxes."""

    names = {0: "box"}

    def predict(self, frame, **kwargs):
        return [_Result(_Boxes([[10, 10, 50, 50], [60, 60, 100, 100]], [0.9, 0.8], [0, 0]), self.names)]

    def track(self, frame, **kwargs):
        return [_Result(_Boxes([[10, 10, 50, 50], [60, 60, 100, 100]], [0.9, 0.8], [0, 0], ids=[1, 2]), self.names)]


class EmptyModel(FakeModel):
    def predict(self, frame, **kwargs):
        return [_Result(_Boxes(np.zeros((0, 4)), [], []), self.names)]


class EightBoxModel(FakeModel):
    """Eight similar upright boxes: enough for the room estimate (needs at least 5)."""

    def predict(self, frame, **kwargs):
        xyxy = [[20 + i * 60, 100, 70 + i * 60, 160] for i in range(8)]
        return [_Result(_Boxes(xyxy, [0.9] * 8, [0] * 8), self.names)]
