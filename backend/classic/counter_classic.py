"""Classical (non-deep-learning) carton finder used as a demo/fallback.
Color segmentation (Lab) + seam removal + watershed. Rough estimate only."""
import cv2, numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def count_boxes(img_bgr, max_width=960, a_thr=134, b_thr=128, seam=12, smooth=2.0, min_dist=7, min_area_frac=0.00012):
    h0, w0 = img_bgr.shape[:2]
    s = min(1.0, max_width / w0)
    im = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img_bgr.copy()
    H, W = im.shape[:2]
    k = W / 960.0
    lab = cv2.cvtColor(cv2.GaussianBlur(im, (3,3), 0), cv2.COLOR_BGR2LAB)
    a, b = lab[...,1].astype(int), lab[...,2].astype(int)
    mask = ((a >= a_thr) & (b >= b_thr)).astype(np.uint8)
    mask[int(H*0.95):] = 0
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ks = max(5, int(11*k)) | 1
    mask[gray < cv2.blur(gray, (ks,ks)) - seam] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    dist = cv2.GaussianBlur(ndi.distance_transform_edt(mask).astype(np.float32), (0,0), smooth*k)
    pk = peak_local_max(dist, min_distance=max(3,int(min_dist*k)), threshold_abs=2.0*k, labels=mask)
    markers = np.zeros(mask.shape, np.int32)
    for i,(y,x) in enumerate(pk,1): markers[y,x]=i
    lbl = watershed(-dist, markers, mask=mask)
    min_area = max(12, min_area_frac*W*H)
    boxes=[]
    for r in range(1, lbl.max()+1):
        ys,xs = np.where(lbl==r)
        if len(ys) < min_area: continue
        boxes.append((xs.min(),ys.min(),xs.max(),ys.max()))
    return im, mask, boxes


def draw_boxes(im, boxes, color=(0, 200, 0)):
    out = im.copy()
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 1)
    return out
