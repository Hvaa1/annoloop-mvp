"""Source calibration PAVA reused from demo co-training."""
import numpy as np

def isotonic(x, y):
    """Weighted PAVA on duplicate input values, with clipped endpoint extension."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 10 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Insufficient/non-finite source calibration samples")
    values, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=y)
    blocks = []
    for i, (total, count) in enumerate(zip(sums, counts)):
        blocks.append([i, i, float(total), int(count)])
        while len(blocks) > 1 and blocks[-2][2]/blocks[-2][3] > blocks[-1][2]/blocks[-1][3]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0], right[1], left[2]+right[2], left[3]+right[3]])
    fitted = np.empty(len(values))
    for first, last, total, count in blocks:
        fitted[first:last+1] = total/count
    return dict(x=values.tolist(), y=fitted.tolist(), samples=len(x))
