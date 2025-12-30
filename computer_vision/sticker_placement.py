import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# 1. Load image
# ==================================================
image = cv2.imread(r"C:\Users\himan\Pictures\Screenshots\tiltedbox.png")
assert image is not None, "Image not found"

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]
img_area = h * w

plt.figure(figsize=(6,6))
plt.imshow(image_rgb)
plt.title("Step 1: Original Image")
plt.axis("off")
plt.show()

# ==================================================
# 2. Preprocessing
# ==================================================
blur = cv2.GaussianBlur(image, (5, 5), 0)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

plt.figure(figsize=(6,6))
plt.imshow(hsv[:, :, 1], cmap="gray")
plt.title("Step 2: Saturation Channel (HSV)")
plt.axis("off")
plt.show()

# ==================================================
# 3. STRICT BROWN SEGMENTATION (CALIBRATED)
# ==================================================
# Values derived from YOUR HSV tuning
lower = np.array([13, 80, 90])
upper = np.array([35, 255, 255])

mask = cv2.inRange(hsv, lower, upper)

plt.figure(figsize=(6,6))
plt.imshow(mask, cmap="gray")
plt.title("Step 3: Raw Brown Mask")
plt.axis("off")
plt.show()

# ==================================================
# 4. Morphological cleanup
# ==================================================
kernel = np.ones((7, 7), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

plt.figure(figsize=(6,6))
plt.imshow(mask, cmap="gray")
plt.title("Step 4: Cleaned Mask")
plt.axis("off")
plt.show()

# ==================================================
# 5. Find contours (keep all)
# ==================================================
contours, _ = cv2.findContours(
    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)

assert contours, "No contours detected"

# ==================================================
# 6. SHAPE-BASED FILTERING (CRITICAL FIX)
# ==================================================
candidates = []

for c in contours:
    area = cv2.contourArea(c)

    # Remove noise
    if area < 3000:
        continue

    # Remove background-sized regions
    if area > 0.7 * img_area:
        continue

    rect = cv2.minAreaRect(c)
    (_, _), (rw, rh), _ = rect

    if rw == 0 or rh == 0:
        continue

    aspect = max(rw, rh) / min(rw, rh)

    # Box-like constraint
    if 1.2 < aspect < 3.5:
        candidates.append(c)

assert candidates, "No valid box contour found"

# Choose the best candidate
contour = max(candidates, key=cv2.contourArea)

debug = image_rgb.copy()
cv2.drawContours(debug, [contour], -1, (255, 0, 0), 3)

plt.figure(figsize=(6,6))
plt.imshow(debug)
plt.title("Step 5: Selected Box Contour")
plt.axis("off")
plt.show()

# ==================================================
# 7. Orientation using minAreaRect (CORRECT)
# ==================================================
(cx, cy), (bw, bh), angle = cv2.minAreaRect(contour)

# OpenCV angle normalization
if bw < bh:
    angle += 90
    bw, bh = bh, bw

box = cv2.boxPoints(((cx, cy), (bw, bh), angle))
box = box.astype(int)

# ==================================================
# 8. Sticker placement (CENTERED & ALIGNED)
# ==================================================
sticker_w = int(0.45 * bw)
sticker_h = int(0.22 * bh)

sticker_rect = ((cx, cy), (sticker_w, sticker_h), angle)
sticker_box = cv2.boxPoints(sticker_rect).astype(int)

# ==================================================
# 9. Final visualization
# ==================================================
output = image_rgb.copy()

# Draw detected box
cv2.drawContours(output, [box], 0, (0, 255, 0), 3)

# Draw sticker
overlay = output.copy()
cv2.fillPoly(overlay, [sticker_box], (255, 0, 255))
output = cv2.addWeighted(overlay, 0.35, output, 0.65, 0)
cv2.drawContours(output, [sticker_box], 0, (255, 0, 255), 3)

# Draw center
cv2.circle(output, (int(cx), int(cy)), 6, (255, 0, 0), -1)

plt.figure(figsize=(7,7))
plt.imshow(output)
plt.title("Step 6: Final Output (Correct Brown Box + Sticker)")
plt.axis("off")
plt.show()

# ==================================================
# 10. Numeric output (ASSIGNMENT READY)
# ==================================================
print("\n===== FINAL OUTPUT =====")
print(f"Box center (px): ({int(cx)}, {int(cy)})")
print(f"Sticker center (px): ({int(cx)}, {int(cy)})")
print(f"Orientation angle (deg): {angle:.2f}")
