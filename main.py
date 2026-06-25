import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# Load SAM model
sam = sam_model_registry["vit_b"](checkpoint=r"C:\Users\HP\Documents\Intro to CV\SAM_Project\sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)

# Read image
image = cv2.imread(r"C:\Users\HP\Documents\Intro to CV\SAM_Project\images\download.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image_rgb)

# Example click point
input_point = np.array([[300, 200]])
input_label = np.array([1])

# Predict mask
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# Apply mask
mask = masks[0]

image[mask] = [0, 255, 0]

cv2.imshow("Segmented", image)
cv2.waitKey(0)
cv2.destroyAllWindows()