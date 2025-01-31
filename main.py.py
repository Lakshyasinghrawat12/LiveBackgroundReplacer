import cv2
import mediapipe as mp
import numpy as np

mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)


background_image = cv2.imread(r'File_location', cv2.IMREAD_COLOR)


cap = cv2.VideoCapture(0)

def apply_feathering(mask, radius=1):
    kernel_size = (radius * 2 + 1, radius * 2 + 1)
    return cv2.GaussianBlur(mask, kernel_size, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    background_resized = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform segmentation
    results = selfie_segmentation.process(frame_rgb)

    # Extract the mask (1 where the person is, 0 elsewhere)
    mask = results.segmentation_mask > 0.5

    # Feather the edges of the mask
    mask = apply_feathering(mask.astype(np.uint8) * 255, radius=1)

    # Convert mask back to 3 channels
    mask_3d = np.dstack((mask, mask, mask))

    # Replace the background
    frame_with_background = np.where(mask_3d, frame, background_resized)

    # Display the output
    cv2.imshow('Selfie Segmentation', frame_with_background)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
