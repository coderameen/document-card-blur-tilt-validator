# document-card-blur-tilt-validator
Production-ready service for validating document and ID card images by detecting blur, tilt, and capture quality issues using DocAligner detector



'main_doc_detection.py' validates document/card images by checking tilt (perspective distortion) and blur.
1. Card Detection:
  Uses DocAligner to detect document/card boundaries and prints detected corner coordinates for debugging.
2. Tilt / Perspective Check:
  Calculates corner angles from detected card edges.
  If any angle deviates more than 10° from 90°, the image is flagged as tilted (is_tilted_flag = True).
3. Blur Check:
  Computes a weighted quality score using:
       a) Laplacian variance
       b) Tenengrad sharpness
       c) Contrast
       d) Edge density
       e) Brightness
  If score < 0.42, the image is flagged as blurred (is_blur_flag = True).
5. Final Validation Result:
  validate_doc_flag = True only when:
    a) No perspective distortion
     b) No blur detected
7. Multi-Card Support:
  Handles both single-card and two-card (front/back) images by splitting into regions and validating each separately.


#Short Summary:
  The main_doc_detection.py script validates document images by detecting card boundaries using DocAligner (printing coordinates) and checking for perspective distortion (10.0° corner angle threshold) and blur (0.42 quality score threshold), returning validate_doc_flag as True only when both checks pass. Both thresholds (10.0 degrees for perspective and 0.42 for blur quality) are configurable parameters that users can adjust according to their specific use cases and quality requirements.
