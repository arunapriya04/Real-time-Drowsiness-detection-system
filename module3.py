# Detection thresholds
EAR_THRESH  = 0.25   # Eye Aspect Ratio below this → eyes closing
MAR_THRESH  = 0.65   # Mouth Aspect Ratio above this → yawning
EAR_FRAMES  = 20     # Consecutive frames below EAR_THRESH → drowsy alert
YAWN_FRAMES = 15     # Consecutive frames above MAR_THRESH → yawn counted

# Calibration
CALIB_FRAMES = 60    # Frames used to compute personal EAR baseline
ADAPT_RATIO  = 0.80  # Baseline multiplier to derive adaptive threshold

# MediaPipe face-mesh landmark indices
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61,  291,  13,  14,  17,   0, 269, 405]
