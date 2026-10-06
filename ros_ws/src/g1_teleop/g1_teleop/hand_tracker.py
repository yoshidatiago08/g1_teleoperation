"""MediaPipe Hands on a crop around the wrist.

At 2 m the whole hand is only ~30 px wide in the camera image, too small for the hand model. The
Pose model already tells us where each wrist is, so we cut a square around it, enlarge it, and run
the hand model on that crop. One call handles one hand.
"""
import cv2
import mediapipe as mp
import numpy as np

HAND_LENGTH = 0.19   # m, wrist to the tip of the middle finger (used to size the crop)
CROP_PIXELS = 256    # the crop is resized to this before the hand model sees it


class HandTracker:
    def __init__(self, crop_margin=2.2, min_confidence=0.5):
        self.crop_margin = crop_margin  # crop side = this x the hand's length in pixels
        # static_image_mode: every call is detected from scratch. The crop moves from frame to
        # frame, so the hand model's own frame-to-frame tracking would not work here.
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=1, model_complexity=1,
            min_detection_confidence=min_confidence)

    def detect(self, rgb, center_px, hand_px):
        """Run the hand model around `center_px` (u, v). `hand_px`: hand length in pixels.

        Returns (world, image_points) or None. world: 21x3 landmarks in metres, in the image's
        axes (x right, y down, z away from the camera), origin at the hand. image_points: 21x2
        pixel positions in the full image, for drawing.
        """
        h, w = rgb.shape[:2]
        side = int(np.clip(self.crop_margin * hand_px, 64, min(h, w)))
        u0 = int(np.clip(center_px[0] - side // 2, 0, w - side))
        v0 = int(np.clip(center_px[1] - side // 2, 0, h - side))
        crop = rgb[v0:v0 + side, u0:u0 + side]
        crop = cv2.resize(crop, (CROP_PIXELS, CROP_PIXELS), interpolation=cv2.INTER_CUBIC)
        result = self.hands.process(crop)
        if not result.multi_hand_world_landmarks:
            return None
        world = np.array([[lm.x, lm.y, lm.z] for lm in result.multi_hand_world_landmarks[0].landmark])
        image = np.array([[u0 + lm.x * side, v0 + lm.y * side]
                          for lm in result.multi_hand_landmarks[0].landmark])
        return world, image

    def close(self):
        self.hands.close()
