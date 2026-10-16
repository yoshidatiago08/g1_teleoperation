# 3D hand landmark reconstruction

How we turn a single camera image of your hand into a metric 3D hand (wrist pose + all 21 landmarks), and why we do it this way.
Code: `src/dex_hand/hand_model.py` (model and fit), `src/dex_hand/vm.py` (calibration), `src/dex_hand/pipeline.py` (use).

## The problem

MediaPipe Hands gives 21 landmarks per hand:

| What MediaPipe gives | How good it is |
|---|---|
| 2D position in the image (pixels) | Good, even for fingers that are partly hidden |
| `z` (depth relative to the wrist) | Noisy, and **a guess** for hidden landmarks |
| Orientation of the hand | Not provided at all |
| Real-world size / distance to the camera | Not provided (scale is unknown) |

The robot needs: the **wrist position** (metric, including distance to the camera), the **palm orientation**, and **wrist → fingertip
vectors** for the finger retargeting. Reading these straight from MediaPipe's 3D output fails exactly where it matters:

- fingers pointing at the camera (end-on) or a fist: landmarks are hidden, their `z` is a guess, distance and tilt come out wrong
- the palm normal is computed from knuckles that can be mis-estimated in depth, so the orientation jumps

Measured on a synthetic hand with realistic noise, raw MediaPipe-style 3D gives a mean palm orientation error of about 28°, distance
errors of up to 36 cm for end-on poses and fingertip errors of 20–140 mm.

## The idea

> Trust MediaPipe where it is reliable (the 2D image position of each landmark). Take everything else from the **geometry of your own
> hand**, measured once.

Concretely, two steps:

1. **Calibration** measures your hand's real geometry (`HandShape`).
2. **Every frame**, a model of that hand (`HandModel`: wrist pose + finger joint angles) is moved until its perspective projection lands on
   MediaPipe's 2D landmarks (`HandFit`). The fitted model *is* the 3D hand.

Why this works:

- **Perspective + known size gives distance.** A 18 cm hand that appears 120 px long is at a definite distance. This does not need depth
  from MediaPipe, and the apparent size of the rigid palm is a strong cue.
- **The palm is rigid.** Wrist, thumb base and the four knuckles never move relative to each other, so the palm orientation comes from
  those 6 points and does not depend on any finger (which may be curled or hidden).
- **Fingers have joint limits.** A hidden fingertip is not unknown: it is wherever the finger could be given its visible knuckle, the joint
  limits and the 2D evidence. The model keeps wrist → fingertip vectors valid even when the tip is hidden.

## Step 1: calibration measures your hand (`HandShape`)

During calibration (key `C`) you hold your hand stretched out, palm to the camera, fingers fully open and nothing hidden. For about a
second the system collects samples of the world landmarks, pixel landmarks and (if available) depth, and builds the rest hand:
21 landmarks in the hand frame (x along the fingers, z along the palm normal, wrist at the origin).

The metric scale of that hand comes from one of three sources, in order of preference:

| Source | When | Accuracy |
|---|---|---|
| Depth sensor (RealSense) | camera has depth | best; the hand is measured with real 3D points |
| `--hand-length-mm` (`HAND_LENGTH_MM`) | webcam; you measure wrist → middle fingertip with a ruler | good |
| MediaPipe's own scale | fallback | rough (a few cm off); the status text says `(mediapipe)` |

The same measurement gives the hand's reach (wrist → each fingertip), which sets the **finger scale** automatically
(robot finger reach ÷ yours, typically about 1.2) instead of a number you tune by hand.

Calibration fails with a message if the fingers are not stretched or the model cannot follow the hand.

## Step 2: the model (`HandModel`)

A forward-kinematics model of the measured hand with 27 parameters:

| Parameters | Count |
|---|---|
| Wrist rotation and translation (increments from the previous frame) | 6 |
| Finger abduction (spreading), index … pinky | 4 |
| Finger flexion: knuckle, middle and end joint of 4 fingers | 12 |
| Thumb base rotation (3) + thumb knuckle and end flexion | 5 |

Joint angles are limited to human ranges, and the end joint is softly coupled to the middle joint (it bends about 0.7 times as much).

## Step 3: the per-frame fit (`HandFit`)

Each frame a nonlinear least squares (`scipy.optimize.least_squares`, robust soft-L1 loss) adjusts the 27 parameters to minimise:

| Term | Meaning | Weight / note |
|---|---|---|
| **2D reprojection** | model landmark projected with the camera intrinsics vs MediaPipe pixel | σ 2.5 px; hidden landmarks trusted 2.5× less but still used |
| **MediaPipe z, visible only** | relative depth among landmarks the model says are not hidden | σ 2 cm; weak tie-breaker for a flat palm that looks the same tilted either way |
| **Depth sensor at the wrist** | metric distance, only if the model can see the wrist and the reading is within 25 cm of the tracked wrist | σ 3 cm; guards against the sensor reading the background |
| **Temporal priors** | wrist rotation ≤ ~0.15 rad/frame, wrist motion ≤ ~3 cm/frame, joints stay near last frame | keeps a hidden finger holding its pose and the flat palm from jittering |
| **Finger coupling** | end joint ≈ 0.7 × middle joint | removes unobservable freedom |

**Visibility** (`visibility()`): the model decides which landmarks are hidden by checking whether another part of the hand is in front of
them along the camera ray. Bones are treated as capsules (radius 9 mm), the palm as four triangles, and a hider must be at least 12 mm
closer to the camera. Hidden landmarks lose their `z` term and get a larger 2D tolerance.

**First frame** (`_first`): a good starting point matters because the problem has a mirror-image ambiguity (a flat hand and its reflection
in depth project nearly identically). The first frame tries many starts: the palm orientation from MediaPipe (and its mirror), the 24
cube rotations, each with open and with closed fingers, with the distance estimated from the bone lengths. The best 8 are refined and the
winner is chosen by 2D fit, then by the lowest depth disagreement. Takes about 0.6 s once.

**Tracking** is then incremental from the previous frame, about 14 ms per frame.

**Watchdogs**
- If the depth disagreement stays above 3.5 cm for 6 frames in a row, the model is probably on the mirror solution: the multi-start search runs again.
- If the median reprojection error is above 12 px (on a 640 px image) or fewer than 6 landmarks are visible, the result is flagged
  **unreliable**: the robot holds and the window says `HAND UNRELIABLE (holding)`.
- If the hand is lost for 15 frames the fitter resets.

## What comes out (`FitResult`)

- palm rotation and wrist position (metric, in camera frame)
- all 21 landmarks in camera frame (`landmarks`) and in hand frame (`hand_landmarks`)
- which are visible, the 2D error in px, the depth disagreement, and an `ok` flag
- `tip_vectors`: wrist → 5 fingertips in the hand frame, which is what dex-retargeting consumes

The pipeline uses the wrist position (scaled and clutched) and rotation for the arm, and `tip_vectors` for the fingers.
`--tracking raw` bypasses all of this and uses MediaPipe's 3D directly (for comparison).

## Results so far

Synthetic hand, 2D noise 1.5 px, depth noise 1 cm (visible) / 5 cm (hidden):

| | Hand model | Raw MediaPipe-style |
|---|---|---|
| Mean palm rotation error | 6.4° | 28° |
| Distance error (end-on poses) | ≤ 1.9 cm | up to 36 cm |
| Fingertip error | 5–15 mm | 20–140 mm |
| First-frame mirror failures | 0 / 150 | – |

**Known limit:** a flat open palm facing the camera is no better than raw (about 10° tilt). Tilt of a flat palm is almost invisible in 2D,
so it depends on MediaPipe's `z`. The temporal prior keeps the jitter low (3.7° mean) but cannot remove this bias.

## What has not been verified

- Real MediaPipe landmarks under occlusion. The design assumes its 2D stays good when fingers are hidden; that needs real recordings.
- Real MediaPipe `z` noise levels (the σ values above are estimates).
- The webcam and RealSense capture paths end to end with the model.
- Accuracy with a hand much farther from the camera (small hand in the image, 2D noise in px matters more).

## How to check it on real data

```bash
# on the VM: record a take (in dexhand/.env)
RECORD=recordings/take1.npz
# end-on fingers-to-camera, a fist, a flat palm, a hand turning

# analyse (also runs on this machine)
python tools/analyse_recording.py recordings/take1.npz
```

The analysis reports the reprojection error, how often the hand is flagged unreliable, the depth disagreement and jitter, and can
plot them. `--replay` plays a recording back through the whole pipeline for tuning without a camera.
