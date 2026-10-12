# Tuning on real hardware

Nothing here replaces measuring. Order of work:

1. **`g1tele-doctor operator`**: USB 3, IMU, MediaPipe, display.
2. **Record a session** with the camera stream: `g1tele-perception --record-rgbd recordings/s1 --no-gui`, do the calibration poses, then reach around the way your tasks need, a few minutes.
3. **Test with realistic noise**: `g1tele-bench synthetic --tilt 10 --pixel-noise 4 --depth-noise 0.004` (the default synthetic noise is cleaner than MediaPipe and RealSense; filters tuned on it shake on real data).
4. **Compare settings on the same data** (no need to stand there again):
   ```bash
   g1tele-bench replay recordings/s1
   g1tele-bench replay recordings/s1 --compare perception.filters.wrist_min_cutoff=0.6,1,1.5
   ```
   Without ground truth it reports high-frequency noise per wrist (`*_hf_noise_mm`), `TRACKING/HELD/LOST` shares and per-frame time. Lower noise at equal tracking share is better, but it hides lag: look at the live window too.
5. **With ground truth** (an AprilTag or marker on the wrist, see below) accuracy and lag become measurable on real data. The synthetic benchmark (`g1tele-bench synthetic --tilt 10`) stays as the regression test for the algorithms.

## Knobs, in the order they matter

| Symptom | Knob |
|---|---|
| Whole arm is offset/scaled wrong in one direction | calibration: "up" source (`g1tele-perception` logs it; IMU or floor beats arms). Tilt the camera down 10-15° or use a D435i/D455 |
| Wrist shakes when you hold still | `perception.filters.wrist_beta` lower (5 -> 3), `wrist_min_cutoff` lower, `perception.bone_sigma` higher (0.02 -> 0.03). Compare with `perception.fusion=false` to see if the fusion is the cause |
| Wrist lags when you move fast | `wrist_beta` higher (5 -> 10; more shake) |
| Wrist is pulled to a wrong depth when the arm is in front of the body | `perception.bone_sigma` smaller (bones decide more); check calibration `bone_spread_cm` in `profiles/<name>.json` (> 2 cm: redo it in better light, nearer) |
| Wrist jumps when a hand covers it | `perception.depth_sigma` smaller (trust depth less) |
| Joints sit 3-5 cm too near the camera | `perception.radii` (shoulder/elbow/wrist): the limb radii; measure your arm |
| Hand orientation flipped | `perception.hand_axes_sign` (the hand model's axis convention was inferred from the old pipeline, not re-verified on the new one) |
| Fingers do not close fully | redo the calibration's open-hand/fist poses (`c`), or `perception.hands=false` for arm-only tasks |
| Too slow | `perception.pose_model_complexity=0`, `camera.fps=60`, `perception.hands=false` |
| Robot elbow ignores yours | `ik.elbow_weight` higher |
| Robot wrist ignores your palm | `ik.rot_weight` higher |

## Hands: what changed and how to compare it on the real hardware

Three hand changes are in, all with a switch, so each can be compared against the old behaviour in one run
(`--set key=value`). None of them has been validated on a real hand yet: the numbers so far are from the
synthetic operator only.

| Change | Switch | Old behaviour | Look for |
|---|---|---|---|
| Fingers from fingertip positions, through the Inspire finger kinematics (the "vector" retargeting of dex-retargeting, `retarget/hand_ik.py`) | `ik.hand_mode=vector` (default) / `curl` | six curls mapped linearly to the joints | thumb opposition, a fist that closes fully, fingers that follow when you pinch |
| Palm normal corrected by a plane fitted to the depth image (`perception/hand_depth.py`) | `perception.hand_depth=true` (default) / `false` | the palm normal comes from the hand model's landmarks only | the wrist no longer flips when the palm turns edge-on; a wrong flip would show as the robot hand snapping to the back-of-hand |
| A finger hidden behind another object keeps its last value | `perception.hand_occlusion=true` / `false` (default) | off | fingers freezing for no reason: that is a false "hidden", raise `perception.occ_front_margin` |

Related knobs: `ik.hand_scale` (robot fingers reach further than yours, relative to palm size),
`ik.hand_huber` (a bad fingertip landmark counts linearly, not squared), `ik.hand_reg` (smoothness of the fingers),
`perception.hand_depth_weight` / `hand_depth_flip_deg` (how far the depth plane may move the palm normal).
The debug dict of the perception pipeline has `hand_palm_source` (`landmarks`, `fused`, `depth`) per hand.

## Reach and hands-only modes

Two ways to steer the robot, switched without touching anything (your hands are busy):

| Mode | What it uses | What moves |
|---|---|---|
| **reach** | MediaPipe Pose at normal distance | the arms follow your arms. The wrist keeps its orientation and the fingers hold (`perception.reach_hands=false`) |
| **hands-only** | MediaPipe Hands on the whole image, you close to the camera | the arm holds; the robot wrist turns like your hand has turned since the start pose, the fingers follow |

**Two nods** switch (nose goes down and back, twice within 1.6 s; `perception.nod_*`). `h` in the window does the same.

1. *reach -> hands-only.* The robot **holds still** while the window guides a calibration: move closer so only your hands are in
   view (`perception.hands_calib_range`) -> open hands -> fists -> hold your start pose. The close-range finger range is
   learnt here. Keep your face in view: it is how you leave.
2. *hands-only -> reach.* Two nods again. The robot **holds** where it is and the window shows circles where your wrists were
   when you left reach mode. Put your wrists on them (`perception.match_tol_m`) and reach mode resumes with no jump.

If the hands never reach the calibration range, or the calibration takes longer than `hands_calib_timeout_s`, it gives up and
goes to the match step. The body model runs at `perception.hands_pose_hz` in hands-only mode, only to see the nods.
Look for: nods that fire by accident while you watch the task (raise `nod_amplitude`), nods that do not fire (lower it),
the left/right hands swapped in hands-only mode (`perception.hands_swap_handedness=false`).

