# Tuning on real hardware

Nothing here replaces measuring. Order of work:

1. **`g1tele-doctor operator`**: USB 3, IMU, MediaPipe, display.
2. **Record a session** with the camera stream: `g1tele-perception --record-rgbd recordings/s1 --no-gui`, do the calibration poses, then reach around the way your tasks need, a few minutes.
3. **Compare settings on the same data** (no need to stand there again):
   ```bash
   g1tele-bench replay recordings/s1
   g1tele-bench replay recordings/s1 --compare perception.filters.wrist_min_cutoff=0.6,1,1.5
   ```
   Without ground truth it reports high-frequency noise per wrist (`*_hf_noise_mm`), `TRACKING/HELD/LOST` shares and per-frame time. Lower noise at equal tracking share is better, but it hides lag: look at the live window too.
4. **With ground truth** (an AprilTag or marker on the wrist, see below) accuracy and lag become measurable on real data. The synthetic benchmark (`g1tele-bench synthetic --tilt 10`) stays as the regression test for the algorithms.

## Knobs, in the order they matter

| Symptom | Knob |
|---|---|
| Whole arm is offset/scaled wrong in one direction | calibration: "up" source (`g1tele-perception` logs it; IMU or floor beats arms). Tilt the camera down 10-15° or use a D435i/D455 |
| Wrist shakes when you hold still | `perception.filters.wrist_min_cutoff` lower (1.5 -> 0.8), but see "slow wander" in the README: the floor is the depth noise |
| Wrist lags when you move fast | `wrist_beta` higher (20 -> 40) |
| Wrist is pulled to a wrong depth when the arm is in front of the body | `perception.bone_sigma` smaller (bones decide more); check calibration `bone_spread_cm` in `profiles/<name>.json` (> 2 cm: redo it in better light, nearer) |
| Wrist jumps when a hand covers it | `perception.depth_sigma` smaller (trust depth less) |
| Joints sit 3-5 cm too near the camera | `perception.radii` (shoulder/elbow/wrist): the limb radii; measure your arm |
| Hand orientation flipped | `perception.hand_axes_sign` (the hand model's axis convention was inferred from the old pipeline, not re-verified on the new one) |
| Fingers do not close fully | redo the calibration's open-hand/fist poses (`c`), or `perception.hands=false` for arm-only tasks |
| Too slow | `perception.pose_model_complexity=0`, `camera.fps=60`, `perception.hands=false` |
| Robot elbow ignores yours | `ik.elbow_weight` higher |
| Robot wrist ignores your palm | `ik.rot_weight` higher |
