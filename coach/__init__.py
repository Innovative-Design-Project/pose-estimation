"""Basketball practice coach: tracking, event detection, form metrics, coaching and storage.

Pipeline per analysed frame (see engine.CoachEngine):
  pose (17 COCO keypoints) -> PlayerTracker + KeypointSmoother -> BallTracker -> hands
  -> ShotDetector / DribbleCounter -> shot metrics -> coaching cue -> voice + SessionStore
"""
