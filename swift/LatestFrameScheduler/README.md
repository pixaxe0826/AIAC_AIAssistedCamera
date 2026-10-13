# Latest-frame scheduler

SwiftPM extraction of the original actor-based scheduler and its non-awaiting frame-consumer dispatch. It has no AVFoundation, Vision, UIKit, or external package dependency. Camera frames, metadata, recommendations, and providers are represented by minimal protocols.

```sh
cd swift/LatestFrameScheduler
swift test
```

The actor allows one in-flight processor call and replaces the single pending frame with the newest submission. Its recommendation adapter retains the original contract-version, frame identity, timestamp, camera-generation, and expiry checks.

`ScheduledFrameConsumer.submit` dispatches to the actor through a task and returns without waiting for inference, matching the camera-facing submission boundary. Processor suspension therefore does not make the caller wait for a model result.

Synthetic tests cover in-flight exclusivity, latest-only pending frames, out-of-order input, generation changes, expiry, and non-awaiting submission. These tests do not exercise a camera device or establish on-device inference latency.
