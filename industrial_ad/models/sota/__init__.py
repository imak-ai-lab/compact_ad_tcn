"""Third-party SOTA anomaly detectors adapted to the project protocol.

These are ports of the official author implementations,
adapted to this repository's interface:

- input/target are batched windows shaped (batch, time_steps, channels);
- the training objective is exposed through `loss_phases()` so adversarial /
  minimax recipes keep their original update structure;
- the author's own anomaly score is exposed through `anomaly_error()`, which
  returns a per-sample error feature vector; the configured score estimator
  then reduces it to a scalar ("mean" reproduces the author's score, while
  "mahalanobis" applies this project's scoring scheme to the same errors).

Nothing here is used by the paper's own model families: the shared training
loop only takes the new code path when a model defines `loss_phases`.
"""

from industrial_ad.models.sota.anomaly_transformer import AnomalyTransformerDetector
from industrial_ad.models.sota.tranad import TranAD
from industrial_ad.models.sota.usad import USAD

__all__ = ["AnomalyTransformerDetector", "TranAD", "USAD"]
