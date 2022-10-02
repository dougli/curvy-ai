from typing import Optional

import torch
from torch.distributions import Categorical


class CategoricalMasked(Categorical):
    def __init__(
        self,
        logits: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        probs=None,
        validate_args=None,
    ):
        self.masks = masks
        if not self.masks:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            logits = torch.where(self.masks, logits, -1e8)
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if not self.masks:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs  # type: ignore
        p_log_p = torch.where(self.masks, p_log_p, 0.0)
        return -p_log_p.sum(-1)
