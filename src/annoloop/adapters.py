"""Blocking GPU adapter contract. No fake/default detector implementation."""
from abc import ABC, abstractmethod


class DetectorAdapter(ABC):
    @abstractmethod
    def predict_target(self, checkpoint, frames):
        """Return frame_id -> {boxes:[id,class_name,box,score,cls_score,iou_score]}.

        Exact requested frame set, including empty frames. No target annotations.
        """

    @abstractmethod
    def predict_source_validation(self, checkpoint):
        """Return {domain:'source_validation', boxes:[..., source_gt_iou]}.

        Read labelled source validation only, not target holdout/gold.
        """

    @abstractmethod
    def train_target(self, checkpoint, labels_path, output, policy):
        """Block until trainer exits and return trained receipt.

        Must register its own output with training_launcher, honor signed ignore
        regions, save every 10 epochs and final. Return checkpoint, optimizer_steps,
        target_gold_used=False, shared_labels_sha256, checkpoint_sha256.
        No second GPU worker may remain live when this method returns.
        """
