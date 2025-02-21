import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_koch_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation.state": np.random.rand(6),
        "observation.images.front": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.low": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.top": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.back_near_tractor": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class KochInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0  # We don't mask for pi0-FAST.

        # Get the state. We are padding from 8 to the model action dim.
        # For pi0-FAST, we don't pad the state (action_dim = 7, which is < 8, so pad is skipped).
        state = transforms.pad_to_dim(data["observation.state"], self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        front_image = _parse_image(data["observation.images.front"])
        low_image = _parse_image(data["observation.images.low"])

        #TODO (Ben): should we be skipping top_image (not using it here)
        # top_image = _parse_image(data["observation.images.top"])
        back_near_tractor_image = _parse_image(data["observation.images.back_near_tractor"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": low_image,
                "right_wrist_0_rgb": back_near_tractor_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            # We are padding from 7 to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class KochOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 6 dims.
        return {"actions": np.asarray(data["actions"][:, :6])}
