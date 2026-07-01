import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# AgiBot G1 bimanual action/state layout (18 dims):
#   0..13  arm joints (14 total, both arms)
#   14,15  waist joints
#   16     left effector (normalized /120 during conversion)
#   17     right effector (normalized /120 during conversion)
AGIBOT_G1_ACTION_DIM = 18


def make_agibot_g1_example() -> dict:
    """Creates a random input example for the AgiBot G1 policy (matches the inference key format)."""
    return {
        "observation/state": np.random.rand(AGIBOT_G1_ACTION_DIM),
        "observation/cam_head_color": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/cam_hand_left": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/cam_hand_right": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "pick and place objects using the omnipicker",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AgibotG1Inputs(transforms.DataTransformFn):
    """Converts AgiBot G1 observations into the format expected by the model."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        head_image = _parse_image(data["observation/cam_head_color"])
        left_hand = _parse_image(data["observation/cam_hand_left"])
        right_hand = _parse_image(data["observation/cam_hand_right"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": left_hand,
                "right_wrist_0_rgb": right_hand,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AgibotG1Outputs(transforms.DataTransformFn):
    """Converts model outputs back to the AgiBot G1 action space (inference only)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :AGIBOT_G1_ACTION_DIM])}
