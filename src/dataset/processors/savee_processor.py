from typing import Tuple
from .base_processor import BaseDatasetLoader


class SaveeLoader(BaseDatasetLoader):
    EMOTION_MAP = {
        "a": "ANG",
        "d": "DIS",
        "f": "FEA",
        "h": "HAP",
        "n": "NEU",
        "sa": "SAD",
        "su": "SUR",
    }

    CLASSES_ = {
        "ANG": 1,
        "DIS": 0,
        "FEA": 0,
        "HAP": 0,
        "NEU": 0,
        "SAD": 0,
        "SUR": 0,
    }

    SAMPLING_RATE = 44050

    def parse_label(self, filename: str) -> int:
        """
        Extracts the class label from the filename.

        Example:
            DC_a01.wav
        """

        parts = filename.split("_")

        # Remove last two digits (e.g. '01')
        emotion_part = parts[-1].replace(".wav", "")[:-2]

        emotion = self.EMOTION_MAP.get(emotion_part, "NEU")

        label = self.CLASSES_.get(emotion, 0)

        return label

    def parse_actor_id(self, filename: str) -> str:
        """
        Extracts the actor ID from the filename.
        """

        parts = filename.split("_")

        actor_id = parts[0]

        return actor_id

    def parse_sex(self, filename: str) -> str:
        """
        SAVEE dataset contains only male actors.
        """

        return "M"
