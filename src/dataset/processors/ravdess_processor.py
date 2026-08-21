from .base_processor import BaseDatasetLoader


class RavdessLoader(BaseDatasetLoader):
    EMOTION_MAP = {
        "01": "NEU",
        "02": "CAL",
        "03": "HAP",
        "04": "SAD",
        "05": "ANG",
        "06": "FEA",
        "07": "DIS",
        "08": "SUR",
    }

    CLASSES_ = {
        "ANG": 1,
        "FEA": 0,
        "DIS": 0,
        "HAP": 0,
        "NEU": 0,
        "CAL": 0,
        "SAD": 0,
        "SUR": 0,
    }

    TENSES_ = {
        "HI": 1,  # High intensity voice
        "MD": 0,  # Medium/normal intensity voice
    }

    SAMPLING_RATE = 16000

    def parse_label(self, filename: str) -> int:
        """
        Extracts the class label from the filename.

        Example:
            03-01-05-01-02-01-12.wav
        """

        parts = filename.replace(".wav", "").split("-")

        emotion_code = parts[2]
        if emotion_code not in self.EMOTION_MAP:
            raise ValueError(
                f"Unsupported RAVDESS emotion code: {emotion_code}"
            )
        emotion = self.EMOTION_MAP[emotion_code]

        label = self.CLASSES_[emotion]

        return label

    def parse_actor_id(self, filename: str) -> str:
        """
        Extracts the actor ID from the filename.

        Example:
            03-01-05-01-02-01-12.wav
        """

        parts = filename.replace(".wav", "").split("-")

        # Last part is the actor ID
        actor_id = parts[-1]

        return actor_id

    def parse_sex(self, filename: str) -> str:
        """
        Determines actor sex based on actor ID.

        Even IDs correspond to female actors,
        odd IDs correspond to male actors.
        """

        parts = filename.replace(".wav", "").split("-")

        actor_id = parts[-1]

        return "F" if int(actor_id) % 2 == 0 else "M"
