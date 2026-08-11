from .base_processor import BaseDatasetLoader


class CremaDLoader(BaseDatasetLoader):
    CLASSES_ = {
        "ANG": 1,
        "FEA": 0,
        "DIS": 0,
        "HAP": 0,
        "NEU": 0,
        "SAD": 0,
    }

    TENSES_ = {
        "XX": 0,  # Undefined
        "HI": 1,  # High intensity voice
        "LO": 0,  # Low intensity voice
        "MD": 1,  # Medium/normal intensity voice
    }

    female_id_set = {
        "1002",
        "1003",
        "1004",
        "1006",
        "1007",
        "1008",
        "1009",
        "1010",
        "1012",
        "1013",
        "1018",
        "1020",
        "1021",
        "1024",
        "1025",
        "1028",
        "1029",
        "1030",
        "1037",
        "1043",
        "1046",
        "1047",
        "1049",
        "1052",
        "1053",
        "1054",
        "1055",
        "1056",
        "1058",
        "1060",
        "1061",
        "1063",
        "1072",
        "1073",
        "1074",
        "1075",
        "1076",
        "1078",
        "1079",
        "1082",
        "1084",
        "1089",
        "1091",
    }

    SAMPLING_RATE = 8000

    def parse_label(self, filename: str) -> int:
        """
        Extracts the class label from the filename.

        Example:
            1001_IEO_ANG_HI.wav
        """
        parts = filename.split("_")

        emotion = parts[-2]
        tense = parts[-1].replace(".wav", "")

        label = self.CLASSES_.get(emotion, 0)

        # Example alternative logic:
        # if emotion in ["DIS"]:
        #     label = self.TENSES_.get(tense, 0)

        return label

    def parse_actor_id(self, filename: str) -> str:
        """
        Extracts the actor ID from the filename.

        Example:
            1001_IEO_ANG_HI.wav
        """
        parts = filename.split("_")

        # First part is the actor ID
        actor_id = parts[0]

        return actor_id

    def parse_sex(self, filename: str) -> str:
        """
        Determines actor sex based on actor ID.
        """
        parts = filename.split("_")

        actor_id = parts[0]

        return "F" if actor_id in self.female_id_set else "M"
