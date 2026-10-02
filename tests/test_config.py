import json

from app.config import Config


def test_config_file_exists():

    config = Config("config.json")

    assert config.data is not None
    assert isinstance(config.data, dict)


def test_config_nested_values():

    config = Config("config.json")

    model = config.get(
        "detection",
        "model"
    )

    confidence = config.get(
        "detection",
        "confidence"
    )

    frame_skip = config.get(
        "detection",
        "frame_skip"
    )

    assert model is not None
    assert confidence is not None
    assert frame_skip is not None


def test_config_default_value():

    config = Config("config.json")

    value = config.get(
        "this_key_does_not_exist",
        default="DEFAULT_VALUE"
    )

    assert value == "DEFAULT_VALUE"


def test_config_camera_settings():

    config = Config("config.json")

    camera_type = config.get(
        "camera",
        "type",
        default="file"
    )

    assert camera_type in [
        "file",
        "rtsp"
    ]