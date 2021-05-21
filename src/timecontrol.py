"""
Everything related to time control
"""
import json
import time
from typing import Optional

# TODO finish this interface and add it into the engine note that we need support for
#  fail high/fail low triggers
#  coming off of opening book
#  extensions (check, recapture, mate threat)


class BaseTimeController:
    """
    Base class for all time controllers.
    """
    def __init__(self, config_filepath: str):
        self.config_filepath: str = config_filepath
        with open(self.config_filepath, "r+") as config_f:
            self.config = json.load(config_f)

        self._parse_base_config()
        self._parse_config()

    def _parse_base_config(self):
        self.name = self.config["name"]
        self.version = self.config["version"]

    def _parse_config(self):
        raise NotImplementedError("_parse_config must be implemented by all time controllers!")

    def on_depth_completed(self, time_to_complete: float, fail_highs: int, fail_lows: int) -> bool:
        pass


class EBFFixedTimeController:
    """
    Effective branching factor time controller for fixed time control. Simply makes it more convenient
    when, if, for example, the searcher is given 30 seconds and in 15 seconds it gets to depth 8 but predicts
    that it will take 45 seconds to search to depth 9 then it shouldn't search depth 9.
    Note that all calculations in this class are done in seconds.
    """
    def __init__(self, config_filepath: str):
        """
        Instantiates but does not start a time controller.
        :param config_filepath: Filepath to the config file
        """
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.prev_depth_time: Optional[float] = None
        self.config_filepath: str = config_filepath

        with open(self.config_filepath, "r+") as config_f:
            self.config = json.load(config_f)

    def _parse_config(self):
        self.name = self.config["name"]
        self.version = self.config["version"]
        self.tc_factor: float = self.config["tc_factor"]

    def start(self, fixed_time: float):
        """
        Starts the time controller for the current move.
        :return: None
        """
        self.start_time = time.time()
        self.end_time = self.start_time + fixed_time

    def on_depth_completed(self, time_to_complete: float) -> bool:
        """
        Must be called after each depth is completed and will return whether to continue on to the next depth.
        :param time_to_complete: Time taken to complete the previous depth
        :return: True if the search should continue to the next depth, False otherwise
        """
        if self.prev_depth_time is not None and self.prev_depth_time > 0:
            time_ebf = time_to_complete / self.prev_depth_time
            predicted_time_next_iteration = time_ebf * time_to_complete
            time_remaining = self.end_time - time.time()
            return predicted_time_next_iteration <= time_remaining * self.tc_factor
        self.prev_depth_time = time_to_complete
        return True

