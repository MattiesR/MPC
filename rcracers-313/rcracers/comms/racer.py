"""
This module consists of a wrapper that handles communication with the RC car. 

See also
    :py:class:`rcracers.comms.serial_comm.Transmitter`, 
    :py:class:`rcracers.comms.serial_comm.MsgStruct`

Example
    >>> with Racer('COM5', sample_period=0.01) as racer:
    >>>     for i in range(1000):
    >>>         racer.apply(drive=0.25, steer=0.5)

    This creates a racer context with a sample period of 0.01. 
    We can apply a drive input between -1 and 1 and a 
    steer input between -1 and 1.

.. warning::
    Note that, when a sample period is set, the apply method
    will block until the sample period has passed, before
    actually applying the input. 
"""

from .transmitter import Transmitter, MsgStruct, serial
from serial.threaded import ReaderThread
import logging

import time

NB_CHANNELS = 2


class Racer:
    """
    The Racer wrapper.

    It connects to the race car and sends control actions at a fixed sample period.
    """
    def __init__(
        self, port: str, sample_period: float = -1, baud_rate: int = 9600
    ) -> None:
        """
        Create a new instance of this racer.

        Args:
            - port (str): the COM port with the dongle (e.g., COM5)
            - sample_period (float): the sample period at which actions are transmitted
            - baud_rate (int): baud rate of serial (default = 9600)
        """
        self.__port = port
        self.__baud_rate = baud_rate
        self.__sample_period = sample_period
        self.__serial = None
        self.__transmitter = None
        self.__reader = None
        self.__time_last_send = {}
        self.__num_channels = 0

    def __enter__(self):
        self.__serial = serial.Serial(self.__port, self.__baud_rate, timeout=0)
        self.__reader = ReaderThread(self.__serial, Transmitter)
        self.__transmitter = self.__reader.__enter__()
        self.__transmitter.configure()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i in range(NB_CHANNELS):
            self.apply(0, 0, i)
        self.__reader.__exit__(exc_type, exc_val, exc_tb)

    def close(self):
        for i in range(NB_CHANNELS):
            self.apply(0, 0, i)
        self.__reader.close()

    def apply(self, drive: float = 0, steer: float = 0, channel: int = 0):
        """
        Apply a control action.

        Args:
            - drive (float[-1, 1]): throttle applied to back motors
            - steer (float[-1, 1]): angle of front wheels (0 = center)
            - channel (1 or 0): the channel to send your message on (used when controlling two race cars)
        """
        if self.__transmitter is None:
            raise Exception("Please create a Racer using a with expression.")

        if self.__sample_period > 0:
            if channel not in self.__time_last_send:
                self.__time_last_send[channel] = -1
                self.__num_channels += 1

            if self.__time_last_send[channel] < 0:
                time_to_sleep = 0
            else:
                time_to_sleep = self.__sample_period - (
                    time.time() - self.__time_last_send[channel]
                )

            while time_to_sleep < 0:
                logging.warn(
                    "Warning: Did not receive message in time. Increase sample_period or reduce computation time."
                )
                self.__time_last_send[channel] += self.__sample_period
                time_to_sleep = self.__sample_period - (
                    time.time() - self.__time_last_send[channel]
                )

            time.sleep(1 / self.__num_channels * time_to_sleep)
            self.__time_last_send[channel] = time.time()

        reverse = drive < 0
        drive = min(255, max(0, int(abs(drive) * 255)))
        steer = min(255, max(1, int(0.5 * (steer + 1) * 255)))
        self.__transmitter.send(
            MsgStruct(drive=drive, steer=steer, reverse=reverse, channel=channel)
        )

    def get_debug(self, timeout: float = 1, channel=0):
        """
        Request debug info from dongle.

        The resulting data will be printed automatically.

        Args:
            timeout (float): timout of wait

        Returns:
            (str): the received message (or None if nothing was received)
        """
        return self.__transmitter.send_and_wait(
            MsgStruct(display=2, channel=channel), timeout=timeout
        )
