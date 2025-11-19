"""
This module enables communication to the car using the transmitter dongle. 
Communication happens over a serial port in five byte messages:

.. list-table:: Package structure
    :widths: 50 50 50 50 50
    :header-rows: 0

    *   - start
        - drive
        - steer
        - head
        - end

Here the start and end bytes are added for safety and always have the same value 
(``0x5B`` and ``0x5D`` respectively). The drive is a ``uint8_t`` configuring 
the throttle and the speed ``uint8_t``

See also
    `serial.threaded.LineReader`_, `serial.threaded.ReaderThread`_, `serial.Serial`_

Example
    First a serial object must be created

    >>> ser = serial.Serial('COM5', 9600, timeout=0)

    With this serial object, we can create the transmitter.
    
    A Transmitter acts as a resource, so a with expression should be used
    to create one:

    >>> with ReaderThread(ser, Transmitter) as transmitter:
    >>>     transmitter.configure()

    Before starting communication we need to give it some time to start up.
    Then we ping the dongle to see if it is active. 
    The configure method handles this. If no response if received 
    it will raise an Exception. 
    
    Sending messages happens through the MsgStruct class. 
    To move the car we send drive commands (between ``0`` and ``255``)
    
    >>> transmitter.send(MsgStruct(drive=120))

    To reverse we use the reverse flag
    
    >>> transmitter.send(MsgStruct(drive=120, reverse=True))

    .. warning::
        There is some static friction on the motor, so a value of drive lower than ``50`` will not
        start the car. Using high values of drive meanwhile will quickly drain the motor. 
    
    To steer the car we can send a reference angle (between ``1`` for right-most angle and ``255`` for left-most angle; a value of ``0`` 
    means the angle is not actively controlled).
    
    >>> transmitter.send(MsgStruct(steer=30))
    
    We can of course send drive and steer commands simultaneously. To get debug info we can use the disp flag.

    >>> transmitter.sendAndWait(MsgStruct(display=2), timeout=1)

    We use ``sendAndWait`` to specify that we should wait for an answer. A display of ``0``
    means nothing will be printed; ``1`` is used to ping the 
    dongle; and ``2`` makes the dongle print debug info. 
    The debug info is printed in an array, the layout of which is given below.
    
.. list-table:: Debug info structure
    :widths: 50 50 50 50 50
    :header-rows: 0

    *   - msg received
        - rf fail ratio
        - avg time/package
        - avg time/rf msg
        - max buffer size


.. _serial.threaded.LineReader:
    https://pyserial.readthedocs.io/en/latest/pyserial_api.html#serial.threaded.LineReader
.. _serial.threaded.ReaderThread:
    https://pyserial.readthedocs.io/en/latest/pyserial_api.html#serial.threaded.ReaderThread
.. _serial.Serial:
    https://pyserial.readthedocs.io/en/latest/pyserial_api.html#serial.Serial
"""


import ctypes

import serial
from serial.threaded import LineReader, ReaderThread

import threading
import time
import sys

import logging


class MsgStruct:
    """
    Message Structure.

    Contains message data and utils for transmission.
    """

    __START_BYTE = bytes.fromhex("5B")
    __END_BYTE = bytes.fromhex("5D")

    __D_SHIFT = 1
    __R_SHIFT = 7
    __C_SHIFT = 3

    CENTER_STEER = int(256 / 2)  # value for center steering
    LEFT_STEER = int(256 / 2)  # value for maximum left steering
    RIGHT_STEER = int(256 / 2)  # value for maximum right steering

    def __init__(
        self,
        drive: int = 0,
        steer: int = 0,
        reverse: bool = False,
        display: int = 0,
        channel=0,
    ) -> None:
        """
        Initialize this MsgStruct.

        Args:
            drive (int): value between 0 and 255 (min throttle, max throttle).
            steer (int): value between 0 and 255 (right, left).
            reverse (bool): reverse the action of drive.
            display (int): display value (0, 1, 2, 3) = (none, ping, dump, car)
        """
        self.drive = ctypes.c_uint8(drive)
        self.steer = ctypes.c_uint8(steer)
        self.reverse = reverse
        self.channel = channel
        self.display = display

    def __build_header(self) -> ctypes.c_uint8:
        """
        Construct the header of this message:

        format: r|000|c|d|v
            r: reverse (1 bit)
            d: display mode (2 bits)
                00 = none
                01 = ping
                10 = dump
                11 = car
            c: channel
            v: message validity (1 bit)
                overwritten by transmitter
        """
        return ctypes.c_uint8(
            (self.display << self.__D_SHIFT)
            + (self.reverse << self.__R_SHIFT)
            + ((1 if self.channel > 0 else 0) << self.__C_SHIFT)
        )

    def __get_bytes(self) -> bytes:
        """
        Get the bytes of the data in this message.
        """
        # print(self.__build_header())
        # print(bytes(self.drive) + bytes(self.steer) + bytes(self.__build_header()))
        return bytes(self.drive) + bytes(self.steer) + bytes(self.__build_header())

    def pack(self):
        """
        Get the packages message, ready for transmission.

        Return:
            bytes: package of 5 bytes.
        """
        return self.__START_BYTE + self.__get_bytes() + self.__END_BYTE


class Transmitter(LineReader):
    """
    Contains utilities for sending and receiving messages over the serial port.
    """

    START_DELAY = 3  # seconds to wait before connection is usually ready

    def __init__(self) -> None:
        """
        Create an instance of this Tranmistter.
        """
        super().__init__()
        self.receivedEvent = threading.Event()
        self.lastMessage = None

    def configure(self, timeout: float = 1, retries: int = 3):
        """
        Configure the transmitter dongle.

        Attempts to ping the dongle and waits for a reply

        Args:
            timeout (float): reply timeout in seconds
            retries (int): amount of retries
        """
        # wait to give the serial some time to start
        time.sleep(Transmitter.START_DELAY)

        # try pinging the transmitter
        if not self.ping(timeout=timeout, retries=retries):
            raise Exception("ping failed.")

        logging.debug("agent active. sending packages ...")

    def connection_made(self, transport):
        """
        Called when the connection to the serial is created.
        """
        super(Transmitter, self).connection_made(transport)
        logging.debug("port opened")

    def handle_line(self, data):
        """
        Called when a message from the serial is received.

        Args:
            data: the reply from the dongle.
        """
        self.receivedEvent.set()
        self.lastMessage = data
        logging.debug("[serial] {}".format(repr(data)))

    def connection_lost(self, exc):
        """
        Called when the connection to the serial is lost.
        """
        if exc:
            serial.traceback.print_exc(exc)
        logging.debug("port closed")

    def write(self, msg):
        """
        Write a binary message to the Serial.
        """
        self.transport.write(msg)

    def send(self, msg: MsgStruct):
        """
        Send a MsgStruct to the Serial.
        """
        self.write(msg.pack())

    def send_and_wait(self, msg: MsgStruct, timeout=1, retries=1):
        """
        Send a message and wait for the reply.

        Args:
            timeout (int): timeout on the wait.
            retries (int): amount of times a messages is send before giving up.
        """
        for _ in range(retries):
            self.receivedEvent.clear()
            self.send(msg)
            if self.receivedEvent.wait(timeout):
                return self.lastMessage
        return None

    def ping(self, timeout: float = 1, retries: int = 1) -> bool:
        """
        Ping the arduino and wait for a reply.

        Args:
            timeout (float): timeout on the wait.
            retries (int): amount of times a messages is send before giving up.
        """
        for _ in range(retries):
            msg = self.send_and_wait(MsgStruct(display=1), timeout=timeout)
            if msg == "~":
                return True
        return False


# =================================================================================
# Example usage.
# =================================================================================
if __name__ == "__main__":
    # set logging level
    logging.basicConfig(level=logging.DEBUG)

    # configuration of serial interface
    SERIAL_PORT = "COM5"
    BAUD_RATE = 9600

    # create the serial interface
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)

    # spawn the reader thread (an instance of Transmitter is stored in transmitter).
    with ReaderThread(ser, Transmitter) as transmitter:
        # configure (ping the car)
        transmitter.configure()

        # get the car moving
        transmitter.send(
            MsgStruct(drive=120, steer=MsgStruct.CENTER_STEER, display=0, reverse=True)
        )
        time.sleep(1)

        N = 6
        drives = [150, 70, 70, 70, 70, 70, 0, 0, 0, 0]
        steers = [130, 130, 255, 255, 255, 128, 128, 128, 128, 128]
        for i in range(N):
            # send some messages
            for j in range(20):
                # turns whole range at steer = 22
                # drive overcoming static friction = 56
                transmitter.send(
                    MsgStruct(
                        drive=abs(drives[i]),
                        steer=steers[i],
                        display=0,
                        reverse=drives[i] < 0,
                    )
                )
                time.sleep(0.01)
            time.sleep(0.1)

        # request debug info
        # [number of messages received, RF failure rate,
        #       average time between complete messages (us), average time of RF transmissions (us),
        #       maximum bytes in serial buffer]
        logging.debug("requesting data ... ")
        transmitter.sendAndWait(
            MsgStruct(drive=0, steer=0, display=2, reverse=False), timeout=1
        )
