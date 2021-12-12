import reader.gen.messages_robocup_ssl_wrapper_pb2 as SSL_Wrapper
import reader.gen.ssl_referee_pb2 as SSL_referee
from reader.MessageType import MessageType


class Reader:

    msg_timestamp = None
    msg_type = None
    msg_size = None

    wrapper_packet = None
    referee_packet = None

    def __init__(self, path):
        self.path = path
        self.file = open(path, "rb")

    def read_header(self):
        name = self.file.read(12)
        version = self.file.read(4)
        print(name.decode("ascii"))
        print(int.from_bytes(version, signed=True, byteorder='big'))

    def has_next(self):
        m_timestamp = self.file.read(8)
        m_type = self.file.read(4)
        m_size = self.file.read(4)

        if m_timestamp == b"" or m_type == b"" or m_size == b"":
            return False

        self.msg_timestamp = int.from_bytes(m_timestamp, signed=True, byteorder='big')
        self.msg_type = int.from_bytes(m_type, signed=True, byteorder='big')
        self.msg_size = int.from_bytes(m_size, signed=True, byteorder='big')

        return True

    def decode_msg(self):
        byte = self.file.read(self.msg_size)

        if self.msg_type == MessageType.MESSAGE_SSL_VISION_2010.value or \
                self.msg_size == MessageType.MESSAGE_SSL_VISION_2014.value:
            self.wrapper_packet = SSL_Wrapper.SSL_WrapperPacket()
            self.wrapper_packet.ParseFromString(byte)
        elif self.msg_type == MessageType.MESSAGE_SSL_REFBOX_2013.value:
            self.referee_packet = SSL_referee.SSL_Referee()
            self.referee_packet.ParseFromString(byte)

        return MessageType(self.msg_type)

    def get_referee_packet(self):
        return self.referee_packet

    def get_wrapper_packet(self):
        return self.wrapper_packet
