"""
Unit tests for ALP Encoder.
"""

import pytest
import struct
from broadcast.alpencoder import ALPEncoder, ALPPacket
from broadcast.utils import crc24q

def create_mock_rtcm_frame(message_type: int = 1005, payload_size: int = 20) -> bytes:
    """Create a mock RTCM 3.x frame for testing."""
    # RTCM header: [preamble (0xD3)] [reserved + length (2 bytes)] [message_type (12 bits)] [payload]
    preamble = 0xD3
    length = payload_size + 3  # message type (12 bits = 1.5 bytes rounded to 2) + payload
    
    # Simplified: [preamble][length_high][length_low][msg_type_high][msg_type_low][payload][crc]
    header = struct.pack('!BBB', preamble, (length >> 8) & 0x03, length & 0xFF)
    msg_header = struct.pack('!H', (message_type << 4) & 0xFFF0)
    payload = b'\x00' * payload_size
    
    frame_without_crc = header + msg_header + payload
    crc = crc24q(frame_without_crc)
    crc_bytes = struct.pack('!I', crc)[1:]  # Take last 3 bytes
    
    return frame_without_crc + crc_bytes

class TestALPEncoder:
    """Test suite for ALP Encoder."""
    
    def test_encoder_initialization(self):
        """Test encoder initializes correctly."""
        encoder = ALPEncoder()
        assert encoder.packet_count == 0
    
    def test_encode_rtcm_basic(self):
        """Test basic RTCM encoding."""
        encoder = ALPEncoder()
        rtcm_frame = create_mock_rtcm_frame()
        
        packet = encoder.encode_rtcm(rtcm_frame)
        
        assert isinstance(packet, ALPPacket)
        assert packet.payload == rtcm_frame
        assert packet.payload_length == len(rtcm_frame)
        assert encoder.packet_count == 1
    
    def test_encode_rtcm_multiple(self):
        """Test encoding multiple RTCM frames."""
        encoder = ALPEncoder()
        frames = [create_mock_rtcm_frame() for _ in range(5)]
        
        packets = [encoder.encode_rtcm(frame) for frame in frames]
        
        assert len(packets) == 5
        assert encoder.packet_count == 5
        for packet, frame in zip(packets, frames):
            assert packet.payload == frame
    
    def test_encode_bitmap(self):
        """Test bitmap tile encoding."""
        encoder = ALPEncoder()
        # 100x100 grayscale bitmap = 10,000 bytes
        bitmap_data = b'\xFF' * 10000
        tile_id = 42
        
        packet = encoder.encode_bitmap(bitmap_data, tile_id)
        
        assert isinstance(packet, ALPPacket)
        # Payload should include 4-byte tile_id + bitmap_data
        assert packet.payload_length == 4 + len(bitmap_data)
        assert encoder.packet_count == 1
    
    def test_encode_batch(self):
        """Test batch encoding of RTCM and bitmaps."""
        encoder = ALPEncoder()
        rtcm_frames = [create_mock_rtcm_frame() for _ in range(3)]
        bitmap_tiles = [(b'\x00' * 100, i) for i in range(2)]
        
        packets = encoder.encode_batch(rtcm_frames, bitmap_tiles)
        
        assert len(packets) == 5  # 3 RTCM + 2 bitmaps
        assert encoder.packet_count == 5
    
    def test_packet_to_bytes(self):
        """Test packet serialization."""
        encoder = ALPEncoder()
        rtcm_frame = create_mock_rtcm_frame()
        packet = encoder.encode_rtcm(rtcm_frame)
        
        serialized = packet.to_bytes()
        
        assert isinstance(serialized, bytes)
        assert len(serialized) == len(packet.header) + len(packet.payload)
    
    def test_get_stats(self):
        """Test statistics reporting."""
        encoder = ALPEncoder()
        encoder.encode_rtcm(create_mock_rtcm_frame())
        encoder.encode_rtcm(create_mock_rtcm_frame())
        
        stats = encoder.get_stats()
        
        assert stats['total_packets_encoded'] == 2

if __name__ == '__main__':
    pytest.main([__file__, '-v'])