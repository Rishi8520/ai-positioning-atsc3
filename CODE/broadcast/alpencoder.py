"""
ALP (ATSC Link-Layer Protocol) Encoder
Encapsulates RTCM correction frames and bitmap tiles into ALP packets.

Based on ATSC A331 specification.
"""

import struct
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

from broadcast.utils import get_timestamp_ms, bytes_to_hex_string
logger = logging.getLogger(__name__)

# ALP Packet Type
ALP_PACKET_TYPE_IPV4 = 0x00
ALP_PACKET_TYPE_COMPRESSED_IPV4 = 0x01
ALP_PACKET_TYPE_SIGNALING = 0x04
ALP_PACKET_TYPE_EXTENSION = 0x07

# ALP Header constants
ALP_HEADER_BASE_SIZE = 2  # bytes
ALP_MAX_PAYLOAD_SIZE = 65535  # bytes

@dataclass
class ALPPacket:
    """ALP packet structure."""
    packet_type: int
    payload_length: int
    payload: bytes
    timestamp: int
    header: bytes
    
    def to_bytes(self) -> bytes:
        """Convert ALP packet to binary format."""
        return self.header + self.payload

    def __len__(self) -> int:
        """Return total packet size."""
        return len(self.header) + len(self.payload)


class ALPEncoder:
    """
    ALP Encoder for ATSC 3.0 broadcast.
    Encapsulates positioning data (RTCM, bitmaps) into ALP packets.
    """
    
    def __init__(self):
        """Initialize ALP encoder."""
        self.packet_count = 0
        logger.info("ALP Encoder initialized")
    
    def encode_rtcm(self, rtcm_frame: bytes) -> ALPPacket:
        """
        Encode RTCM correction frame into ALP packet.
        
        Args:
            rtcm_frame: Raw RTCM 3.x binary frame
        
        Returns:
            ALPPacket containing the RTCM data
        """
        if len(rtcm_frame) > ALP_MAX_PAYLOAD_SIZE:
            raise ValueError(
                f"RTCM frame too large: {len(rtcm_frame)} bytes "
                f"(max {ALP_MAX_PAYLOAD_SIZE})"
            )
        
        # Create ALP header
        # Simplified header: [packet_type (1 byte)] [length (2 bytes, big-endian)]
        packet_type = ALP_PACKET_TYPE_EXTENSION  # Use extension type for custom data
        payload_length = len(rtcm_frame)
        
        header = struct.pack(
            '!BH',  # Big-endian: byte, unsigned short
            packet_type,
            payload_length
        )
        
        packet = ALPPacket(
            packet_type=packet_type,
            payload_length=payload_length,
            payload=rtcm_frame,
            timestamp=get_timestamp_ms(),
            header=header
        )
        
        self.packet_count += 1
        
        logger.debug(
            f"Encoded RTCM frame: {len(rtcm_frame)} bytes -> "
            f"ALP packet: {len(packet)} bytes "
            f"(header: {bytes_to_hex_string(header)})"
        )
        
        return packet
    
    def encode_bitmap(self, bitmap_data: bytes, tile_id: int = 0) -> ALPPacket:
        """
        Encode bitmap tile into ALP packet.
        
        Args:
            bitmap_data: Raw bitmap data (e.g., 100x100 pixels)
            tile_id: Identifier for this tile
        
        Returns:
            ALPPacket containing the bitmap data
        """
        # Prepend tile metadata: [tile_id (4 bytes)]
        metadata = struct.pack('!I', tile_id)
        payload = metadata + bitmap_data
        
        if len(payload) > ALP_MAX_PAYLOAD_SIZE:
            raise ValueError(
                f"Bitmap payload too large: {len(payload)} bytes "
                f"(max {ALP_MAX_PAYLOAD_SIZE})"
            )
        
        packet_type = ALP_PACKET_TYPE_EXTENSION
        payload_length = len(payload)
        
        header = struct.pack('!BH', packet_type, payload_length)
        
        packet = ALPPacket(
            packet_type=packet_type,
            payload_length=payload_length,
            payload=payload,
            timestamp=get_timestamp_ms(),
            header=header
        )
        
        self.packet_count += 1
        
        logger.debug(
            f"Encoded bitmap tile {tile_id}: {len(bitmap_data)} bytes -> "
            f"ALP packet: {len(packet)} bytes"
        )
        
        return packet
    
    def encode_batch(
        self,
        rtcm_frames: List[bytes],
        bitmap_tiles: Optional[List[Tuple[bytes, int]]] = None
    ) -> List[ALPPacket]:
        """
        Encode multiple RTCM frames and bitmap tiles.
        
        Args:
            rtcm_frames: List of RTCM binary frames
            bitmap_tiles: List of (bitmap_data, tile_id) tuples
        
        Returns:
            List of ALP packets
        """
        packets = []
        
        # Encode RTCM frames
        for rtcm_frame in rtcm_frames:
            try:
                packet = self.encode_rtcm(rtcm_frame)
                packets.append(packet)
            except Exception as e:
                logger.error(f"Failed to encode RTCM frame: {e}")
        
        # Encode bitmap tiles
        if bitmap_tiles:
            for bitmap_data, tile_id in bitmap_tiles:
                try:
                    packet = self.encode_bitmap(bitmap_data, tile_id)
                    packets.append(packet)
                except Exception as e:
                    logger.error(f"Failed to encode bitmap tile {tile_id}: {e}")
        
        logger.info(
            f"Encoded batch: {len(rtcm_frames)} RTCM frames, "
            f"{len(bitmap_tiles) if bitmap_tiles else 0} bitmap tiles -> "
            f"{len(packets)} ALP packets"
        )
        
        return packets
    
    def get_stats(self) -> dict:
        """Get encoder statistics."""
        return {
            'total_packets_encoded': self.packet_count,
        }