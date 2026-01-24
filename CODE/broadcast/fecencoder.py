"""
FEC (Forward Error Correction) Encoder for ATSC 3.0
Implements LDPC and Reed-Solomon encoding with AI-controlled overhead.

Based on ATSC A322 specification.
"""

import numpy as np
import scipy.sparse as sp
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

# Import LDPC from scikit-commpy
from commpy.channelcoding.ldpc import (
    get_ldpc_code_params,
    triang_ldpc_systematic_encode
)

# Import Reed-Solomon
import reedsolo

from broadcast.config import FECCodeRate
from broadcast.utils import get_timestamp_ms

logger = logging.getLogger(__name__)


@dataclass
class FECConfig:
    """FEC configuration parameters."""
    ldpc_rate: FECCodeRate
    reed_solomon_nsym: int  # Number of RS error correction symbols
    overhead_pct: float
    
    def __post_init__(self):
        """Validate configuration."""
        if self.overhead_pct < 0 or self.overhead_pct > 100:
            raise ValueError(f"FEC overhead must be 0-100%, got {self.overhead_pct}")
        if self.reed_solomon_nsym < 0 or self.reed_solomon_nsym > 255:
            raise ValueError(f"RS symbols must be 0-255, got {self.reed_solomon_nsym}")


@dataclass
class FECEncodedData:
    """Container for FEC-encoded data."""
    original_data: bytes
    encoded_data: bytes
    ldpc_applied: bool
    rs_applied: bool
    code_rate: float
    overhead_bytes: int
    timestamp: int


class FECEncoder:
    """
    Forward Error Correction encoder implementing LDPC and Reed-Solomon.
    
    The encoder applies two-stage error correction:
    1. Inner code: LDPC (configurable rate)
    2. Outer code: Reed-Solomon (configurable symbols)
    
    The amount of redundancy is AI-controllable via overhead_pct parameter.
    """
    
    # ATSC 3.0 LDPC code parameters (ATSC A/322:2017 Section 6.1.3)
    # Using short blocks (Ninner = 16200) for PoC
    # Production can use normal blocks (Ninner = 64800) for better performance
    # Format: (n, k) where n=codeword length, k=information bits (Kpayload from Table 6.2)
    LDPC_CODES = {
        FECCodeRate.RATE_2_15: (16200, 2160),    # Code rate 2/15 = 0.133
        FECCodeRate.RATE_3_15: (16200, 3240),    # Code rate 3/15 = 0.200
        FECCodeRate.RATE_4_15: (16200, 4320),    # Code rate 4/15 = 0.267
        FECCodeRate.RATE_5_15: (16200, 5400),    # Code rate 5/15 = 0.333
        FECCodeRate.RATE_6_15: (16200, 6480),    # Code rate 6/15 = 0.400
        FECCodeRate.RATE_7_15: (16200, 7560),    # Code rate 7/15 = 0.500
        FECCodeRate.RATE_8_15: (16200, 8640),    # Code rate 8/15 = 0.533
        FECCodeRate.RATE_9_15: (16200, 9720),    # Code rate 9/15 = 0.600
        FECCodeRate.RATE_10_15: (16200, 10800),  # Code rate 10/15 = 0.667
        FECCodeRate.RATE_11_15: (16200, 11880),  # Code rate 11/15 = 0.733
        FECCodeRate.RATE_12_15: (16200, 12960),  # Code rate 12/15 = 0.800
        FECCodeRate.RATE_13_15: (16200, 14040),  # Code rate 13/15 = 0.867
    }


    
    def __init__(
        self,
        default_ldpc_rate: FECCodeRate = FECCodeRate.RATE_8_15,
        default_rs_symbols: int = 16
    ):
        """
        Initialize FEC encoder.
        
        Args:
            default_ldpc_rate: Default LDPC code rate (from FECCodeRate enum)
            default_rs_symbols: Default number of Reed-Solomon error correction symbols
        """
        self.default_ldpc_rate = default_ldpc_rate
        self.default_rs_symbols = default_rs_symbols
        self.encoded_packet_count = 0
        
        # Cache LDPC matrices for performance
        self._ldpc_matrices = {}
        
        logger.info(
            f"FEC Encoder initialized: "
            f"LDPC rate={default_ldpc_rate.name}, "
            f"RS symbols={default_rs_symbols}"
        )
    
    def encode(
        self,
        data: bytes,
        config: Optional[FECConfig] = None
    ) -> FECEncodedData:
        """
        Apply FEC encoding to data.
        
        Args:
            data: Input data bytes to protect
            config: Optional FEC configuration (uses defaults if None)
        
        Returns:
            FECEncodedData containing original and encoded data with metadata
        """
        if config is None:
            config = FECConfig(
                ldpc_rate=self.default_ldpc_rate,
                reed_solomon_nsym=self.default_rs_symbols,
                overhead_pct=15.0  # Default 15% overhead
            )
        
        original_size = len(data)
        encoded_data = data
        ldpc_applied = False
        rs_applied = False
        
        # Apply Reed-Solomon encoding (outer code)
        if config.reed_solomon_nsym > 0:
            encoded_data = self._apply_reed_solomon(
                encoded_data,
                config.reed_solomon_nsym
            )
            rs_applied = True
            logger.debug(
                f"Reed-Solomon applied: {original_size} -> {len(encoded_data)} bytes "
                f"(+{len(encoded_data) - original_size} bytes)"
            )
        
        # Apply LDPC encoding (inner code)
        if config.ldpc_rate != FECCodeRate.RATE_13_15:  # If not maximum rate
            encoded_data = self._apply_ldpc(
                encoded_data,
                config.ldpc_rate
            )
            ldpc_applied = True
            logger.debug(
                f"LDPC applied: rate={config.ldpc_rate.name}, "
                f"final size={len(encoded_data)} bytes"
            )
        
        # Calculate final code rate
        code_rate = original_size / len(encoded_data) if len(encoded_data) > 0 else 1.0
        overhead_bytes = len(encoded_data) - original_size
        
        result = FECEncodedData(
            original_data=data,
            encoded_data=encoded_data,
            ldpc_applied=ldpc_applied,
            rs_applied=rs_applied,
            code_rate=code_rate,
            overhead_bytes=overhead_bytes,
            timestamp=get_timestamp_ms()
        )
        
        self.encoded_packet_count += 1
        
        logger.debug(
            f"FEC encoding complete: {original_size} -> {len(encoded_data)} bytes, "
            f"code_rate={code_rate:.3f}, overhead={overhead_bytes} bytes"
        )
        
        return result
    
    def _apply_reed_solomon(self, data: bytes, nsym: int) -> bytes:
        """
        Apply Reed-Solomon encoding.
        
        Args:
            data: Input data
            nsym: Number of error correction symbols (2*nsym errors can be corrected)
        
        Returns:
            RS-encoded data (original + parity symbols)
        """
        # Create RS encoder
        rs = reedsolo.RSCodec(nsym)
        
        # Encode the data (adds nsym parity bytes)
        encoded = rs.encode(data)
        
        return encoded
    
    def _apply_ldpc(self, data: bytes, code_rate: FECCodeRate) -> bytes:
        """
        Apply LDPC encoding using systematic encoding.
        
        Args:
            data: Input data (after RS encoding)
            code_rate: LDPC code rate
        
        Returns:
            LDPC-encoded data
        """
        # Get LDPC parameters
        n, k = self.LDPC_CODES[code_rate]  # n=codeword bits, k=info bits
        
        # Convert bytes to bits
        data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        
        # Pad data to multiple of k bits if needed
        total_bits = len(data_bits)
        if total_bits % k != 0:
            padding_bits = k - (total_bits % k)
            data_bits = np.concatenate([
                data_bits,
                np.zeros(padding_bits, dtype=np.uint8)
            ])
        
        # Process in blocks of k bits
        encoded_blocks = []
        num_blocks = len(data_bits) // k
        
        for i in range(num_blocks):
            block_start = i * k
            block_end = block_start + k
            info_block = data_bits[block_start:block_end]
            
            # Get LDPC code parameters
            ldpc_params = self._get_ldpc_params(code_rate)
            
            # Encode using systematic LDPC encoding
            try:
                # triang_ldpc_systematic_encode expects (message_bits, ldpc_code_params, pad)
                encoded_block = triang_ldpc_systematic_encode(
                    info_block.astype(np.int8),
                    ldpc_params,
                    pad=False  # We already padded
                )
                encoded_blocks.append(encoded_block)
            except Exception as e:
                # If encoding fails, use fallback: add parity bits as zeros
                logger.warning(f"LDPC encoding failed, using fallback: {e}")
                parity_bits = np.zeros(n - k, dtype=np.uint8)
                encoded_block = np.concatenate([info_block, parity_bits])
                encoded_blocks.append(encoded_block)
        
        # Concatenate all encoded blocks
        if encoded_blocks:
            all_encoded_bits = np.concatenate(encoded_blocks)
        else:
            all_encoded_bits = data_bits
        
        # Convert bits back to bytes
        encoded_bytes = np.packbits(all_encoded_bits).tobytes()
        
        return encoded_bytes
    
    def _get_ldpc_params(self, code_rate: FECCodeRate) -> dict:
        """
        Get LDPC code parameters dictionary for commpy encoder.
        
        Args:
            code_rate: LDPC code rate
        
        Returns:
            Dictionary with complete LDPC parameters for triang_ldpc_systematic_encode
        """
        if code_rate in self._ldpc_matrices:
            return self._ldpc_matrices[code_rate]
        
        n, k = self.LDPC_CODES[code_rate]
        m = n - k  # Number of parity bits (check nodes)
        
        # Generate complete LDPC code parameters
        ldpc_params = self._generate_ldpc_code_params(n, k, m)
        
        # Cache it
        self._ldpc_matrices[code_rate] = ldpc_params
        
        return ldpc_params
    
    def _generate_ldpc_code_params(self, n: int, k: int, m: int) -> dict:
        """
        Generate complete LDPC code parameters with proper systematic structure.
        
        Creates H = [P | I_m] where P is sparse and I_m is identity.
        This guarantees the code is systematic and encodable.
        
        Args:
            n: Codeword length (variable nodes)
            k: Information bits length
            m: Parity bits length (check nodes)
        
        Returns:
            Complete ldpc_code_params dictionary for commpy
        """
        n_vnodes = n  # Variable nodes = codeword bits
        n_cnodes = m  # Check nodes = parity checks
        
        # Create systematic parity check matrix: H = [P | I_m]
        # Where P is (m x k) sparse matrix and I_m is (m x m) identity
        
        # Design parameters for column weight (variable node degree)
        col_weight = 3  # Each info bit in 3 parity checks (sparse!)
        
        # Build the P matrix (parity part) as adjacency lists
        # Then build the identity part
        
        # For systematic LDPC:
        # - First k variable nodes are information bits (connect to P part)
        # - Last m variable nodes are parity bits (connect to I part)
        
        # Build adjacency lists
        vnode_adj_list = []
        vnode_deg_list = []
        cnode_adj_list = [[] for _ in range(n_cnodes)]
        cnode_deg_list = np.zeros(n_cnodes, dtype=np.int32)
        
        # Process information bit variables (first k vnodes) - connect to P matrix
        for vnode in range(k):
            # Each info bit connects to col_weight check nodes
            # Distribute connections to avoid overloading any single check node
            connections = []
            for w in range(col_weight):
                # Spread connections across check nodes
                cnode = (vnode * col_weight + w) % n_cnodes
                connections.append(cnode)
                cnode_adj_list[cnode].append(vnode)
                cnode_deg_list[cnode] += 1
            
            vnode_adj_list.append(connections)
            vnode_deg_list.append(col_weight)
        
        # Process parity bit variables (last m vnodes) - connect to I matrix
        # Each parity bit connects to exactly ONE check node (identity structure)
        for i in range(m):
            vnode = k + i  # Parity variable node index
            cnode = i      # Corresponding check node
            
            vnode_adj_list.append([cnode])
            vnode_deg_list.append(1)  # Degree 1 for parity bits
            cnode_adj_list[cnode].append(vnode)
            cnode_deg_list[cnode] += 1
        
        # Convert to numpy arrays with padding
        vnode_deg_list = np.array(vnode_deg_list, dtype=np.int32)
        max_vnode_deg = int(np.max(vnode_deg_list))
        max_cnode_deg = int(np.max(cnode_deg_list))
        
        # Pad adjacency lists to max degree
        vnode_adj_padded = np.full((n_vnodes, max_vnode_deg), -1, dtype=np.int32)
        for vnode in range(n_vnodes):
            for j, cnode in enumerate(vnode_adj_list[vnode]):
                vnode_adj_padded[vnode, j] = cnode
        
        cnode_adj_padded = np.full((n_cnodes, max_cnode_deg), -1, dtype=np.int32)
        for cnode in range(n_cnodes):
            for j, vnode in enumerate(cnode_adj_list[cnode]):
                cnode_adj_padded[cnode, j] = vnode
        
        # Build index mapping arrays
        vnode_cnode_map = np.zeros(n_vnodes * max_vnode_deg, dtype=np.int32)
        cnode_vnode_map = np.zeros(n_cnodes * max_cnode_deg, dtype=np.int32)
        
        # Build vnode -> cnode position mapping
        for vnode in range(n_vnodes):
            for j in range(vnode_deg_list[vnode]):
                cnode = vnode_adj_padded[vnode, j]
                if cnode >= 0:
                    # Find position of vnode in cnode's adjacency list
                    for pos in range(cnode_deg_list[cnode]):
                        if cnode_adj_padded[cnode, pos] == vnode:
                            vnode_cnode_map[vnode * max_vnode_deg + j] = pos
                            break
        
        # Build cnode -> vnode position mapping
        for cnode in range(n_cnodes):
            for j in range(cnode_deg_list[cnode]):
                vnode = cnode_adj_padded[cnode, j]
                if vnode >= 0:
                    # Find position of cnode in vnode's adjacency list
                    for pos in range(vnode_deg_list[vnode]):
                        if vnode_adj_padded[vnode, pos] == cnode:
                            cnode_vnode_map[cnode * max_cnode_deg + j] = pos
                            break
        
        # Flatten adjacency lists
        vnode_adj_flat = vnode_adj_padded.flatten()
        cnode_adj_flat = cnode_adj_padded.flatten()
        
        # Build sparse parity check matrix H = [P | I]
        H = self._build_systematic_parity_matrix(
            n_vnodes, n_cnodes, k, vnode_adj_padded, vnode_deg_list
        )
        
        # Create complete ldpc_code_params dictionary
        ldpc_params = {
            'n_vnodes': n_vnodes,
            'n_cnodes': n_cnodes,
            'max_vnode_deg': max_vnode_deg,
            'max_cnode_deg': max_cnode_deg,
            'vnode_adj_list': vnode_adj_flat,
            'cnode_adj_list': cnode_adj_flat,
            'vnode_cnode_map': vnode_cnode_map,
            'cnode_vnode_map': cnode_vnode_map,
            'vnode_deg_list': vnode_deg_list,
            'cnode_deg_list': cnode_deg_list,
            'parity_check_matrix': H,
        }
        
        logger.debug(
            f"Generated systematic LDPC: n={n_vnodes}, k={k}, m={n_cnodes}, "
            f"max_vdeg={max_vnode_deg}, max_cdeg={max_cnode_deg}"
        )
        
        return ldpc_params
    
    def _build_systematic_parity_matrix(
        self,
        n_vnodes: int,
        n_cnodes: int,
        k: int,
        vnode_adj_list: np.ndarray,
        vnode_deg_list: np.ndarray
    ) -> 'sp.sparse.csc_matrix':
        """
        Build systematic parity check matrix H = [P | I] from adjacency list.
        
        Args:
            n_vnodes: Number of variable nodes (n)
            n_cnodes: Number of check nodes (m)
            k: Number of information bits
            vnode_adj_list: Variable node adjacency list
            vnode_deg_list: Degree of each variable node
        
        Returns:
            Sparse parity check matrix H in CSC format (m × n)
        """
        import scipy.sparse as sp
        
        # Build using COO format, then convert to CSC
        row_indices = []
        col_indices = []
        
        for vnode in range(n_vnodes):
            for j in range(vnode_deg_list[vnode]):
                cnode = vnode_adj_list[vnode, j]
                if cnode >= 0:
                    row_indices.append(cnode)
                    col_indices.append(vnode)
        
        # All connections have value 1 in GF(2)
        data = np.ones(len(row_indices), dtype=np.int8)
        
        # Create sparse matrix
        H = sp.coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_cnodes, n_vnodes),
            dtype=np.int8
        )
        
        # Convert to CSC format (required by commpy)
        H_csc = H.tocsc()
        
        logger.debug(f"Built H matrix: shape={H_csc.shape}, nnz={H_csc.nnz}")
        
        return H_csc
    
    def encode_with_overhead_pct(
        self,
        data: bytes,
        overhead_pct: float
    ) -> FECEncodedData:
        """
        Encode data with specified overhead percentage (AI-controlled).
        
        This method allows the AI to specify desired overhead as a percentage,
        and the encoder selects appropriate LDPC rate and RS symbols.
        
        Args:
            data: Input data
            overhead_pct: Desired overhead percentage (0-100)
        
        Returns:
            FECEncodedData with approximately the requested overhead
        """
        # Map overhead percentage to FEC parameters
        ldpc_rate, rs_symbols = self._map_overhead_to_params(overhead_pct)
        
        config = FECConfig(
            ldpc_rate=ldpc_rate,
            reed_solomon_nsym=rs_symbols,
            overhead_pct=overhead_pct
        )
        
        return self.encode(data, config)
    
    def _map_overhead_to_params(
        self,
        overhead_pct: float
    ) -> Tuple[FECCodeRate, int]:
        """
        Map overhead percentage to LDPC rate and RS symbols.
        
        Args:
            overhead_pct: Desired overhead (0-100)
        
        Returns:
            (ldpc_rate, rs_symbols) tuple
        """
        if overhead_pct < 10:
            # Minimal protection
            return FECCodeRate.RATE_12_15, 8
        elif overhead_pct < 20:
            # Light protection
            return FECCodeRate.RATE_10_15, 12
        elif overhead_pct < 30:
            # Medium protection (default)
            return FECCodeRate.RATE_8_15, 16
        else:
            # Heavy protection (for challenging conditions)
            return FECCodeRate.RATE_6_15, 20
    
    def get_stats(self) -> dict:
        """Get encoder statistics."""
        return {
            'total_packets_encoded': self.encoded_packet_count,
            'default_ldpc_rate': self.default_ldpc_rate.name,
            'default_rs_symbols': self.default_rs_symbols,
            'cached_ldpc_matrices': len(self._ldpc_matrices),
        }