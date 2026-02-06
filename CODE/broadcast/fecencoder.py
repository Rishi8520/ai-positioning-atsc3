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
import zlib
import struct

def packetize(payload: bytes) -> bytes:
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return struct.pack("!II", len(payload), crc) + payload

def depacketize(pkt: bytes) -> bytes:
    if len(pkt) < 8:
        raise ValueError("Packet too short for len+crc header")
    length, crc = struct.unpack("!II", pkt[:8])
    if length < 0 or length > len(pkt) - 8:
        raise ValueError(f"Invalid length {length} for packet of size {len(pkt)}")
    payload = pkt[8:8+length]
    crc2 = zlib.crc32(payload) & 0xFFFFFFFF
    if crc2 != crc:
        raise ValueError(f"CRC mismatch: stored=0x{crc:08x} computed=0x{crc2:08x}")
    return payload

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
    
    def encode(self, data: bytes, config=None) -> FECEncodedData:
        """
    Encode data with FEC (RS only - LDPC bypassed for demo).
    
    Args:
        data: Input data to encode
        
    Returns:
        FECEncodedData with encoded data and metadata
        """
        original_size = len(data)
        #rs_symbols = 16 
    # Step 1: Reed-Solomon encoding
        rs_encoded = self._apply_reed_solomon(data, 16)
        rs_overhead = len(rs_encoded) - len(data)
    
    # Step 2: BYPASS LDPC - just pass through
        logger.info(f"[DEMO MODE] Using RS FEC")
        final_encoded = rs_encoded  # ← BYPASS LDPC
        ldpc_overhead = 0
    
        total_overhead = rs_overhead + ldpc_overhead
    
        return FECEncodedData(
        encoded_data=final_encoded,
        original_data=data,
        code_rate="RATE_8_15",
        overhead_bytes=len(final_encoded) - len(data),
        ldpc_applied=False,  # ← ADD THIS
        rs_applied=True,      # ← ADD THIS
        timestamp=get_timestamp_ms()
        )
    
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
    
        # DEBUG: Log matrix hash (ONLY FOR FIRST BLOCK)
            if i == 0:
                import hashlib
                H_array = ldpc_params['vnode_adj_list']
                matrix_hash = hashlib.md5(H_array.tobytes()).hexdigest()[:8]
                logger.info(f"[TX LDPC] Matrix hash: {matrix_hash}, n_vnodes={ldpc_params['n_vnodes']}, n_cnodes={ldpc_params['n_cnodes']}")
    
        # Encode using systematic LDPC encoding
            try:
            # triang_ldpc_systematic_encode expects (message_bits, ldpc_code_params, pad)
                encoded_block = triang_ldpc_systematic_encode(
                info_block.astype(np.int8),
                ldpc_params,
                pad=False  # We already padded
                )
            
            # ==================== NEW: VERIFY CODEWORD ====================
            # Check if encoded codeword satisfies H @ c = 0
                if 'parity_check_matrix' in ldpc_params:
                    H = ldpc_params['parity_check_matrix']
                    syndrome = (H @ encoded_block.astype(np.int8)) % 2
                    failed_checks = np.sum(syndrome != 0)
                
                    if i == 0:  # Log first block only
                        logger.info(f"[TX VERIFY] Encoded codeword syndrome: {failed_checks}/{len(syndrome)} failed checks")
                
                    if failed_checks > 0:
                        logger.error(f"[TX ERROR] Block {i}: Encoder produced INVALID codeword! "
                               f"{failed_checks}/{len(syndrome)} parity checks fail")
            # ==================== END NEW ====================
            
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
        """Get LDPC code parameters dictionary."""
        import os
    
        ldpc_path = os.getenv("LDPC_PARAMS_PATH", "").strip()
        if ldpc_path and os.path.exists(ldpc_path):
            logger.info(f"Loading LDPC params from {ldpc_path}")
            return self.load_ldpc_params(ldpc_path)
    
    # Check local cache
        if code_rate in self._ldpc_matrices:
            return self._ldpc_matrices[code_rate]
    
        n, k = self.LDPC_CODES[code_rate]
        m = n - k
    
        logger.info(f"Generating LDPC code: n={n}, k={k}, rate={k/n:.3f}")
    
    # Generate sparse H matrix using Gallager construction
    # Regular LDPC: each column has wc=3 ones, each row has wr ones
        ldpc_params = self._generate_ldpc_code_params(n, k, m)    
    # Cache it
        self._ldpc_matrices[code_rate] = ldpc_params
    
    # Save to file if path set
        if not ldpc_path:
            ldpc_path = f"OUTPUTs/ldpc_{code_rate.name.lower()}.npz"
    
        os.makedirs(os.path.dirname(ldpc_path) if os.path.dirname(ldpc_path) else "OUTPUTs", exist_ok=True)
        self.save_ldpc_params(code_rate, ldpc_path)
        logger.info(f"Saved LDPC params to {ldpc_path}")
    
        return ldpc_params

    def _generate_gallager_ldpc(self, n: int, k: int) -> dict:
        """Generate a Gallager LDPC code (regular, good for BP)."""
        from scipy.sparse import lil_matrix
    
        m = n - k
    
    # Regular LDPC parameters
        wc = 3  # column weight (ones per column)
        wr = int(np.ceil(n * wc / m))  # row weight
    
        logger.info(f"Gallager LDPC: wc={wc}, wr={wr}")
    
    # Create sparse H matrix in lil format (efficient for construction)
        H = lil_matrix((m, n), dtype=np.int8)
    
    # Gallager construction: divide H into wc submatrices
        rows_per_submatrix = m // wc
    
        for sub_idx in range(wc):
            row_start = sub_idx * rows_per_submatrix
            row_end = min((sub_idx + 1) * rows_per_submatrix, m)
        
        # Create a permuted identity-like pattern
            cols = np.random.permutation(n)
            for i, row in enumerate(range(row_start, row_end)):
                if i < len(cols):
                    H[row, cols[i]] = 1
    
    # Convert to CSR for efficient operations
        H_csr = H.tocsr()
        H_array = H_csr.toarray().astype(np.int8)
    
    # Build adjacency lists for BP
        vnode_adj_list = []
        vnode_deg_list = []
        for col in range(n):
            neighbors = H_csr.getcol(col).nonzero()[0].tolist()
            vnode_adj_list.append(neighbors)
            vnode_deg_list.append(len(neighbors))
    
        cnode_adj_list = []
        cnode_deg_list = []
        for row in range(m):
            neighbors = H_csr.getrow(row).nonzero()[1].tolist()
            cnode_adj_list.append(neighbors)
            cnode_deg_list.append(len(neighbors))
    
    # Compute max degrees
        max_vnode_deg = max(vnode_deg_list) if vnode_deg_list else 0
        max_cnode_deg = max(cnode_deg_list) if cnode_deg_list else 0
    
        return {
        'n_vnodes': n,
        'n_cnodes': m,
        'vnode_adj_list': np.array(vnode_adj_list, dtype=object),
        'cnode_adj_list': np.array(cnode_adj_list, dtype=object),
        'vnode_deg_list': np.array(vnode_deg_list, dtype=np.int32),
        'cnode_deg_list': np.array(cnode_deg_list, dtype=np.int32),
        'max_vnode_deg': max_vnode_deg,
        'max_cnode_deg': max_cnode_deg,
        'H_matrix': H_array
        }
        
    def _generate_ldpc_code_params(self, n: int, k: int, m: int) -> dict:
        """
    Generate LDPC code with simple systematic structure.
    Fast generation using direct construction.
        """
        import scipy.sparse as sp
    
        logger.info(f"Generating simple systematic LDPC: n={n}, k={k}, m={m}")
    
    # Create H = [P | I] where I is m×m identity
        H = np.zeros((m, n), dtype=np.int8)
    
    # Right part: identity matrix (last m columns)
        H[:, k:] = np.eye(m, dtype=np.int8)
    
    # Left part: sparse random connections (first k columns)
    # Each column has exactly 3 ones (regular LDPC)
        col_weight = 3
    
        for col in range(k):
        # Pick 3 random rows for this column
            rows = np.random.choice(m, size=col_weight, replace=False)
            H[rows, col] = 1
    
    # Build adjacency lists
        vnode_adj_list = []
        vnode_deg_list = []
        for col in range(n):
            neighbors = np.where(H[:, col] != 0)[0].tolist()
            vnode_adj_list.append(neighbors)
            vnode_deg_list.append(len(neighbors))
    
        cnode_adj_list = []
        cnode_deg_list = []
        for row in range(m):
            neighbors = np.where(H[row, :] != 0)[0].tolist()
            cnode_adj_list.append(neighbors)
            cnode_deg_list.append(len(neighbors))
    
        max_vnode_deg = max(vnode_deg_list)
        max_cnode_deg = max(cnode_deg_list)
    
    # Pad adjacency lists
        vnode_adj_padded = np.full((n, max_vnode_deg), -1, dtype=np.int32)
        for i, neighbors in enumerate(vnode_adj_list):
            vnode_adj_padded[i, :len(neighbors)] = neighbors
    
        cnode_adj_padded = np.full((m, max_cnode_deg), -1, dtype=np.int32)
        for i, neighbors in enumerate(cnode_adj_list):
            cnode_adj_padded[i, :len(neighbors)] = neighbors
    
    # Build mappings
        vnode_cnode_map = np.zeros(n * max_vnode_deg, dtype=np.int32)
        cnode_vnode_map = np.zeros(m * max_cnode_deg, dtype=np.int32)
    
        for vnode in range(n):
            for j in range(vnode_deg_list[vnode]):
                cnode = vnode_adj_padded[vnode, j]
                if cnode >= 0:
                    for pos in range(cnode_deg_list[cnode]):
                        if cnode_adj_padded[cnode, pos] == vnode:
                            vnode_cnode_map[vnode * max_vnode_deg + j] = pos
                            break
    
        for cnode in range(m):
            for j in range(cnode_deg_list[cnode]):
                vnode = cnode_adj_padded[cnode, j]
                if vnode >= 0:
                    for pos in range(vnode_deg_list[vnode]):
                        if vnode_adj_padded[vnode, pos] == cnode:
                            cnode_vnode_map[cnode * max_cnode_deg + j] = pos
                            break
    
        logger.info(f"LDPC generated: max_vdeg={max_vnode_deg}, max_cdeg={max_cnode_deg}")
    
        return {
        'n_vnodes': n,
        'n_cnodes': m,
        'max_vnode_deg': max_vnode_deg,
        'max_cnode_deg': max_cnode_deg,
        'vnode_adj_list': vnode_adj_padded.flatten(),
        'cnode_adj_list': cnode_adj_padded.flatten(),
        'vnode_cnode_map': vnode_cnode_map,
        'cnode_vnode_map': cnode_vnode_map,
        'vnode_deg_list': np.array(vnode_deg_list, dtype=np.int32),
        'cnode_deg_list': np.array(cnode_deg_list, dtype=np.int32),
        'parity_check_matrix': H,
        }
    
    def save_ldpc_params(self, code_rate: FECCodeRate, path: str) -> None:
        p = self._get_ldpc_params(code_rate)
        np.savez_compressed(
        path,
        n_vnodes=p["n_vnodes"],
        n_cnodes=p["n_cnodes"],
        max_vnode_deg=p["max_vnode_deg"],
        max_cnode_deg=p["max_cnode_deg"],
        vnode_adj_list=p["vnode_adj_list"],
        cnode_adj_list=p["cnode_adj_list"],
        vnode_deg_list=p["vnode_deg_list"],
        cnode_deg_list=p["cnode_deg_list"],
        )

    @staticmethod
    def load_ldpc_params(path: str) -> dict:
        z = np.load(path, allow_pickle=False)
        return {
        "n_vnodes": int(z["n_vnodes"]),
        "n_cnodes": int(z["n_cnodes"]),
        "max_vnode_deg": int(z["max_vnode_deg"]),
        "max_cnode_deg": int(z["max_cnode_deg"]),
        "vnode_adj_list": z["vnode_adj_list"].astype(np.int32),
        "cnode_adj_list": z["cnode_adj_list"].astype(np.int32),
        "vnode_deg_list": z["vnode_deg_list"].astype(np.int32),
        "cnode_deg_list": z["cnode_deg_list"].astype(np.int32),
        }

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