#!/usr/bin/env python3
"""
Test RTKLIB integration with Python subprocess.
"""

import subprocess
import os
import sys

def test_rtklib_binary(binary_name):
    """Test if an RTKLIB binary is available and executable."""
    try:
        result = subprocess.run(
            [binary_name],
            capture_output=True,
            text=True,
            timeout=2
        )
        # RTKLIB binaries return help/usage when run without args
        print(f"✓ {binary_name:15s} available")
        return True
    except FileNotFoundError:
        print(f"✗ {binary_name:15s} NOT FOUND")
        return False
    except subprocess.TimeoutExpired:
        print(f"✓ {binary_name:15s} available (timeout as expected)")
        return True
    except Exception as e:
        print(f"✗ {binary_name:15s} ERROR: {e}")
        return False

def main():
    print("=" * 60)
    print("RTKLIB Installation Verification")
    print("=" * 60)
    
    # Check RTKLIB_HOME environment variable
    rtklib_home = os.getenv('RTKLIB_HOME')
    if rtklib_home:
        print(f"\n✓ RTKLIB_HOME: {rtklib_home}")
    else:
        print("\n✗ RTKLIB_HOME not set")
    
    print("\n[RTKLIB Console Applications]")
    all_ok = True
    all_ok &= test_rtklib_binary("rnx2rtkp")  # RTK post-processing
    all_ok &= test_rtklib_binary("rtkrcv")    # RTK receiver
    all_ok &= test_rtklib_binary("str2str")   # Stream converter
    all_ok &= test_rtklib_binary("convbin")   # RINEX converter
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ RTKLIB installation successful!")
        print("\nYou can now use RTKLIB from Python via subprocess.")
    else:
        print("✗ Some RTKLIB binaries are missing.")
        print("Please check the installation steps.")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()