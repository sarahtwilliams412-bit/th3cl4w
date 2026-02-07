#!/usr/bin/env python3
"""
Raw SPDP (Simple Participant Discovery Protocol) Sniffer

Listens for DDS discovery packets on the RTPS multicast group WITHOUT
requiring any DDS library. Pure socket-based — works anywhere Python runs.

SPDP uses UDP multicast on 239.255.0.1, port 7400 + (domain_id * 250 + offset).
For domain 0: port 7400.

Requirements: None (stdlib only)

Usage:
    python3 dds_spdp_sniffer.py [interface_ip] [domain_id]
    python3 dds_spdp_sniffer.py 192.168.123.222
    python3 dds_spdp_sniffer.py 192.168.123.222 0
"""

import socket
import struct
import sys
import time
import signal
from datetime import datetime

# RTPS protocol constants
RTPS_MAGIC = b'RTPS'
SPDP_MULTICAST_GROUP = '239.255.0.1'

# DDS port calculation: PB + DG * domain_id + d0
# Default: PB=7400, DG=250, PG=2, d0=0, d1=10, d2=1, d3=11
def spdp_multicast_port(domain_id):
    """Calculate SPDP multicast port for a given domain ID."""
    PB = 7400  # port base
    DG = 250   # domain gain
    d0 = 0     # multicast meta offset
    return PB + DG * domain_id + d0

def spdp_unicast_port(domain_id, participant_id):
    """Calculate SPDP unicast port."""
    PB = 7400
    DG = 250
    PG = 2
    d1 = 10
    return PB + DG * domain_id + d1 + PG * participant_id


def parse_rtps_header(data):
    """Parse RTPS message header."""
    if len(data) < 20:
        return None
    if data[:4] != RTPS_MAGIC:
        return None

    protocol_version = (data[4], data[5])
    vendor_id = (data[6], data[7])
    guid_prefix = data[8:20]

    # Known vendor IDs
    vendors = {
        (0x01, 0x01): "RTI Connext",
        (0x01, 0x02): "ADLink OpenSplice",
        (0x01, 0x03): "OCI OpenDDS",
        (0x01, 0x0F): "eProsima FastDDS",
        (0x01, 0x10): "Eclipse CycloneDDS",
        (0x01, 0x12): "Eclipse CycloneDDS",  # alternate
    }
    vendor_name = vendors.get(tuple(vendor_id), f"Unknown ({vendor_id[0]:02x}.{vendor_id[1]:02x})")

    return {
        "version": f"{protocol_version[0]}.{protocol_version[1]}",
        "vendor_id": vendor_id,
        "vendor_name": vendor_name,
        "guid_prefix": guid_prefix.hex(),
    }


def parse_submessages(data):
    """Parse RTPS submessages (basic — extract submessage IDs)."""
    offset = 20  # after RTPS header
    submessages = []

    submsg_names = {
        0x01: "PAD",
        0x06: "ACKNACK",
        0x07: "HEARTBEAT",
        0x08: "GAP",
        0x09: "INFO_TS",
        0x0c: "INFO_DST",
        0x0e: "INFO_SRC",
        0x12: "DATA_W",
        0x15: "DATA",
        0x16: "DATA_FRAG",
        0x17: "NACK_FRAG",
        0x18: "HEARTBEAT_FRAG",
    }

    while offset + 4 <= len(data):
        submsg_id = data[offset]
        flags = data[offset + 1]
        little_endian = flags & 0x01
        if little_endian:
            submsg_length = struct.unpack_from('<H', data, offset + 2)[0]
        else:
            submsg_length = struct.unpack_from('>H', data, offset + 2)[0]

        name = submsg_names.get(submsg_id, f"0x{submsg_id:02x}")
        submessages.append({"id": submsg_id, "name": name, "length": submsg_length})

        if submsg_length == 0:
            break
        offset += 4 + submsg_length

    return submessages


class SPDPSniffer:
    """Raw UDP multicast sniffer for DDS SPDP discovery."""

    def __init__(self, bind_ip='0.0.0.0', interface_ip=None, domain_id=0):
        self.bind_ip = bind_ip
        self.interface_ip = interface_ip or '0.0.0.0'
        self.domain_id = domain_id
        self.port = spdp_multicast_port(domain_id)
        self.sock = None
        self.running = False
        self.seen_guids = {}
        self.packet_count = 0

    def start(self):
        """Set up multicast listener socket."""
        print(f"[*] SPDP Sniffer — Domain {self.domain_id}")
        print(f"[*] Multicast group: {SPDP_MULTICAST_GROUP}:{self.port}")
        print(f"[*] Interface IP: {self.interface_ip}")
        print("-" * 60)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Some systems need SO_REUSEPORT
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass

        self.sock.bind(('', self.port))

        # Join multicast group on the specified interface
        mreq = struct.pack(
            '4s4s',
            socket.inet_aton(SPDP_MULTICAST_GROUP),
            socket.inet_aton(self.interface_ip)
        )
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self.sock.settimeout(2.0)

        print(f"[+] Listening for RTPS/SPDP packets...")
        print(f"[*] Press Ctrl+C to stop\n")

    def sniff(self, duration_sec=60):
        """Sniff packets for the given duration."""
        self.running = True
        start = time.time()

        while self.running and (time.time() - start) < duration_sec:
            try:
                data, addr = self.sock.recvfrom(65536)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[!] Recv error: {e}")
                continue

            self.packet_count += 1
            header = parse_rtps_header(data)
            if not header:
                continue

            guid = header['guid_prefix']
            is_new = guid not in self.seen_guids

            if is_new:
                self.seen_guids[guid] = {
                    "first_seen": datetime.now().isoformat(),
                    "source": f"{addr[0]}:{addr[1]}",
                    "vendor": header['vendor_name'],
                    "version": header['version'],
                    "packet_count": 0,
                }
                print(f"[+] NEW DDS PARTICIPANT DETECTED!")
                print(f"    Source:  {addr[0]}:{addr[1]}")
                print(f"    GUID:   {guid}")
                print(f"    Vendor: {header['vendor_name']}")
                print(f"    RTPS:   v{header['version']}")

                submessages = parse_submessages(data)
                if submessages:
                    names = [s['name'] for s in submessages]
                    print(f"    Submsg: {', '.join(names)}")
                print()

            self.seen_guids[guid]['packet_count'] += 1

        self.print_summary()

    def print_summary(self):
        """Print summary of all discovered participants."""
        print(f"\n{'=' * 60}")
        print(f"SPDP SNIFF COMPLETE")
        print(f"{'=' * 60}")
        print(f"Total packets: {self.packet_count}")
        print(f"Unique participants: {len(self.seen_guids)}")

        for guid, info in self.seen_guids.items():
            print(f"\n  GUID: {guid}")
            print(f"    Source:  {info['source']}")
            print(f"    Vendor:  {info['vendor']}")
            print(f"    Packets: {info['packet_count']}")

        if not self.seen_guids:
            print("\n[!] No DDS participants found.")
            print("    Troubleshooting:")
            print("    1. Verify robot is on and connected to 192.168.123.0/24")
            print("    2. Verify your interface has an IP in that subnet")
            print("    3. Check: ip route | grep 239.255")
            print("    4. Try: sudo ip route add 239.255.0.0/16 dev <interface>")
            print("    5. Check firewall: sudo iptables -L -n | grep 7400")

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()


def main():
    interface_ip = sys.argv[1] if len(sys.argv) > 1 else '0.0.0.0'
    domain_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    sniffer = SPDPSniffer(interface_ip=interface_ip, domain_id=domain_id)

    def signal_handler(sig, frame):
        print("\n[*] Stopping...")
        sniffer.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    sniffer.start()
    sniffer.sniff(duration_sec=duration)
    sniffer.stop()


if __name__ == "__main__":
    main()
