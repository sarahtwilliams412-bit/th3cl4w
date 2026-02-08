#!/usr/bin/env python3
"""
DDS Discovery Tool for Unitree Robots

Uses CycloneDDS to create a DomainParticipant on domain 0 and discover
any DDS participants/topics on the 192.168.123.0/24 network.

Requirements:
    pip install cyclonedds==0.10.2

Usage:
    python3 dds_discover.py [interface_name]
    python3 dds_discover.py eth0
    python3 dds_discover.py  # auto-detect interface
"""

import sys
import time
import signal
import json
from datetime import datetime

try:
    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.builtin import DcpsParticipant, DcpsTopic, DcpsEndpoint
    from cyclonedds.builtin import BuiltinDataReader
    from cyclonedds.sub import DataReader
    from cyclonedds.core import Listener, DDSException
    from cyclonedds.util import duration
    HAS_CYCLONEDDS = True
except ImportError:
    HAS_CYCLONEDDS = False

UNITREE_DOMAIN_ID = 0

# CycloneDDS XML config templates (from unitree_sdk2_python source)
CONFIG_WITH_INTERFACE = '''<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
    <Domain Id="any">
        <General>
            <Interfaces>
                <NetworkInterface name="{interface}" priority="default" multicast="default"/>
            </Interfaces>
        </General>
        <Tracing>
            <Verbosity>warning</Verbosity>
            <OutputFile>/tmp/dds_discover.log</OutputFile>
        </Tracing>
    </Domain>
</CycloneDDS>'''

CONFIG_AUTO = '''<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
    <Domain Id="any">
        <General>
            <Interfaces>
                <NetworkInterface autodetermine="true" priority="default" multicast="default"/>
            </Interfaces>
        </General>
    </Domain>
</CycloneDDS>'''


class DDSDiscoverer:
    """Discovers DDS participants and topics on a network."""

    def __init__(self, interface=None, domain_id=UNITREE_DOMAIN_ID):
        self.interface = interface
        self.domain_id = domain_id
        self.domain = None
        self.participant = None
        self.discovered_participants = {}
        self.discovered_topics = {}
        self.discovered_endpoints = {}
        self.running = False

    def start(self):
        """Initialize DDS domain and participant."""
        config = (
            CONFIG_WITH_INTERFACE.format(interface=self.interface)
            if self.interface
            else CONFIG_AUTO
        )

        print(f"[*] Creating DDS Domain (id={self.domain_id})")
        if self.interface:
            print(f"[*] Binding to interface: {self.interface}")
        else:
            print("[*] Auto-detecting network interface")

        try:
            self.domain = Domain(self.domain_id, config)
            self.participant = DomainParticipant(self.domain_id)
        except DDSException as e:
            print(f"[!] Failed to create DDS domain/participant: {e}")
            print("[!] Check: is the network interface up? Is multicast enabled?")
            return False
        except Exception as e:
            print(f"[!] Unexpected error: {e}")
            return False

        print(f"[+] DDS Participant created on domain {self.domain_id}")
        return True

    def discover(self, timeout_sec=30):
        """Run discovery for the specified duration."""
        if not self.participant:
            print("[!] Not initialized. Call start() first.")
            return

        self.running = True
        print(f"\n[*] Discovering DDS entities for {timeout_sec}s...")
        print(f"[*] Listening for SPDP/SEDP announcements on domain {self.domain_id}")
        print("-" * 60)

        # Built-in topic readers for discovery
        participant_reader = BuiltinDataReader(self.participant, DcpsParticipant)
        topic_reader = BuiltinDataReader(self.participant, DcpsTopic)

        start_time = time.time()
        poll_interval = 0.5

        while self.running and (time.time() - start_time) < timeout_sec:
            # Read discovered participants
            try:
                samples = participant_reader.take(10)
                for sample in samples:
                    key = str(sample.key)
                    if key not in self.discovered_participants:
                        self.discovered_participants[key] = {
                            "key": key,
                            "qos": str(sample.qos) if hasattr(sample, 'qos') else "N/A",
                            "discovered_at": datetime.now().isoformat(),
                        }
                        print(f"\n[+] PARTICIPANT DISCOVERED")
                        print(f"    Key: {key}")
                        if hasattr(sample, 'qos'):
                            print(f"    QoS: {sample.qos}")
            except Exception:
                pass

            # Read discovered topics
            try:
                samples = topic_reader.take(10)
                for sample in samples:
                    topic_name = sample.topic_name if hasattr(sample, 'topic_name') else str(sample)
                    type_name = sample.type_name if hasattr(sample, 'type_name') else "unknown"
                    tkey = f"{topic_name}:{type_name}"
                    if tkey not in self.discovered_topics:
                        self.discovered_topics[tkey] = {
                            "topic_name": topic_name,
                            "type_name": type_name,
                            "discovered_at": datetime.now().isoformat(),
                        }
                        print(f"\n[+] TOPIC DISCOVERED")
                        print(f"    Name: {topic_name}")
                        print(f"    Type: {type_name}")
            except Exception:
                pass

            # Progress indicator
            elapsed = int(time.time() - start_time)
            if elapsed % 5 == 0 and elapsed > 0:
                sys.stdout.write(f"\r[*] {elapsed}s elapsed | "
                                 f"{len(self.discovered_participants)} participants | "
                                 f"{len(self.discovered_topics)} topics")
                sys.stdout.flush()

            time.sleep(poll_interval)

        print(f"\n\n{'=' * 60}")
        print(f"DISCOVERY COMPLETE")
        print(f"{'=' * 60}")

    def print_summary(self):
        """Print a summary of all discovered entities."""
        print(f"\n--- Participants ({len(self.discovered_participants)}) ---")
        for key, info in self.discovered_participants.items():
            print(f"  {key} (found at {info['discovered_at']})")

        print(f"\n--- Topics ({len(self.discovered_topics)}) ---")
        for tkey, info in self.discovered_topics.items():
            print(f"  {info['topic_name']}  [{info['type_name']}]")

        if not self.discovered_participants and not self.discovered_topics:
            print("\n[!] No DDS entities found. Possible causes:")
            print("    - Robot is not powered on or not connected")
            print("    - Wrong network interface specified")
            print("    - Firewall blocking UDP 7400-7500")
            print("    - Multicast not enabled on interface")
            print("    - Robot uses a different domain ID")
            print(f"\n    Expected robot IP: 192.168.123.10 (domain {self.domain_id})")

    def save_results(self, filepath="dds_discovery_results.json"):
        """Save discovery results to JSON."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "domain_id": self.domain_id,
            "interface": self.interface or "auto",
            "participants": self.discovered_participants,
            "topics": self.discovered_topics,
        }
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[+] Results saved to {filepath}")

    def stop(self):
        """Clean shutdown."""
        self.running = False
        if self.participant:
            del self.participant
        if self.domain:
            del self.domain


def main():
    if not HAS_CYCLONEDDS:
        print("[!] cyclonedds not installed.")
        print("[*] Install with: pip install cyclonedds==0.10.2")
        print("[*] If that fails, build from source:")
        print("    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x")
        print("    cd cyclonedds && mkdir build install && cd build")
        print("    cmake .. -DCMAKE_INSTALL_PREFIX=../install && cmake --build . --target install")
        print("    export CYCLONEDDS_HOME=$(pwd)/../install")
        print("    pip install cyclonedds==0.10.2")
        sys.exit(1)

    interface = sys.argv[1] if len(sys.argv) > 1 else None
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    discoverer = DDSDiscoverer(interface=interface)

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n[*] Stopping discovery...")
        discoverer.stop()
        discoverer.print_summary()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    if not discoverer.start():
        sys.exit(1)

    discoverer.discover(timeout_sec=timeout)
    discoverer.print_summary()
    discoverer.save_results()
    discoverer.stop()


if __name__ == "__main__":
    main()
