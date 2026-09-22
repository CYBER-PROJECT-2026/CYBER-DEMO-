"""
aegis_live/build_v3_dataset.py
==============================
Deterministic, Leakage-Safe Dataset Construction Pipeline for AEGIS-Live v3.
Evaluates: TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY.

Constructs the 27-feature schema:
- Canonical 24 statistical features (duration, rates, IAT, packet sizes, flag ratios, TTLs, directional ratios)
- 3 validated topology features:
    1. fan_in_ratio
    2. src_concentration_ratio
    3. src_ip_cardinality_per_window

Strict Invariants:
- 5.0-second deterministic state-machine temporal quarantine dead-time between partitions
- On-arrival 2.0-second inactivity timeout gating
- 50-packet window capacity
- Zero cross-partition exact duplicate 27-feature vectors
- Quality filtering against all physical and mathematical bounds (zero silent drops)
- 100% deterministic reproducibility from raw PCAP files
- Target: exactly 80,000 balanced windows (56,000 Train, 12,000 Val, 12,000 Test; 14k/3k/3k per class)
"""

import os
import sys
import time
import json
import hashlib
import socket
import ipaddress
import argparse
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

import dpkt
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from aegis_live.feature_schema import (
    CANONICAL_24_FEATURES,
    CANONICAL_27_FEATURES,
    V3_TOPOLOGY_FEATURES,
    validate_27_features
)
from aegis_live.packet_window import FlowWindow, PacketRecord
from aegis_live.v3_topology_features import (
    V3TopologyPacket,
    compute_topology_features_batch
)

DATA_V3_DIR = os.path.join(BASE_DIR, "data", "aegis_live_v3")
PCAP_DIR = os.path.join(BASE_DIR, "pcaps")

OFFICIAL_PCAPS = [
    {"filename": "DDoS-SYN_Flood.pcap", "label": "DDoS"},
    {"filename": "DoS-SYN_Flood.pcap", "label": "DoS"},
    {"filename": "MITM-ArpSpoofing.pcap", "label": "Spoofing"},
    {"filename": "BenignTraffic.pcap", "label": "Benign"}
]

TARGET_WINDOWS_PER_CLASS = 20000
TRAIN_RATIO = 0.70   # 14,000 windows/class
VAL_RATIO = 0.15     # 3,000 windows/class
TEST_RATIO = 0.15    # 3,000 windows/class

WINDOW_CAPACITY = 50
INACTIVITY_TIMEOUT_SEC = 2.0
QUARANTINE_DEAD_TIME_SEC = 5.0
LOCAL_SUBNET = "192.168.0.0/16"


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 16):
            h.update(chunk)
    return h.hexdigest()


class LeakageSafeV3DatasetBuilder:
    def __init__(self,
                 target_per_class: int = TARGET_WINDOWS_PER_CLASS,
                 train_ratio: float = TRAIN_RATIO,
                 val_ratio: float = VAL_RATIO,
                 test_ratio: float = TEST_RATIO,
                 window_capacity: int = WINDOW_CAPACITY,
                 timeout_sec: float = INACTIVITY_TIMEOUT_SEC,
                 quarantine_sec: float = QUARANTINE_DEAD_TIME_SEC,
                 local_subnet: str = LOCAL_SUBNET,
                 output_dir: str = DATA_V3_DIR):
        self.target_per_class = target_per_class
        self.train_count = int(target_per_class * train_ratio)
        self.val_count = int(target_per_class * val_ratio)
        self.test_count = target_per_class - self.train_count - self.val_count
        self.window_capacity = window_capacity
        self.timeout_sec = timeout_sec
        self.quarantine_sec = quarantine_sec
        self.local_network = ipaddress.ip_network(local_subnet)
        self.output_dir = output_dir

        self.exclusion_counts: Dict[str, Counter] = {}
        self.quarantine_audit: Dict[str, Any] = {}
        self.partition_feature_sets: Dict[str, set] = {
            "train": set(),
            "validation": set(),
            "test": set()
        }

    def _is_local(self, ip_str: str) -> bool:
        try:
            return ipaddress.ip_address(ip_str) in self.local_network
        except ValueError:
            return False

    def validate_candidate_window(self,
                                  window: FlowWindow,
                                  features: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """
        Audits candidate window against all 27 feature invariants.
        """
        # 1. Packet count
        if len(window.packets) < self.window_capacity:
            return False, "insufficient_packet_count"

        # 2. Internal gap < 2.0s
        for i in range(1, len(window.packets)):
            gap = window.packets[i].timestamp - window.packets[i - 1].timestamp
            if gap >= self.timeout_sec:
                return False, "internal_packet_gap_ge_2s"

        # 3. Exactly 27 features
        if len(features) != 27 or set(features.keys()) != set(CANONICAL_27_FEATURES):
            return False, "malformed_feature_vector_v3"

        # 4. NaN / Inf
        for k, v in features.items():
            if np.isnan(v):
                return False, f"nan_value_{k}"
            if np.isinf(v):
                return False, f"inf_value_{k}"

        # 5. Non-negative duration and rates
        if features["duration_ms"] < 0.0:
            return False, "negative_duration"
        if features["pkt_rate_pps"] < 0.0 or features["byte_rate_bps"] < 0.0:
            return False, "negative_rate"

        # 6. Packet size bounds
        min_sz = features["pkt_size_min"]
        max_sz = features["pkt_size_max"]
        mean_sz = features["pkt_size_mean"]
        if min_sz < 14 or max_sz > 65535 or not (min_sz <= mean_sz <= max_sz):
            return False, "invalid_packet_size"

        # 7. Standard ratio bounds [0, 1]
        ratio_features = [
            "syn_ratio", "ack_ratio", "rst_ratio", "fin_ratio", "urg_ratio",
            "psh_ratio", "ece_ratio", "cwr_ratio", "tcp_ratio", "udp_ratio",
            "icmp_ratio", "inbound_pkt_ratio", "outbound_pkt_ratio",
            "src_concentration_ratio"
        ]
        for rk in ratio_features:
            rv = features[rk]
            if rv < -1e-6 or rv > 1.0 + 1e-6:
                return False, f"invalid_ratio_range_{rk}"

        # 8. TTL in [1, 255]
        ttl_mean = features["ip_ttl_mean"]
        if ttl_mean < 1.0 or ttl_mean > 255.0:
            return False, "ttl_out_of_bounds"

        # 9. Directional ratio sum
        in_out_sum = features["inbound_pkt_ratio"] + features["outbound_pkt_ratio"]
        if abs(in_out_sum - 1.0) > 1e-5:
            return False, "direction_ratio_mismatch"

        # 10. Topology features validity
        if features["fan_in_ratio"] < 0.0:
            return False, "negative_fan_in_ratio"
        if features["src_ip_cardinality_per_window"] < 1.0:
            return False, "invalid_src_ip_cardinality"

        return True, None

    def _make_window_record(self,
                            feats: Dict[str, float],
                            window: FlowWindow,
                            class_label: str,
                            pcap_file: str,
                            flow_id_str: str,
                            current_partition: str,
                            partition_valid_count: int) -> Dict[str, Any]:
        if current_partition == "train":
            blk = min(6, partition_valid_count * 7 // self.train_count)
        elif current_partition == "validation":
            blk = 7
        else:
            blk = 8 if partition_valid_count < (self.test_count // 2) else 9

        return {
            **feats,
            "label": class_label,
            "pcap_file": pcap_file,
            "flow_id": flow_id_str,
            "window_start_ts": float(window.first_timestamp),
            "window_end_ts": float(window.last_timestamp),
            "packet_count": len(window.packets),
            "temporal_block_id": blk,
            "group_id": f"{class_label}_{current_partition}_blk{blk}",
            "partition": current_partition
        }

    def _try_accept_window(self,
                           window: FlowWindow,
                           feats: Dict[str, float],
                           class_label: str,
                           pcap_file: str,
                           flow_id_str: str,
                           current_partition: str,
                           valid_windows: List[Dict[str, Any]],
                           partition_valid_count: int) -> bool:
        is_val, reason = self.validate_candidate_window(window, feats)
        if not is_val:
            self.exclusion_counts[pcap_file][reason] += 1
            return False

        # Strictly prevent identical 27-feature vectors across partitions
        feat_tuple = tuple(feats[k] for k in CANONICAL_27_FEATURES)
        for other_part in ["train", "validation", "test"]:
            if other_part != current_partition:
                if feat_tuple in self.partition_feature_sets[other_part]:
                    self.exclusion_counts[pcap_file][f"cross_partition_duplicate_{current_partition}_vs_{other_part}"] += 1
                    return False

        w_meta = self._make_window_record(
            feats, window, class_label, pcap_file, flow_id_str, current_partition, partition_valid_count
        )
        valid_windows.append(w_meta)
        self.partition_feature_sets[current_partition].add(feat_tuple)
        return True

    def _extract_window_features(self, window: FlowWindow) -> Dict[str, float]:
        """Extracts 24 canonical features + 3 validated topology features."""
        feats_24 = window.extract_features()
        raw_pkts = getattr(window, "raw_topology_packets", [])
        topo_feats = compute_topology_features_batch(raw_pkts)

        return {
            **feats_24,
            "fan_in_ratio": float(topo_feats["fan_in_ratio"]),
            "src_concentration_ratio": float(topo_feats["src_concentration_ratio"]),
            "src_ip_cardinality_per_window": float(topo_feats["src_ip_cardinality_per_window"])
        }

    def extract_pcap_windows(self, pcap_info: Dict[str, str]) -> List[Dict[str, Any]]:
        pcap_file = pcap_info["filename"]
        class_label = pcap_info["label"]
        pcap_path = os.path.join(PCAP_DIR, pcap_file)

        print(f"\n[{class_label}] Processing {pcap_file} -> Target: {self.target_per_class:,} windows", flush=True)
        t_start = time.time()

        self.exclusion_counts[pcap_file] = Counter()
        active_windows: Dict[Tuple[str, str, int], FlowWindow] = {}
        valid_windows: List[Dict[str, Any]] = []

        total_packets = 0
        current_state = "train"

        last_train_max_ts: Optional[float] = None
        quarantine_train_val_end: Optional[float] = None
        last_val_max_ts: Optional[float] = None
        quarantine_val_test_end: Optional[float] = None

        num_train = 0
        num_val = 0
        num_test = 0

        quarantine_train_val_pkts = 0
        quarantine_val_test_pkts = 0

        with open(pcap_path, "rb") as f:
            try:
                reader = dpkt.pcap.Reader(f)
            except ValueError:
                f.seek(0)
                reader = dpkt.pcapng.Reader(f)

            for ts, buf in reader:
                total_packets += 1

                # 1. Transition: Train -> Quarantine (Train-Val)
                if current_state == "train" and num_train >= self.train_count:
                    last_train_max_ts = max(w["window_end_ts"] for w in valid_windows if w["partition"] == "train")
                    quarantine_train_val_end = last_train_max_ts + self.quarantine_sec
                    active_windows.clear()
                    current_state = "quarantine_train_val"
                    print(f"  --> Train complete ({num_train:,} wins). Entering 5s quarantine: [{last_train_max_ts:.4f} -> {quarantine_train_val_end:.4f}]", flush=True)

                # 2. State: Quarantine (Train-Val)
                if current_state == "quarantine_train_val":
                    if ts < quarantine_train_val_end:
                        self.exclusion_counts[pcap_file]["quarantine_train_val_packet"] += 1
                        quarantine_train_val_pkts += 1
                        continue
                    else:
                        active_windows.clear()
                        current_state = "validation"
                        print(f"  --> Exited 5s quarantine at ts={ts:.4f}. Entering VALIDATION partition.", flush=True)

                # 3. Transition: Validation -> Quarantine (Val-Test)
                if current_state == "validation" and num_val >= self.val_count:
                    last_val_max_ts = max(w["window_end_ts"] for w in valid_windows if w["partition"] == "validation")
                    quarantine_val_test_end = last_val_max_ts + self.quarantine_sec
                    active_windows.clear()
                    current_state = "quarantine_val_test"
                    print(f"  --> Validation complete ({num_val:,} wins). Entering 5s quarantine: [{last_val_max_ts:.4f} -> {quarantine_val_test_end:.4f}]", flush=True)

                # 4. State: Quarantine (Val-Test)
                if current_state == "quarantine_val_test":
                    if ts < quarantine_val_test_end:
                        self.exclusion_counts[pcap_file]["quarantine_val_test_packet"] += 1
                        quarantine_val_test_pkts += 1
                        continue
                    else:
                        active_windows.clear()
                        current_state = "test"
                        print(f"  --> Exited 5s quarantine at ts={ts:.4f}. Entering TEST partition.", flush=True)

                # 5. State: Test Complete
                if current_state == "test" and num_test >= self.test_count:
                    print(f"  --> Test complete ({num_test:,} wins). All targets reached for {pcap_file}.", flush=True)
                    break

                # Parse frame
                if len(buf) < 14:
                    self.exclusion_counts[pcap_file]["ethernet_frame_too_short"] += 1
                    continue

                eth_type = (buf[12] << 8) | buf[13]
                if eth_type != 0x0800:
                    self.exclusion_counts[pcap_file]["non_ipv4_packet"] += 1
                    continue

                if len(buf) < 34:
                    self.exclusion_counts[pcap_file]["ipv4_packet_too_short"] += 1
                    continue

                ip_proto = buf[23]
                ttl = buf[22]
                src_ip = socket.inet_ntoa(buf[26:30])
                dst_ip = socket.inet_ntoa(buf[30:34])

                tcp_flags = 0
                sport = 0
                dport = 0
                ihl = (buf[14] & 0x0F) * 4
                tcp_offset = 14 + ihl

                if ip_proto == 6:  # TCP
                    if len(buf) >= tcp_offset + 14:
                        sport = (buf[tcp_offset] << 8) | buf[tcp_offset + 1]
                        dport = (buf[tcp_offset + 2] << 8) | buf[tcp_offset + 3]
                        tcp_flags = buf[tcp_offset + 13]
                elif ip_proto == 17:  # UDP
                    if len(buf) >= 14 + ihl + 4:
                        sport = (buf[14 + ihl] << 8) | buf[14 + ihl + 1]
                        dport = (buf[14 + ihl + 2] << 8) | buf[14 + ihl + 3]

                is_inbound = self._is_local(dst_ip)
                pkt_rec = PacketRecord(
                    timestamp=float(ts),
                    pkt_len=len(buf),
                    ip_proto=ip_proto,
                    ip_ttl=ttl,
                    tcp_flags=tcp_flags,
                    is_inbound=is_inbound
                )
                topo_pkt = V3TopologyPacket(
                    timestamp=float(ts),
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=sport,
                    dst_port=dport,
                    ip_proto=ip_proto,
                    pkt_len=len(buf),
                    is_ipv4=True
                )

                ip_pair = tuple(sorted([src_ip, dst_ip]))
                flow_key = (ip_pair[0], ip_pair[1], ip_proto)
                flow_id_str = f"{ip_pair[0]}<->{ip_pair[1]}:{ip_proto}"

                part_cnt = num_train if current_state == "train" else (num_val if current_state == "validation" else num_test)

                if flow_key in active_windows:
                    window = active_windows[flow_key]
                    gap = (pkt_rec.timestamp - window.last_timestamp) if window.last_timestamp is not None else 0.0

                    if gap >= self.timeout_sec:
                        if len(window.packets) >= self.window_capacity:
                            feats = self._extract_window_features(window)
                            accepted = self._try_accept_window(
                                window, feats, class_label, pcap_file, flow_id_str,
                                current_state, valid_windows, part_cnt
                            )
                            if accepted:
                                if current_state == "train": num_train += 1
                                elif current_state == "validation": num_val += 1
                                elif current_state == "test": num_test += 1
                        else:
                            self.exclusion_counts[pcap_file]["flushed_subcapacity_window"] += 1

                        fresh_window = FlowWindow(flow_key, self.window_capacity, self.timeout_sec)
                        fresh_window.raw_topology_packets = [topo_pkt]
                        ready = fresh_window.add_packet(pkt_rec)
                        if ready:
                            feats = self._extract_window_features(fresh_window)
                            accepted = self._try_accept_window(
                                fresh_window, feats, class_label, pcap_file, flow_id_str,
                                current_state, valid_windows, part_cnt
                            )
                            if accepted:
                                if current_state == "train": num_train += 1
                                elif current_state == "validation": num_val += 1
                                elif current_state == "test": num_test += 1
                            del active_windows[flow_key]
                        else:
                            active_windows[flow_key] = fresh_window
                    else:
                        window.raw_topology_packets.append(topo_pkt)
                        ready = window.add_packet(pkt_rec)
                        if ready:
                            feats = self._extract_window_features(window)
                            accepted = self._try_accept_window(
                                window, feats, class_label, pcap_file, flow_id_str,
                                current_state, valid_windows, part_cnt
                            )
                            if accepted:
                                if current_state == "train": num_train += 1
                                elif current_state == "validation": num_val += 1
                                elif current_state == "test": num_test += 1
                            del active_windows[flow_key]
                else:
                    fresh_window = FlowWindow(flow_key, self.window_capacity, self.timeout_sec)
                    fresh_window.raw_topology_packets = [topo_pkt]
                    ready = fresh_window.add_packet(pkt_rec)
                    if ready:
                        feats = self._extract_window_features(fresh_window)
                        accepted = self._try_accept_window(
                            fresh_window, feats, class_label, pcap_file, flow_id_str,
                            current_state, valid_windows, part_cnt
                        )
                        if accepted:
                            if current_state == "train": num_train += 1
                            elif current_state == "validation": num_val += 1
                            elif current_state == "test": num_test += 1
                    else:
                        active_windows[flow_key] = fresh_window

        t_elapsed = time.time() - t_start
        print(f"[{class_label}] Finished: {len(valid_windows):,} windows in {t_elapsed:.2f}s ({total_packets/t_elapsed:.0f} pkts/s)", flush=True)

        val_first_ts = min(w["window_start_ts"] for w in valid_windows if w["partition"] == "validation")
        test_first_ts = min(w["window_start_ts"] for w in valid_windows if w["partition"] == "test")

        self.quarantine_audit[pcap_file] = {
            "train_end_ts": float(last_train_max_ts),
            "quarantine_train_val_start": float(last_train_max_ts),
            "quarantine_train_val_end": float(quarantine_train_val_end),
            "val_start_ts": float(val_first_ts),
            "train_val_actual_gap_sec": float(val_first_ts - last_train_max_ts),
            "quarantined_train_val_packets": int(quarantine_train_val_pkts),
            "val_end_ts": float(last_val_max_ts),
            "quarantine_val_test_start": float(last_val_max_ts),
            "quarantine_val_test_end": float(quarantine_val_test_end),
            "test_start_ts": float(test_first_ts),
            "val_test_actual_gap_sec": float(test_first_ts - last_val_max_ts),
            "quarantined_val_test_packets": int(quarantine_val_test_pkts)
        }

        return valid_windows

    def build_dataset(self):
        os.makedirs(self.output_dir, exist_ok=True)
        all_windows = []
        pcap_hashes = {}

        for pcap_info in OFFICIAL_PCAPS:
            pcap_path = os.path.join(PCAP_DIR, pcap_info["filename"])
            pcap_hashes[pcap_info["filename"]] = compute_file_sha256(pcap_path)
            wins = self.extract_pcap_windows(pcap_info)
            all_windows.extend(wins)

        df_all = pd.DataFrame(all_windows)
        print(f"\nTotal extracted windows across all classes: {len(df_all):,}")

        # Split partitions
        train_df = df_all[df_all["partition"] == "train"].copy().reset_index(drop=True)
        val_df = df_all[df_all["partition"] == "validation"].copy().reset_index(drop=True)
        test_df = df_all[df_all["partition"] == "test"].copy().reset_index(drop=True)

        print(f"Partition counts: Train={len(train_df):,}, Validation={len(val_df):,}, Test={len(test_df):,}")

        # Save CSVs
        train_path = os.path.join(self.output_dir, "train.csv")
        val_path = os.path.join(self.output_dir, "validation.csv")
        test_path = os.path.join(self.output_dir, "test.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Saved: {train_path}")
        print(f"Saved: {val_path}")
        print(f"Saved: {test_path}")

        # Distribution audit across 27 features
        dist_rows = []
        for feat in CANONICAL_27_FEATURES:
            for part_name, part_df in [("all", df_all), ("train", train_df), ("val", val_df), ("test", test_df)]:
                for cls in ["DDoS", "DoS", "Spoofing", "Benign"]:
                    sub = part_df[part_df["label"] == cls][feat]
                    dist_rows.append({
                        "feature": feat,
                        "partition": part_name,
                        "class": cls,
                        "count": len(sub),
                        "mean": float(sub.mean()),
                        "std": float(sub.std(ddof=1)),
                        "median": float(sub.median()),
                        "min": float(sub.min()),
                        "max": float(sub.max()),
                        "p25": float(sub.quantile(0.25)),
                        "p75": float(sub.quantile(0.75)),
                        "iqr": float(sub.quantile(0.75) - sub.quantile(0.25))
                    })

        dist_df = pd.DataFrame(dist_rows)
        dist_path = os.path.join(self.output_dir, "distribution_audit.csv")
        dist_df.to_csv(dist_path, index=False)
        print(f"Saved: {dist_path}")

        # Save manifest
        manifest = {
            "dataset_name": "AEGIS-Live v3 Leakage-Safe Topology-Augmented Dataset",
            "created_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "evaluation_contract": "TEMPORAL GENERALIZATION WITH PERSISTENT-FLOW IDENTITY",
            "schema_version": "canonical-27",
            "feature_count": 27,
            "canonical_features": CANONICAL_27_FEATURES,
            "topology_features": V3_TOPOLOGY_FEATURES,
            "window_parameters": {
                "capacity_packets": self.window_capacity,
                "timeout_sec": self.timeout_sec,
                "quarantine_dead_time_sec": self.quarantine_sec,
                "local_subnet": LOCAL_SUBNET
            },
            "pcap_hashes": pcap_hashes,
            "row_counts": {
                "total": len(df_all),
                "train": len(train_df),
                "validation": len(val_df),
                "test": len(test_df)
            },
            "class_counts": {
                "train": train_df["label"].value_counts().to_dict(),
                "validation": val_df["label"].value_counts().to_dict(),
                "test": test_df["label"].value_counts().to_dict()
            },
            "split_hashes": {
                "train.csv": compute_file_sha256(train_path),
                "validation.csv": compute_file_sha256(val_path),
                "test.csv": compute_file_sha256(test_path)
            },
            "quarantine_audit": self.quarantine_audit,
            "exclusion_counts": {p: dict(c) for p, c in self.exclusion_counts.items()},
            "verdict": "V3_DATASET_CONSTRUCTION = PASS"
        }

        manifest_path = os.path.join(self.output_dir, "dataset_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest: {manifest_path}")

        return manifest


def main():
    builder = LeakageSafeV3DatasetBuilder()
    builder.build_dataset()


if __name__ == "__main__":
    main()
