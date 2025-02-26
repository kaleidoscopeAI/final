import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor

class SecurityNode:
    def __init__(self, node_id: str, monitor_ports: List[int]):
        self.node_id = node_id
        self.monitor_ports = set(monitor_ports)
        self.packet_history = defaultdict(list)
        self.anomaly_patterns = set()
        self.threat_score = 0.0
        
    def process_packet(self, packet: Dict) -> Tuple[bool, float]:
        # Extract key features
        src_ip = packet['src_ip']
        dst_ip = packet['dst_ip']
        port = packet['port']
        payload_size = packet['size']
        
        # Feature vector construction
        features = np.array([
            self._calculate_frequency(src_ip),
            self._calculate_port_entropy(port),
            self._calculate_size_anomaly(payload_size),
            self._calculate_connection_pattern(src_ip, dst_ip)
        ])
        
        # Anomaly detection
        anomaly_score = self._detect_anomaly(features)
        is_anomaly = anomaly_score > 0.85
        
        if is_anomaly:
            self.anomaly_patterns.add((src_ip, port))
            self.threat_score = min(1.0, self.threat_score + 0.1)
        
        return is_anomaly, anomaly_score
    
    def _calculate_frequency(self, ip: str) -> float:
        recent = self.packet_history[ip][-100:] if self.packet_history[ip] else []
        return len(recent) / 100.0
    
    def _calculate_port_entropy(self, port: int) -> float:
        if port not in self.monitor_ports:
            return 1.0
        return 0.0
    
    def _calculate_size_anomaly(self, size: int) -> float:
        history = [p['size'] for packets in self.packet_history.values() 
                  for p in packets[-1000:]]
        if not history:
            return 0.0
        mean = np.mean(history)
        std = np.std(history) + 1e-6
        return abs(size - mean) / std
    
    def _calculate_connection_pattern(self, src: str, dst: str) -> float:
        pattern = f"{src}->{dst}"
        recent_patterns = [p for packets in self.packet_history.values() 
                         for p in packets[-1000:]]
        return len([p for p in recent_patterns if p == pattern]) / 1000.0
    
    def _detect_anomaly(self, features: np.ndarray) -> float:
        weights = np.array([0.3, 0.2, 0.3, 0.2])
        return np.dot(features, weights)

class SecurityNetwork:
    def __init__(self):
        self.network = nx.DiGraph()
        self.nodes: Dict[str, SecurityNode] = {}
        self.blocked_ips: Set[str] = set()
        
    def add_security_node(self, node_id: str, monitor_ports: List[int]):
        node = SecurityNode(node_id, monitor_ports)
        self.nodes[node_id] = node
        self.network.add_node(node_id)
        
    def process_traffic(self, packets: List[Dict]):
        with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
            futures = []
            for packet in packets:
                node_id = self._get_responsible_node(packet)
                if node_id:
                    futures.append(
                        executor.submit(self._process_packet, node_id, packet)
                    )
            
            anomalies = []
            for future in futures:
                result = future.result()
                if result:
                    anomalies.append(result)
                    
        self._handle_anomalies(anomalies)
                    
    def _get_responsible_node(self, packet: Dict) -> str:
        # Route packet to most suitable node based on IP and port
        target_port = packet['port']
        for node_id, node in self.nodes.items():
            if target_port in node.monitor_ports:
                return node_id
        return list(self.nodes.keys())[0]  # Default to first node
    
    def _process_packet(self, node_id: str, packet: Dict) -> Dict:
        if packet['src_ip'] in self.blocked_ips:
            return None
            
        node = self.nodes[node_id]
        is_anomaly, score = node.process_packet(packet)
        
        if is_anomaly:
            return {
                'node_id': node_id,
                'packet': packet,
                'score': score
            }
        return None
    
    def _handle_anomalies(self, anomalies: List[Dict]):
        for anomaly in anomalies:
            if anomaly['score'] > 0.95:
                self.blocked_ips.add(anomaly['packet']['src_ip'])
                
            # Update network topology based on anomaly patterns
            src_node = anomaly['node_id']
            for dst_node in self.nodes:
                if src_node != dst_node:
                    weight = self.network.get_edge_data(src_node, dst_node, 
                                                      default={'weight': 0})['weight']
                    new_weight = weight * 0.9 + anomaly['score'] * 0.1
                    self.network.add_edge(src_node, dst_node, weight=new_weight)

def main():
    # Initialize security network
    security_net = SecurityNetwork()
    
    # Add monitoring nodes
    security_net.add_security_node("node1", [80, 443])  # Web traffic
    security_net.add_security_node("node2", [22, 23])   # SSH/Telnet
    security_net.add_security_node("node3", [53])       # DNS
    
    # Simulate network traffic
    while True:
        packets = [
            {
                'src_ip': f"192.168.1.{np.random.randint(1, 255)}",
                'dst_ip': f"10.0.0.{np.random.randint(1, 255)}",
                'port': np.random.choice([22, 23, 53, 80, 443]),
                'size': np.random.randint(64, 1500)
            }
            for _ in range(1000)  # Process 1000 packets per batch
        ]
        
        security_net.process_traffic(packets)
        time.sleep(0.1)  # 100ms batch processing interval

if __name__ == "__main__":
    main()
