# -*- coding: utf-8 -*-
"""
GeNIe XDSL Conditional Probability Table (CPT) Parser
Parse .xdsl files and export each node's CPT as structured CSV
"""

# ==================== Configuration Area ====================
INPUT_FILE = "genie_ready_2024.xdsl"           # Input .xdsl filename
OUTPUT_CPT_FILE = "data/cpt_extracted.csv"    # CPT probability table output file
# ===========================================================

import xml.etree.ElementTree as ET
import csv
import itertools
import sys
import os
from pathlib import Path
from typing import List, Dict

# Set Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


class XDSLParser:
    """Parse Bayesian network files in GeNIe XDSL format"""
    
    def __init__(self, xdsl_file: str):
        self.xdsl_file = xdsl_file
        self.tree = ET.parse(xdsl_file)
        self.root = self.tree.getroot()
        self.nodes = {}
        self.edges = []
        self.all_nodes_list = []  # Save all node names for column generation
        
    def parse(self):
        """Parse XDSL file"""
        self._parse_nodes()
        self._parse_edges()
        self._parse_cpts()
        
    def _parse_nodes(self):
        """Parse all nodes and their states"""
        # Nodes in XDSL are typically under <nodes> or <cpt> tags
        for cpt_node in self.root.findall('.//cpt'):
            node_id = cpt_node.get('id')
            
            # Get states
            states = []
            for state in cpt_node.findall('.//state'):
                state_id = state.get('id')
                states.append(state_id)
            
            # Get parent nodes (from <parents> tag)
            parents = []
            parents_elem = cpt_node.find('parents')
            if parents_elem is not None and parents_elem.text:
                # Parent nodes are separated by spaces
                parents = parents_elem.text.strip().split()
            
            self.nodes[node_id] = {
                'id': node_id,
                'states': states,
                'parents': parents,
                'probabilities': []
            }
            self.all_nodes_list.append(node_id)
    
    def _parse_edges(self):
        """Build edge list from parsed parent-child relationships"""
        for node_id, node_data in self.nodes.items():
            for parent in node_data['parents']:
                self.edges.append((parent, node_id))
    
    def _parse_cpts(self):
        """Parse conditional probability tables"""
        for cpt_node in self.root.findall('.//cpt'):
            node_id = cpt_node.get('id')
            
            # Find probability table
            probabilities_elem = cpt_node.find('probabilities')
            if probabilities_elem is not None and probabilities_elem.text:
                # Parse probability values (space separated)
                prob_text = probabilities_elem.text.strip()
                probs = [float(p) for p in prob_text.split()]
                self.nodes[node_id]['probabilities'] = probs
    
    def extract_cpt_rows(self) -> List[Dict]:
        """
        Expand each node's CPT into structured rows
        Return format: [{
            'target_node': str,
            'target_state': str,
            'parent_states': dict,  # {parent_name: state_value}
            'probability': float
        }]
        """
        all_rows = []
        
        for node_id, node_data in self.nodes.items():
            parents = node_data['parents']
            states = node_data['states']
            probs = node_data['probabilities']
            
            if not probs:
                continue
            
            # Get parent node states
            parent_states_list = []
            for parent in parents:
                if parent in self.nodes:
                    parent_states_list.append(self.nodes[parent]['states'])
            
            # Generate all combinations of parent node states
            if parent_states_list:
                parent_combinations = list(itertools.product(*parent_states_list))
            else:
                parent_combinations = [()]  # Case with no parent nodes
            
            # CPT is arranged according to GeNIe order:
            # For each parent combination, list all state probabilities of the current node
            idx = 0
            for parent_combo in parent_combinations:
                for state in states:
                    if idx < len(probs):
                        # Build parent state dictionary
                        parent_dict = {}
                        for i, parent in enumerate(parents):
                            parent_dict[parent] = parent_combo[i]
                        
                        row = {
                            'target_node': node_id,
                            'target_state': state,
                            'parent_states': parent_dict,
                            'probability': probs[idx]
                        }
                        all_rows.append(row)
                        idx += 1
        
        return all_rows
    
    def export_to_csv(self, output_file: str):
        """
        Export to CSV file
        Format: Target_Node, Target_State, Probability, Parent_Node1, Parent_Node2, ...
        """
        rows = self.extract_cpt_rows()
        
        if not rows:
            print("[WARNING] No probability data found")
            return
        
        # Collect all possible parent nodes
        all_parent_nodes = set()
        for row in rows:
            all_parent_nodes.update(row['parent_states'].keys())
        
        # Sort parent node columns alphabetically
        all_parent_nodes = sorted(self.all_nodes_list)
        
        # Build CSV header: Target_Node, Target_State, Probability, Parent_xxx, ...
        header = ['Target_Node', 'Target_State', 'Probability']
        for parent in all_parent_nodes:
            header.append(f'Parent_{parent}')
        
        # Write to CSV
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for row in rows:
                csv_row = [
                    row['target_node'],
                    row['target_state'],
                    row['probability']
                ]
                
                # Add parent node states (in header order)
                for parent in all_parent_nodes:
                    csv_row.append(row['parent_states'].get(parent, ''))
                
                writer.writerow(csv_row)
        
        print(f"[COMPLETE] Successfully exported {len(rows)} records to: {output_file}")
        print(f"[STATS] Contains {len(self.nodes)} nodes")
        print(f"[STATS] Parent node columns: {len(all_parent_nodes)}")
        
        return len(rows)


def main():
    print("=" * 60)
    print("GeNIe XDSL Conditional Probability Table Extraction Tool")
    print("=" * 60)
    
    # Read configuration
    input_file = INPUT_FILE
    output_cpt_file = OUTPUT_CPT_FILE
    
    # Auto-generate output filename if empty
    if not output_cpt_file or output_cpt_file.strip() == "":
        input_path = Path(input_file)
        output_cpt_file = input_path.stem + '_cpt.csv'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"[ERROR] File '{input_file}' not found")
        print(f"\nPlease check:")
        print(f"  1. Whether the filename is correct")
        print(f"  2. Whether the file is in the same directory as the script")
        print(f"  3. Current working directory: {os.getcwd()}")
        sys.exit(1)
    
    try:
        # Start parsing
        print(f"\n[PARSING] Parsing file: {input_file}")
        xdsl_parser = XDSLParser(input_file)
        xdsl_parser.parse()
        
        # Export CPT
        print(f"\n[EXPORT] Exporting conditional probability table...")
        num_rows = xdsl_parser.export_to_csv(output_cpt_file)
        
        # Display node statistics
        print(f"\n[DETAILS] Network statistics:")
        print(f"  - Total nodes: {len(xdsl_parser.nodes)}")
        print(f"  - Total edges: {len(xdsl_parser.edges)}")
        print(f"  - CPT records: {num_rows}")
        
        # Display brief information for each node
        print(f"\n[NODES] Node list:")
        for node_id, node_data in sorted(xdsl_parser.nodes.items()):
            parents = node_data['parents']
            states = node_data['states']
            num_probs = len(node_data['probabilities'])
            
            parent_info = f"{len(parents)} parent nodes" if parents else "Root node"
            print(f"  • {node_id}: {len(states)} states, {parent_info}, {num_probs} probability values")
        
        print(f"\n" + "=" * 60)
        print(f"[COMPLETE] Processing complete!")
        print(f"[OUTPUT] CPT file: {output_cpt_file}")
        print("=" * 60)
        
    except ET.ParseError as e:
        print(f"\n[ERROR] XML parsing error: {e}")
        print("Please ensure the file is in valid XDSL format (exported from GeNIe)")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()