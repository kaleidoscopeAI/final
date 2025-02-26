import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation
import json
from enum import Enum
from typing import List, Dict, Tuple

class MoleculeType(Enum):
    ORGANIC = "organic"
    INORGANIC = "inorganic"
    BIOCHEMICAL = "biochemical"

class MolecularSystem:
    def __init__(self):
        # Molecular database with structure information
        self.molecule_database = {
            # Organic Molecules
            1: {"name": "Methane", "formula": "CH4", "type": MoleculeType.ORGANIC,
                "structure": [("C", (0,0,0)), ("H", (0.4,0,0)), ("H", (-0.4,0,0)), 
                            ("H", (0,0.4,0)), ("H", (0,-0.4,0))]},
            2: {"name": "Ethanol", "formula": "C2H5OH", "type": MoleculeType.ORGANIC,
                "structure": [("C", (0,0,0)), ("C", (0.4,0,0)), ("O", (0.8,0,0)), 
                            ("H", (0,0.4,0)), ("H", (0,-0.4,0))]},
            # Inorganic Molecules
            50: {"name": "Water", "formula": "H2O", "type": MoleculeType.INORGANIC,
                 "structure": [("O", (0,0,0)), ("H", (0.4,0.4,0)), ("H", (-0.4,0.4,0))]},
            51: {"name": "Carbon Dioxide", "formula": "CO2", "type": MoleculeType.INORGANIC,
                 "structure": [("C", (0,0,0)), ("O", (0.5,0,0)), ("O", (-0.5,0,0))]},
            # Biochemical Molecules
            90: {"name": "Glucose", "formula": "C6H12O6", "type": MoleculeType.BIOCHEMICAL,
                 "structure": [("C", (0,0,0)), ("C", (0.4,0,0)), ("O", (0.8,0,0))]}
        }
        
        # Reaction database
        self.reaction_database = {
            (1, 50): {
                "name": "Methane + Water",
                "products": ["CO", "H2"],
                "energy": 50.0,
                "conditions": "High temperature, catalyst"
            }
        }
        
    def list_molecules(self):
        """Display available molecules grouped by type."""
        grouped_molecules = {}
        for id, mol in self.molecule_database.items():
            type_name = mol["type"].value
            if type_name not in grouped_molecules:
                grouped_molecules[type_name] = []
            grouped_molecules[type_name].append(
                f"{id}: {mol['name']} ({mol['formula']})"
            )
        
        print("\nAvailable Molecules:")
        print("===================")
        for type_name, molecules in grouped_molecules.items():
            print(f"\n{type_name.upper()}:")
            for mol in molecules:
                print(mol)
    
    def get_molecule(self, id: int) -> dict:
        """Retrieve molecule data by ID."""
        return self.molecule_database.get(id, None)
    
    def analyze_reaction(self, mol_id1: int, mol_id2: int) -> dict:
        """Analyze potential reaction between two molecules."""
        reaction_key = tuple(sorted([mol_id1, mol_id2]))
        return self.reaction_database.get(reaction_key, None)

class MolecularVisualizer:
    def __init__(self, resolution: int = 12):
        self.resolution = resolution
        self.system = MolecularSystem()
    
    def visualize_molecules(self, mol_ids: List[int], show_reaction: bool = False):
        """Create visualization for multiple molecules with optional reaction analysis."""
        n_molecules = len(mol_ids)
        
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=n_molecules,
            specs=[[{'type': 'scene'}] * n_molecules],
            subplot_titles=[self.system.get_molecule(id)["name"] for id in mol_ids]
        )
        
        # Add each molecule
        for i, mol_id in enumerate(mol_ids, 1):
            molecule = self.system.get_molecule(mol_id)
            if molecule:
                self._add_molecule_to_subplot(fig, molecule, row=1, col=i)
        
        # Add reaction analysis if requested
        if show_reaction and len(mol_ids) == 2:
            reaction = self.system.analyze_reaction(mol_ids[0], mol_ids[1])
            if reaction:
                fig.add_annotation(
                    text=f"Reaction: {reaction['name']}<br>Products: {', '.join(reaction['products'])}<br>Energy: {reaction['energy']} kJ/mol",
                    xref="paper", yref="paper",
                    x=0.5, y=1.1,
                    showarrow=False
                )
        
        # Update layout
        fig.update_layout(
            height=600,
            title_text="Molecular Comparison Visualization",
            showlegend=True
        )
        
        return fig
    
    def _add_molecule_to_subplot(self, fig, molecule: dict, row: int, col: int):
        """Add molecule visualization to a subplot."""
        structure = molecule["structure"]
        
        # Add atoms
        for atom_type, position in structure:
            fig.add_trace(
                go.Scatter3d(
                    x=[position[0]], y=[position[1]], z=[position[2]],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=self._get_atom_color(atom_type),
                        symbol='circle'
                    ),
                    name=f"{atom_type} ({molecule['name']})"
                ),
                row=row, col=col
            )
        
        # Add bonds
        self._add_bonds(fig, structure, row, col)
    
    def _add_bonds(self, fig, structure, row, col):
        """Add bonds between atoms."""
        for i, (atom1_type, pos1) in enumerate(structure):
            for atom2_type, pos2 in structure[i+1:]:
                distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                if distance < 0.6:  # Bond threshold
                    fig.add_trace(
                        go.Scatter3d(
                            x=[pos1[0], pos2[0]],
                            y=[pos1[1], pos2[1]],
                            z=[pos1[2], pos2[2]],
                            mode='lines',
                            line=dict(color='grey', width=2),
                            showlegend=False
                        ),
                        row=row, col=col
                    )
    
    @staticmethod
    def _get_atom_color(atom_type: str) -> str:
        """Get color for atom visualization."""
        colors = {
            'H': 'white',
            'C': 'grey',
            'O': 'red',
            'N': 'blue',
            'P': 'orange',
            'S': 'yellow'
        }
        return colors.get(atom_type, 'purple')

def main():
    visualizer = MolecularVisualizer()
    
    while True:
        print("\nMolecular Visualization System")
        print("============================")
        print("1. List available molecules")
        print("2. Visualize single molecule")
        print("3. Compare two molecules")
        print("4. Analyze reaction")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            visualizer.system.list_molecules()
            
        elif choice == '2':
            mol_id = int(input("Enter molecule ID: "))
            fig = visualizer.visualize_molecules([mol_id])
            fig.show()
            
        elif choice == '3':
            mol_id1 = int(input("Enter first molecule ID: "))
            mol_id2 = int(input("Enter second molecule ID: "))
            fig = visualizer.visualize_molecules([mol_id1, mol_id2])
            fig.show()
            
        elif choice == '4':
            mol_id1 = int(input("Enter first molecule ID: "))
            mol_id2 = int(input("Enter second molecule ID: "))
            fig = visualizer.visualize_molecules([mol_id1, mol_id2], show_reaction=True)
            fig.show()
            
        elif choice == '5':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()