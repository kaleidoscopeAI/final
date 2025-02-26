def display_molecules(database: MolecularDatabase):
    """Display available molecules with error handling."""
    try:
        molecules = database._molecules.values()
        if not molecules:
            print("No molecules available in database.")
            return
            
        print("\nAvailable Molecules:")
        print("===================")
        for molecule in molecules:
            print(f"{molecule.id}: {molecule.name} ({molecule.formula})")
            
    except Exception as e:
        logging.error(f"Error displaying molecules: {str(e)}")
        raise

def visualize_single_molecule(database: MolecularDatabase, 
                            visualizer: MolecularVisualizer):
    """Handle single molecule visualization with error checking."""
    try:
        mol_id = int(input("Enter molecule ID: ").strip())
        
        # Validate molecule exists
        molecule = database.get_molecule(mol_id)
        if not molecule:
            raise InvalidMoleculeError(f"Molecule ID {mol_id} not found")
            
        # Create visualization
        fig = visualizer.create_visualization([mol_id])
        fig.show()
        
    except ValueError:
        raise ValueError("Please enter a valid number")
    except Exception as e:
        logging.error(f"Visualization error: {str(e)}")
        raise

def analyze_molecular_reaction(database: MolecularDatabase,
                             analyzer: ReactionAnalyzer,
                             visualizer: MolecularVisualizer):
    """Handle reaction analysis with comprehensive error checking."""
    try:
        # Get molecule IDs
        mol_id1 = int(input("Enter first molecule ID: ").strip())
        mol_id2 = int(input("Enter second molecule ID: ").strip())
        
        # Validate molecules exist
        mol1 = database.get_molecule(mol_id1)
        mol2 = database.get_molecule(mol_id2)
        
        if not mol1 or not mol2:
            raise InvalidMoleculeError("One or both molecules not found")
        
        # Analyze reaction
        result = analyzer.analyze_reaction(mol_id1, mol_id2)
        
        # Display results
        print("\nReaction Analysis Results:")
        print("=========================")
        print(f"Reactants: {mol1.name} + {mol2.name}")
        print(f"Energy change: {result['energy']:.2f} kJ/mol")
        print(f"Probability: {result['probability']*100:.1f}%")
        if result['products']:
            print("Possible products:", ", ".join(result['products']))
        else:
            print("No predicted products")
        
        # Show visualization
        fig = visualizer.create_visualization([mol_id1, mol_id2])
        fig.show()
        
    except ValueError:
        raise ValueError("Please enter valid numbers")
    except Exception as e:
        logging.error(f"Reaction analysis error: {str(e)}")
        raise
        import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation
from enum import Enum
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import logging
import sys
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('molecular_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class MoleculeError(Exception):
    """Base exception class for molecular system errors."""
    pass

class InvalidMoleculeError(MoleculeError):
    """Raised when an invalid molecule is requested."""
    pass

class ReactionError(MoleculeError):
    """Raised when there's an error in reaction analysis."""
    pass

class ValidationError(MoleculeError):
    """Raised when input validation fails."""
    pass

@dataclass
class Atom:
    """Represents an atom with its properties."""
    symbol: str
    position: Tuple[float, float, float]
    charge: float = 0.0
    
    def validate(self):
        """Validate atom properties."""
        if not isinstance(self.symbol, str) or len(self.symbol) > 2:
            raise ValidationError(f"Invalid atom symbol: {self.symbol}")
        if len(self.position) != 3:
            raise ValidationError(f"Invalid position coordinates for {self.symbol}")
        return True

@dataclass
class Molecule:
    """Represents a molecule with its properties."""
    id: int
    name: str
    formula: str
    type: str
    atoms: List[Atom]
    energy: float = 0.0
    
    def validate(self):
        """Validate molecule properties."""
        try:
            for atom in self.atoms:
                atom.validate()
            return True
        except ValidationError as e:
            raise ValidationError(f"Validation failed for molecule {self.name}: {str(e)}")

class MolecularDatabase:
    """Manages molecular data and operations."""
    
    def __init__(self):
        self._molecules: Dict[int, Molecule] = {}
        self._reactions: Dict[Tuple[int, int], Dict] = {}
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the molecular database with predefined molecules."""
        try:
            # Initialize basic molecules
            self._add_basic_molecules()
            # Initialize reaction database
            self._initialize_reactions()
            logging.info("Database initialized successfully")
        except Exception as e:
            logging.error(f"Database initialization failed: {str(e)}")
            raise
    
    def _add_basic_molecules(self):
        """Add basic molecules to the database."""
        molecules = [
            Molecule(
                id=1,
                name="Water",
                formula="H2O",
                type="inorganic",
                atoms=[
                    Atom("O", (0, 0, 0)),
                    Atom("H", (0.4, 0.4, 0)),
                    Atom("H", (-0.4, 0.4, 0))
                ]
            ),
            Molecule(
                id=2,
                name="Methane",
                formula="CH4",
                type="organic",
                atoms=[
                    Atom("C", (0, 0, 0)),
                    Atom("H", (0.4, 0, 0)),
                    Atom("H", (-0.4, 0, 0)),
                    Atom("H", (0, 0.4, 0)),
                    Atom("H", (0, -0.4, 0))
                ]
            ),
            # Add more molecules as needed
        ]
        
        for molecule in molecules:
            self.add_molecule(molecule)
    
    def add_molecule(self, molecule: Molecule):
        """Add a molecule to the database with validation."""
        try:
            molecule.validate()
            self._molecules[molecule.id] = molecule
            logging.info(f"Added molecule: {molecule.name}")
        except ValidationError as e:
            logging.error(f"Failed to add molecule: {str(e)}")
            raise

    def get_molecule(self, molecule_id: int) -> Optional[Molecule]:
        """Retrieve a molecule by ID with error handling."""
        try:
            molecule = self._molecules.get(molecule_id)
            if not molecule:
                raise InvalidMoleculeError(f"Molecule ID {molecule_id} not found")
            return molecule
        except Exception as e:
            logging.error(f"Error retrieving molecule {molecule_id}: {str(e)}")
            raise

class ReactionAnalyzer:
    """Analyzes molecular reactions."""
    
    def __init__(self, database: MolecularDatabase):
        self.database = database
        self.reactions = {}
    
    def analyze_reaction(self, mol_id1: int, mol_id2: int) -> Dict:
        """Analyze potential reaction between two molecules."""
        try:
            mol1 = self.database.get_molecule(mol_id1)
            mol2 = self.database.get_molecule(mol_id2)
            
            if not mol1 or not mol2:
                raise ReactionError("Invalid molecules for reaction analysis")
            
            # Perform reaction analysis
            result = self._compute_reaction(mol1, mol2)
            logging.info(f"Analyzed reaction between {mol1.name} and {mol2.name}")
            return result
            
        except Exception as e:
            logging.error(f"Reaction analysis failed: {str(e)}")
            raise
    
    def _compute_reaction(self, mol1: Molecule, mol2: Molecule) -> Dict:
        """Compute reaction properties and outcomes."""
        # Implement reaction computation logic here
        return {
            "energy": self._calculate_reaction_energy(mol1, mol2),
            "products": self._predict_products(mol1, mol2),
            "probability": self._calculate_reaction_probability(mol1, mol2)
        }
    
    def _calculate_reaction_energy(self, mol1: Molecule, mol2: Molecule) -> float:
        """Calculate the energy change for the reaction."""
        # Implement energy calculation
        return 0.0
    
    def _predict_products(self, mol1: Molecule, mol2: Molecule) -> List[str]:
        """Predict possible products of the reaction."""
        # Implement product prediction
        return []
    
    def _calculate_reaction_probability(self, mol1: Molecule, mol2: Molecule) -> float:
        """Calculate the probability of reaction occurrence."""
        # Implement probability calculation
        return 0.0

class MolecularVisualizer:
    """Handles molecular visualization."""
    
    def __init__(self, database: MolecularDatabase):
        self.database = database
    
    def create_visualization(self, molecule_ids: List[int]) -> go.Figure:
        """Create a visualization for multiple molecules."""
        try:
            molecules = [self.database.get_molecule(mid) for mid in molecule_ids]
            fig = self._setup_figure(len(molecules))
            
            for i, molecule in enumerate(molecules, 1):
                self._add_molecule_to_plot(fig, molecule, row=1, col=i)
            
            self._update_layout(fig)
            return fig
            
        except Exception as e:
            logging.error(f"Visualization creation failed: {str(e)}")
            raise
    
    def _setup_figure(self, n_molecules: int) -> go.Figure:
        """Set up the figure with appropriate subplots."""
        return make_subplots(
            rows=1, cols=n_molecules,
            specs=[[{'type': 'scene'}] * n_molecules]
        )
    
    def _add_molecule_to_plot(self, fig: go.Figure, molecule: Molecule, row: int, col: int):
        """Add a molecule to the plot."""
        # Add atoms
        for atom in molecule.atoms:
            self._add_atom_to_plot(fig, atom, molecule.name, row, col)
        
        # Add bonds
        self._add_bonds_to_plot(fig, molecule, row, col)
    
    def _update_layout(self, fig: go.Figure):
        """Update the figure layout."""
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Molecular Visualization"
        )

def main():
    """Main program loop with error handling."""
    try:
        database = MolecularDatabase()
        analyzer = ReactionAnalyzer(database)
        visualizer = MolecularVisualizer(database)
        
        while True:
            try:
                print("\nMolecular Analysis System")
                print("========================")
                print("1. View available molecules")
                print("2. Visualize molecule")
                print("3. Analyze reaction")
                print("4. Exit")
                
                choice = input("\nEnter choice (1-4): ").strip()
                
                if choice == '1':
                    display_molecules(database)
                elif choice == '2':
                    visualize_single_molecule(database, visualizer)
                elif choice == '3':
                    analyze_molecular_reaction(database, analyzer, visualizer)
                elif choice == '4':
                    print("Exiting...")
                    break
                else:
                    print("Invalid choice. Please try again.")
                    
            except ValidationError as e:
                print(f"Validation error: {str(e)}")
            except MoleculeError as e:
                print(f"Molecule error: {str(e)}")
            except ValueError as e:
                print(f"Invalid input: {str(e)}")
            except Exception as e:
                print(f"An unexpected error occurred: {str(e)}")
                logging.exception("Unexpected error in main loop")
    
    except Exception as e:
        print(f"Critical error: {str(e)}")
        logging.critical("Critical error in main program", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()