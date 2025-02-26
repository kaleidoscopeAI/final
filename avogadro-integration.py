import os
import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PySide6.QtCore import Qt, Signal, Slot
import avogadro
from avogadro.core import Molecule, Atom, Bond
from avogadro.qtgui import (
    PeriodicTableView, 
    MoleculeViewWidget, 
    ToolPluginFactory, 
    CustomTool
)
from rdkit import Chem
from rdkit.Chem import AllChem
import threading
import queue

class QuantumViewerTool(CustomTool):
    """Custom Avogadro tool for quantum visualization"""
    def __init__(self):
        super().__init__()
        self.action_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.start()

    def mousePressEvent(self, event):
        """Handle mouse press events for atom selection"""
        if event.button() == Qt.LeftButton:
            hit = self.hit_test(event.pos())
            if hit.type == Avogadro.AtomType:
                self.action_queue.put(('select_atom', hit.index))

    def _process_queue(self):
        while self.running:
            try:
                action, data = self.action_queue.get(timeout=0.1)
                if action == 'select_atom':
                    self._handle_atom_selection(data)
            except queue.Empty:
                continue

    def _handle_atom_selection(self, atom_idx):
        atom = self.molecule.atom(atom_idx)
        self.molecule.setAtomSelected(atom_idx, True)
        self.widget.update()

class QuantumMoleculeWidget(MoleculeViewWidget):
    """Enhanced molecule viewer with quantum visualization"""
    stateChanged = Signal(dict)  # Emit quantum state updates

    def __init__(self, parent=None):
        super().__init__(parent)
        self.quantum_overlay = True
        self.state_data = {}
        self.setup_tools()

    def setup_tools(self):
        factory = ToolPluginFactory()
        factory.registerPlugin('Quantum', QuantumViewerTool)
        self.addTool('Quantum')

    def update_quantum_state(self, state_data: dict):
        """Update quantum state visualization"""
        self.state_data = state_data
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.quantum_overlay and self.state_data:
            self._draw_quantum_overlay()

    def _draw_quantum_overlay(self):
        painter = self.painter()
        painter.begin(self)
        
        # Draw quantum state representations
        for atom_idx, state in self.state_data.get('atom_states', {}).items():
            pos = self.molecule.atom(atom_idx).position3d()
            screen_pos = self.camera.project(pos)
            
            # Draw quantum state indicator
            radius = state.get('amplitude', 1.0) * 10
            painter.setPen(Qt.red)
            painter.drawEllipse(screen_pos, radius, radius)
            
        painter.end()

class AvogadroQuantumIntegration:
    """Main integration class for Avogadro2 and quantum system"""
    def __init__(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.window = QMainWindow()
        self.setup_ui()
        self.setup_avogadro()
        self.init_quantum_system()

    def setup_ui(self):
        self.window.setWindowTitle("Quantum Molecular Viewer")
        self.window.resize(1200, 800)

        # Central widget
        central = QWidget()
        layout = QVBoxLayout(central)
        
        # Create molecule viewer
        self.viewer = QuantumMoleculeWidget()
        layout.addWidget(self.viewer)
        
        self.window.setCentralWidget(central)

    def setup_avogadro(self):
        # Initialize Avogadro core
        self.avo = avogadro.core()
        self.molecule = None
        self.force_field = None

    def init_quantum_system(self):
        """Initialize quantum processing system"""
        from quantum_processor import QuantumProcessor
        self.quantum = QuantumProcessor(n_qubits=6)
        self.quantum_thread = threading.Thread(target=self._quantum_loop)
        self.quantum_thread.start()

    def _quantum_loop(self):
        """Background quantum state evolution"""
        while True:
            if self.molecule:
                state = self.quantum.evolve_state(self.molecule)
                self.viewer.update_quantum_state({
                    'atom_states': self._map_quantum_to_atoms(state)
                })
            time.sleep(0.1)

    def load_molecule(self, identifier: str) -> bool:
        """Load molecule from SMILES or file"""
        try:
            if self._is_smiles(identifier):
                mol = Chem.MolFromSmiles(identifier)
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                self.molecule = self._convert_rdkit_to_avogadro(mol)
            else:
                self.molecule = self.avo.io.FileFormatManager.readFile(identifier)

            # Set molecule in viewer
            self.viewer.setMolecule(self.molecule)
            return True
        except Exception as e:
            print(f"Failed to load molecule: {e}")
            return False

    def _convert_rdkit_to_avogadro(self, rdkit_mol) -> Molecule:
        """Convert RDKit molecule to Avogadro format"""
        avo_mol = Molecule()
        
        # Transfer atoms
        conf = rdkit_mol.GetConformer()
        for i in range(rdkit_mol.GetNumAtoms()):
            atom = rdkit_mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            avo_atom = avo_mol.addAtom(atom.GetAtomicNum())
            avo_atom.setPosition3d(pos.x, pos.y, pos.z)
            
        # Transfer bonds
        for bond in rdkit_mol.GetBonds():
            avo_mol.addBond(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond.GetBondTypeAsDouble()
            )
            
        return avo_mol

    def _map_quantum_to_atoms(self, quantum_state: np.ndarray) -> dict:
        """Map quantum state to atomic visualization data"""
        n_atoms = self.molecule.atomCount()
        state_dim = len(quantum_state)
        atom_states = {}
        
        for i in range(n_atoms):
            # Map quantum amplitudes to atoms
            idx_start = i * (state_dim // n_atoms)
            idx_end = (i + 1) * (state_dim // n_atoms)
            amplitude = np.sum(np.abs(quantum_state[idx_start:idx_end])**2)
            
            atom_states[i] = {
                'amplitude': amplitude,
                'phase': np.angle(np.mean(quantum_state[idx_start:idx_end]))
            }
            
        return atom_states

    @staticmethod
    def _is_smiles(identifier: str) -> bool:
        return all(c in 'CNOPSFIBrClHc[]()=#-+' for c in identifier)

    def run(self):
        """Start the application"""
        self.window.show()
        return self.app.exec_()

def main():
    integration = AvogadroQuantumIntegration()
    
    # Load example molecule
    integration.load_molecule("CCO")  # Ethanol
    
    # Run application
    return integration.run()

if __name__ == "__main__":
    sys.exit(main())