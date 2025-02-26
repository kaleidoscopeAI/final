# specialized/molecular_patterns.py
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdDecomposition, rdMolDescriptors
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class MolecularPattern:
    """Container for molecular pattern information."""
    smiles: str
    fingerprint: np.ndarray
    features: Dict[str, float]
    quantum_signature: Optional[np.ndarray] = None
    resonance_score: float = 0.0

class PatternEncoder(nn.Module):
    """Neural network for encoding molecular patterns."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

class MolecularPatternRecognition:
    """Advanced molecular pattern recognition system."""
    
    def __init__(self, model_dim: int = 64):
        self.model_dim = model_dim
        self.pattern_encoder = PatternEncoder(input_dim=2048, hidden_dim=model_dim)
        self.pattern_database = {}
        self.similarity_threshold = 0.8
        
    def analyze_molecule(self, smiles: str) -> MolecularPattern:
        """Analyze a molecule and extract patterns."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                raise ValueError("Invalid SMILES string")
                
            # Generate fingerprint
            fingerprint = self._generate_fingerprint(mol)
            
            # Calculate features
            features = self._calculate_features(mol)
            
            # Create quantum signature if molecule is not too large
            quantum_signature = None
            if mol.GetNumAtoms() <= 50:  # Size threshold
                quantum_signature = self._generate_quantum_signature(mol)
            
            # Calculate resonance score
            resonance_score = self._calculate_resonance(mol)
            
            pattern = MolecularPattern(
                smiles=smiles,
                fingerprint=fingerprint,
                features=features,
                quantum_signature=quantum_signature,
                resonance_score=resonance_score
            )
            
            # Store pattern
            self.pattern_database[smiles] = pattern
            
            return pattern
            
        except Exception as e:
            raise RuntimeError(f"Error analyzing molecule: {str(e)}")
    
    def _generate_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """Generate molecular fingerprint."""
        # Morgan fingerprint with radius 2
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return np.array(fingerprint)
    
    def _calculate_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate molecular features."""
        features = {
            "MolWeight": Descriptors.ExactMolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "RotatableBonds": Descriptors.NumRotatableBonds(mol),
            "HBondDonors": rdMolDescriptors.CalcNumHBD(mol),
            "HBondAcceptors": rdMolDescriptors.CalcNumHBA(mol),
            "RingCount": rdMolDescriptors.CalcNumRings(mol),
            "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "Complexity": Descriptors.BertzCT(mol)
        }
        return features
    
    def _generate_quantum_signature(self, mol: Chem.Mol) -> np.ndarray:
        """Generate quantum signature for molecule."""
        # Add hydrogens and generate 3D conformer
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Get atomic positions
        conf = mol.GetConformer()
        positions = conf.GetPositions()
        
        # Create quantum state from positions
        n_atoms = mol.GetNumAtoms()
        quantum_dim = 2 ** (n_atoms.bit_length())
        state = np.zeros(quantum_dim, dtype=complex)
        
        for i, pos in enumerate(positions):
            phase = np.exp(2j * np.pi * np.sum(pos) / 30.0)
            if i < quantum_dim:
                state[i] = phase
                
        return state / np.linalg.norm(state)
    
    def _calculate_resonance(self, mol: Chem.Mol) -> float:
        """Calculate molecular resonance score."""
        # Get conjugated systems
        conjsystems = rdDecomposition.ConjugatedSystems(mol)
        
        if not conjsystems:
            return 0.0
            
        # Calculate resonance contribution from each conjugated system
        scores = []
        for system in conjsystems:
            size = len(system)
            contribution = np.tanh(size / 6.0)  # Normalize by typical aromatic ring size
            scores.append(contribution)
            
        return np.mean(scores)
    
    def find_similar_patterns(self, pattern: MolecularPattern) -> List[Tuple[str, float]]:
        """Find similar patterns in the database."""
        similarities = []
        
        for smiles, stored_pattern in self.pattern_database.items():
            if smiles != pattern.smiles:
                similarity = self._calculate_similarity(pattern, stored_pattern)
                if similarity >= self.similarity_threshold:
                    similarities.append((smiles, similarity))
                    
        return sorted(similarities, key=lambda x: x[1], reverse=True)
    
    def _calculate_similarity(self, pattern1: MolecularPattern, 
                            pattern2: MolecularPattern) -> float:
        """Calculate similarity between two patterns."""
        # Fingerprint similarity (Tanimoto)
        fp_sim = np.sum(pattern1.fingerprint & pattern2.fingerprint) / \
                np.sum(pattern1.fingerprint | pattern2.fingerprint)
        
        # Feature similarity
        common_features = set(pattern1.features.keys()) & set(pattern2.features.keys())
        if common_features:
            feature_diffs = [
                abs(pattern1.features[f] - pattern2.features[f]) / 
                max(abs(pattern1.features[f]), abs(pattern2.features[f]), 1e-6)
                for f in common_features
            ]
            feature_sim = 1 - np.mean(feature_diffs)
        else:
            feature_sim = 0.0
            
        # Quantum similarity if available
        quantum_sim = 0.0
        if (pattern1.quantum_signature is not None and 
            pattern2.quantum_signature is not None):
            quantum_sim = np.abs(np.dot(pattern1.quantum_signature.conj(),
                                      pattern2.quantum_signature))
            
        # Resonance similarity
        resonance_sim = 1 - abs(pattern1.resonance_score - pattern2.resonance_score)
        
        # Combine similarities
        weights = {
            'fingerprint': 0.4,
            'features': 0.3,
            'quantum': 0.2,
            'resonance': 0.1
        }
        
        total_sim = (weights['fingerprint'] * fp_sim +
                    weights['features'] * feature_sim +
                    weights['quantum'] * quantum_sim +
                    weights['resonance'] * resonance_sim)
        
        return float(total_sim)

    def encode_pattern(self, pattern: MolecularPattern) -> np.ndarray:
        """Encode pattern into latent space."""
        # Convert pattern to tensor
        fingerprint_tensor = torch.FloatTensor(pattern.fingerprint)
        
        # Encode
        with torch.no_grad():
            encoding = self.pattern_encoder(fingerprint_tensor)
            
        return encoding.numpy()
