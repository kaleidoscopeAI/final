# analysis/molecular/structure_analyzer.py (continued)

            # Calculate pocket properties
            center = points.mean(axis=0)
            volume = self._calculate_volume(points)
            
            if volume > 100:  # Minimum volume threshold (Å³)
                pockets.append({
                    'position': center,
                    'volume': volume,
                    'points': points
                })
        
        return pockets
        
    def _analyze_pocket_features(self, mol: Chem.Mol, 
                               pocket: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze chemical features of a binding pocket."""
        features = {
            'hydrophobic': [],
            'hbond_donors': [],
            'hbond_acceptors': [],
            'charged_positive': [],
            'charged_negative': [],
            'aromatic': []
        }
        
        # Get features from factory
        mol_features = self.feature_factory.GetFeaturesForMol(mol)
        
        # Find features near pocket
        pocket_center = pocket['position']
        for feature in mol_features:
            position = feature.GetPos()
            dist = np.linalg.norm(pocket_center - np.array(position))
            
            if dist < 5.0:  # Distance threshold (Å)
                feature_family = feature.GetFamily()
                if feature_family == 'Hydrophobic':
                    features['hydrophobic'].append(position)
                elif feature_family == 'HBondDonor':
                    features['hbond_donors'].append(position)
                elif feature_family == 'HBondAcceptor':
                    features['hbond_acceptors'].append(position)
                elif feature_family == 'positive':
                    features['charged_positive'].append(position)
                elif feature_family == 'negative':
                    features['charged_negative'].append(position)
                elif feature_family == 'Aromatic':
                    features['aromatic'].append(position)
                    
        return features
        
    def _score_binding_site(self, features: Dict[str, List]) -> float:
        """Score binding site based on feature composition."""
        score = 0.0
        
        # Weight different features
        weights = {
            'hydrophobic': 0.3,
            'hbond_donors': 0.2,
            'hbond_acceptors': 0.2,
            'charged_positive': 0.15,
            'charged_negative': 0.15,
            'aromatic': 0.2
        }
        
        for feature_type, positions in features.items():
            # Score based on feature count and distribution
            if positions:
                count_score = min(len(positions) / 5.0, 1.0)  # Normalize count
                distribution_score = self._calculate_distribution_score(positions)
                feature_score = (count_score + distribution_score) / 2
                score += weights[feature_type] * feature_score
                
        return score
        
    def _calculate_volume(self, points: np.ndarray) -> float:
        """Calculate volume of a pocket."""
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points)
            return hull.volume
        except:
            return 0.0
            
    def _calculate_distribution_score(self, positions: List) -> float:
        """Calculate how well distributed features are."""
        if len(positions) < 2:
            return 0.0
            
        positions = np.array(positions)
        
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(positions)
        
        # Score based on distance distribution
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # Prefer well-distributed features (higher std)
        return min(std_dist / mean_dist, 1.0) if mean_dist > 0 else 0.0
        
    def analyze_conformers(self, mol: Chem.Mol, 
                         n_conformers: int = 10) -> List[Dict[str, Any]]:
        """Generate and analyze multiple conformers."""
        conformers = []
        
        # Generate conformers
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=n_conformers,
            pruneRmsThresh=1.0,
            randomSeed=42
        )
        
        # Optimize each conformer
        for conf_id in range(mol.GetNumConformers()):
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
            
            # Calculate conformer properties
            energy = AllChem.MMFFGetMoleculeForceField(
                mol, confId=conf_id
            ).CalcEnergy()
            
            rmsd = Chem.rdMolAlign.GetBestRMS(
                mol, mol,
                ref_conf=0,
                probe_conf=conf_id
            ) if conf_id > 0 else 0.0
            
            conformers.append({
                'id': conf_id,
                'energy': energy,
                'rmsd': rmsd,
                'binding_sites': self.find_binding_sites(mol)
            })
            
        return conformers
        
    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status."""
        return {
            'feature_factory_initialized': bool(self.feature_factory),
            'feature_types': list(self.feature_factory.GetFeatureFamilies())
        }

if __name__ == "__main__":
    # Example usage
    analyzer = StructureAnalyzer()
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    binding_sites = analyzer.find_binding_sites(mol)
    print(f"Found {len(binding_sites)} potential binding sites")