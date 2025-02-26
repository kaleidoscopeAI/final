import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator

class DynamicMolecularGrid:
    def __init__(self, resolution: int = 10):
        self.resolution = resolution
        self.grid_points = resolution + 1  # Include endpoints
        self.atoms = []
        
    def create_base_grid(self):
        """Create evenly distributed grid lines in 3D."""
        # Create base coordinates
        coords = np.linspace(-1, 1, self.grid_points)
        
        # Initialize lists to store lines
        lines_x, lines_y, lines_z = [], [], []
        
        # Create lines along X axis
        for y in coords:
            for z in coords:
                line = np.array([(x, y, z) for x in coords])
                lines_x.append(line)
                
        # Create lines along Y axis
        for x in coords:
            for z in coords:
                line = np.array([(x, y, z) for y in coords])
                lines_y.append(line)
                
        # Create lines along Z axis
        for x in coords:
            for y in coords:
                line = np.array([(x, y, z) for z in coords])
                lines_z.append(line)
                
        return lines_x, lines_y, lines_z
    
    def compute_field_potential(self, point, atoms):
        """Compute molecular field potential at a given point."""
        potential = 0
        for atom in atoms:
            pos = np.array(atom['position'])
            dist = np.linalg.norm(point - pos)
            # Add small epsilon to prevent division by zero
            dist = max(dist, 1e-6)
            # Field strength decreases with square of distance
            strength = atom['charge'] / (dist * dist)
            potential += strength
        return potential
    
    def deform_grid(self, lines_x, lines_y, lines_z):
        """Deform grid lines based on molecular field."""
        deformed_x = []
        deformed_y = []
        deformed_z = []
        
        # Create interpolation grid for the field
        coords = np.linspace(-1, 1, self.grid_points)
        X, Y, Z = np.meshgrid(coords, coords, coords)
        field = np.zeros((self.grid_points, self.grid_points, self.grid_points))
        
        # Compute field potential at each grid point
        for i in range(self.grid_points):
            for j in range(self.grid_points):
                for k in range(self.grid_points):
                    point = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                    field[i,j,k] = self.compute_field_potential(point, self.atoms)
        
        # Create interpolator for the field
        interpolator = RegularGridInterpolator((coords, coords, coords), field)
        
        # Deform each line based on field potential
        max_displacement = 0.2  # Maximum displacement of points
        
        def deform_line(line):
            points = []
            for point in line:
                # Get field value at point
                field_value = interpolator(point)[0]
                # Calculate displacement vector
                displacement = max_displacement * field_value
                # Add displacement to original position
                deformed_point = point + displacement * np.sign(field_value)
                points.append(deformed_point)
            return np.array(points)
        
        # Apply deformation to all lines
        for line in lines_x:
            deformed_x.append(deform_line(line))
        for line in lines_y:
            deformed_y.append(deform_line(line))
        for line in lines_z:
            deformed_z.append(deform_line(line))
            
        return deformed_x, deformed_y, deformed_z
    
    def add_atom(self, position, atom_type, charge):
        """Add an atom to the molecular structure."""
        self.atoms.append({
            'position': np.array(position),
            'type': atom_type,
            'charge': charge
        })
    
    def visualize(self):
        """Create interactive 3D visualization."""
        # Create base grid
        lines_x, lines_y, lines_z = self.create_base_grid()
        
        # Deform grid based on molecular field
        if self.atoms:
            lines_x, lines_y, lines_z = self.deform_grid(lines_x, lines_y, lines_z)
        
        # Create figure
        fig = go.Figure()
        
        # Add grid lines with gradient color based on z-coordinate
        def add_lines(lines, color='blue'):
            for line in lines:
                fig.add_trace(go.Scatter3d(
                    x=line[:,0], y=line[:,1], z=line[:,2],
                    mode='lines',
                    line=dict(
                        color=color,
                        width=2
                    ),
                    showlegend=False
                ))
        
        # Add all grid lines
        add_lines(lines_x, 'rgb(70,130,180)')
        add_lines(lines_y, 'rgb(70,130,180)')
        add_lines(lines_z, 'rgb(70,130,180)')
        
        # Add atoms as spheres
        atom_colors = {
            'H': 'white',
            'C': 'grey',
            'N': 'blue',
            'O': 'red'
        }
        
        for atom in self.atoms:
            pos = atom['position']
            color = atom_colors.get(atom['type'], 'purple')
            
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers',
                marker=dict(
                    size=10,
                    color=color,
                    symbol='circle'
                ),
                name=f"{atom['type']} atom"
            ))
        
        # Update layout
        fig.update_layout(
            scene=dict(
                aspectmode='cube',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                xaxis=dict(range=[-1.1, 1.1], title='X'),
                yaxis=dict(range=[-1.1, 1.1], title='Y'),
                zaxis=dict(range=[-1.1, 1.1], title='Z')
            ),
            title='Dynamic Molecular Grid Visualization'
        )
        
        fig.show()

# Example usage
if __name__ == "__main__":
    # Create molecular grid
    mol_grid = DynamicMolecularGrid(resolution=8)
    
    # Add water molecule
    mol_grid.add_atom((0.0, 0.0, 0.0), 'O', charge=-0.834)
    
    # Add hydrogens at water bond angle (104.5°)
    angle = 104.5 * np.pi / 180
    bond_length = 0.5
    
    h1_x = bond_length * np.sin(angle/2)
    h1_y = bond_length * np.cos(angle/2)
    h2_x = -bond_length * np.sin(angle/2)
    h2_y = bond_length * np.cos(angle/2)
    
    mol_grid.add_atom((h1_x, h1_y, 0.0), 'H', charge=0.417)
    mol_grid.add_atom((h2_x, h2_y, 0.0), 'H', charge=0.417)
    
    # Visualize
    mol_grid.visualize()