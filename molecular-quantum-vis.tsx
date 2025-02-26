import React, { useEffect, useRef, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const QuantumMolecularVisualization = () => {
  const canvasRef = useRef(null);
  const [rotation, setRotation] = useState({ x: 0, y: 0, z: 0 });

  // Initialize with sample molecule data
  const [molecularData, setMolecularData] = useState({
    atoms: [
      { id: 1, element: 'C', position: { x: 0, y: 0, z: 0 } },
      { id: 2, element: 'O', position: { x: 20, y: 0, z: 0 } },
      { id: 3, element: 'N', position: { x: -20, y: 0, z: 0 } },
      { id: 4, element: 'H', position: { x: 0, y: 20, z: 0 } }
    ],
    bonds: [
      { start: { x: 0, y: 0, z: 0 }, end: { x: 20, y: 0, z: 0 }, order: 2 },
      { start: { x: 0, y: 0, z: 0 }, end: { x: -20, y: 0, z: 0 }, order: 1 },
      { start: { x: 0, y: 0, z: 0 }, end: { x: 0, y: 20, z: 0 }, order: 1 }
    ]
  });

  // Initialize quantum data
  const [quantumData, setQuantumData] = useState({
    states: Array.from({ length: 50 }, () => ({ value: Math.random() })),
    field: Array.from({ length: 64 }, () => Math.random() * 0.5)
  });

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animationId;

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Update rotation
      setRotation(prev => ({
        ...prev,
        y: prev.y + 0.01
      }));

      // Draw molecular structure
      drawMolecule(ctx);
      
      // Update quantum states
      updateQuantumStates();

      animationId = requestAnimationFrame(render);
    };

    const drawMolecule = (ctx) => {
      // Draw bonds
      molecularData.bonds.forEach(bond => {
        const start = project3D(bond.start);
        const end = project3D(bond.end);
        drawBond(ctx, start, end, bond.order);
      });

      // Draw atoms
      molecularData.atoms.forEach(atom => {
        const pos = project3D(atom.position);
        drawAtom(ctx, pos, atom.element);
      });
    };

    const project3D = (point) => {
      // Apply rotation
      const cosY = Math.cos(rotation.y);
      const sinY = Math.sin(rotation.y);
      const cosX = Math.cos(rotation.x);
      const sinX = Math.sin(rotation.x);

      const x = point.x * cosY - point.z * sinY;
      const y = point.y;
      const z = point.x * sinY + point.z * cosY;

      // Apply perspective
      const scale = 800 / (800 + z);
      return {
        x: canvas.width/2 + x * scale,
        y: canvas.height/2 + y * scale,
        z,
        scale
      };
    };

    const drawAtom = (ctx, pos, element) => {
      const colors = {
        'H': '#FFFFFF',
        'C': '#808080',
        'N': '#0000FF',
        'O': '#FF0000'
      };

      const radius = element === 'H' ? 10 : 15;
      
      // Create gradient for 3D effect
      const gradient = ctx.createRadialGradient(
        pos.x - radius/3,
        pos.y - radius/3,
        0,
        pos.x,
        pos.y,
        radius
      );
      gradient.addColorStop(0, '#FFFFFF');
      gradient.addColorStop(0.3, colors[element]);
      gradient.addColorStop(1, '#000000');

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius * pos.scale, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();

      // Add quantum glow effect
      const quantumValue = quantumData.states[0].value;
      if (quantumValue > 0.5) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius * 1.5 * pos.scale, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(0, 255, 255, ${quantumValue * 0.5})`;
        ctx.stroke();
      }
    };

    const drawBond = (ctx, start, end, order) => {
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.lineTo(end.x, end.y);
      ctx.lineWidth = 2 * start.scale;
      ctx.strokeStyle = '#FFFFFF';
      ctx.stroke();
    };

    const updateQuantumStates = () => {
      setQuantumData(prev => ({
        states: prev.states.map(state => ({
          value: 0.5 + 0.5 * Math.sin(Date.now() * 0.001 + Math.random())
        })),
        field: prev.field.map(v => 
          Math.max(0, Math.min(1, v + (Math.random() - 0.5) * 0.1))
        )
      }));
    };

    render();
    return () => cancelAnimationFrame(animationId);
  }, [rotation, molecularData]);

  return (
    <div className="grid grid-cols-2 gap-4 bg-gray-900 p-4">
      <div className="bg-gray-800 rounded-lg p-4">
        <h2 className="text-xl font-bold text-white mb-4">Quantum Molecular Structure</h2>
        <canvas
          ref={canvasRef}
          width={600}
          height={600}
          className="bg-black rounded-lg"
        />
      </div>
      
      <div className="space-y-4">
        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-xl font-bold text-white mb-4">Quantum States</h2>
          <div className="h-64">
            <ResponsiveContainer>
              <LineChart data={quantumData.states}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#00ffff" 
                  dot={false} 
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-xl font-bold text-white mb-4">Field Intensity</h2>
          <div className="grid grid-cols-8 gap-1">
            {quantumData.field.slice(0, 64).map((value, i) => (
              <div
                key={i}
                className="aspect-square rounded"
                style={{
                  backgroundColor: `rgba(0, 255, 255, ${value})`,
                  transition: 'background-color 0.5s ease'
                }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default QuantumMolecularVisualization;