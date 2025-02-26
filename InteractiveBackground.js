import React, { useState, useEffect, useCallback } from 'react';
import { useTheme } from 'next-themes';
import { useMotion } from '@react-hook/motion';

const InteractiveBackground = () => {
  const { resolvedTheme } = useTheme();
  const viewportRef = React.useRef(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [particles, setParticles] = useState([]);
  const [connections, setConnections] = useState([]);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  const [mounted, setMounted] = useState(false);

  // Adaptive color scheme
  const themeColors = {
    dark: {
      particle: '#4f46e5',
      connection: '#3b82f677',
      highlight: '#10b981'
    },
    light: {
      particle: '#3b82f6',
      connection: '#60a5fa77',
      highlight: '#059669'
    }
  };

  // Generate initial particles
  const generateParticles = useCallback((count = 50) => {
    return Array.from({ length: count }, (_, i) => ({
      id: i,
      position: {
        x: Math.random() * dimensions.width,
        y: Math.random() * dimensions.height,
        z: Math.random() * 1000 - 500
      },
      velocity: {
        x: (Math.random() - 0.5) * 0.1,
        y: (Math.random() - 0.5) * 0.1,
        z: (Math.random() - 0.5) * 0.1
      },
      size: 2 + Math.random() * 3,
      connections: 0
    }));
  }, [dimensions]);

  // Mouse interaction
  const { motion } = useMotion({
    defaultState: { x: 0, y: 0 },
    damping: 0.1
  });

  // Resize handler
  useEffect(() => {
    const updateDimensions = () => {
      setDimensions({
        width: window.innerWidth,
        height: window.innerHeight
      });
    };

    window.addEventListener('resize', updateDimensions);
    updateDimensions();
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Initialize particles
  useEffect(() => {
    if (dimensions.width > 0 && dimensions.height > 0) {
      setParticles(generateParticles(100));
    }
  }, [dimensions, generateParticles]);

  // Animation loop
  useEffect(() => {
    let animationFrame;
    
    const animate = () => {
      setParticles(prev => prev.map(p => ({
        ...p,
        position: {
          x: (p.position.x + p.velocity.x + motion.x * 0.1) % dimensions.width,
          y: (p.position.y + p.velocity.y + motion.y * 0.1) % dimensions.height,
          z: p.position.z + p.velocity.z
        },
        size: Math.max(1, p.size + Math.sin(Date.now() * 0.002 + p.id) * 0.5)
      })));

      animationFrame = requestAnimationFrame(animate);
    };

    animationFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrame);
  }, [motion, dimensions]);

  // Adaptive connection system
  useEffect(() => {
    const newConnections = [];
    particles.forEach((a, i) => {
      particles.slice(i + 1).forEach(b => {
        const distance = Math.sqrt(
          Math.pow(a.position.x - b.position.x, 2) +
          Math.pow(a.position.y - b.position.y, 2) +
          Math.pow(a.position.z - b.position.z, 2)
        );

        if (distance < 150 && a.connections < 3 && b.connections < 3) {
          newConnections.push({ a, b, strength: 1 - distance / 150 });
          a.connections++;
          b.connections++;
        }
      });
    });
    setConnections(newConnections);
  }, [particles]);

  // Project 3D to 2D with perspective
  const project = (x, y, z) => {
    const scale = 1000 / (1000 + z + rotation.y * 100);
    return {
      x: (x + rotation.x * 50) * scale + dimensions.width / 2,
      y: (y + rotation.y * 50) * scale + dimensions.height / 2,
      scale
    };
  };

  return (
    <div 
      ref={viewportRef}
      className="fixed inset-0 -z-10 overflow-hidden"
      style={{
        background: resolvedTheme === 'dark' 
          ? 'radial-gradient(circle, #1e1b4b 0%, #0f172a 100%)' 
          : 'radial-gradient(circle, #f0f4ff 0%, #e0e7ff 100%)'
      }}
    >
      <svg 
        className="absolute inset-0" 
        width={dimensions.width} 
        height={dimensions.height}
        style={{ pointerEvents: 'none' }}
      >
        {/* Connections */}
        {connections.map(({ a, b, strength }, i) => {
          const aProj = project(a.position.x, a.position.y, a.position.z);
          const bProj = project(b.position.x, b.position.y, b.position.z);
          return (
            <line
              key={i}
              x1={aProj.x}
              y1={aProj.y}
              x2={bProj.x}
              y2={bProj.y}
              stroke={themeColors[resolvedTheme].connection}
              strokeWidth={0.5 * strength}
              opacity={0.3 * strength}
            />
          );
        })}

        {/* Particles */}
        {particles.map((particle) => {
          const { x, y, scale } = project(
            particle.position.x,
            particle.position.y,
            particle.position.z
          );

          return (
            <circle
              key={particle.id}
              cx={x}
              cy={y}
              r={particle.size * scale}
              fill={themeColors[resolvedTheme].particle}
              opacity={0.8 - Math.abs(particle.position.z) / 1000}
            >
              <animate
                attributeName="r"
                values={`${particle.size * scale};${particle.size * scale * 1.2};${particle.size * scale}`}
                dur={`${2 + (particle.id % 3)}s`}
                repeatCount="indefinite"
              />
            </circle>
          );
        })}
      </svg>
    </div>
  );
};

export default InteractiveBackground;
