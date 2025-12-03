
export interface SimParams {
  gravity: number;
  deltaTime: number;
  mouseX: number;
  mouseY: number;
  isMouseDown: number; // 1.0 for true, 0.0 for false
  attractorStrength: number;
  canvasWidth: number;
  canvasHeight: number;
  timeScale: number;
}

export interface CameraState {
  pitch: number;
  yaw: number;
  zoom: number;
}

export interface SimulationStats {
  particleCount: number;
  activeFragmentCount: number;
  fps: number;
  riskLevel: 'LOW' | 'MODERATE' | 'CRITICAL' | 'CATASTROPHIC';
  collisionEvents: number;
  energyTotal: number;
  simulatedTime: number; // Unix Timestamp
}

export interface AIAnalysis {
  analysis: string;
  recommendation: string;
  mitigationProtocol: string;
}

export interface SatelliteMetadata {
  id: number;
  name: string;
  noradId: string;
  intDesignator: string;
  type: 'PAYLOAD' | 'ROCKET BODY' | 'DEBRIS' | 'UNKNOWN' | 'FRAGMENT';
}

export interface OrbitalElements {
  semiMajorAxis: number; // km
  eccentricity: number; 
  inclination: number; // degrees
  raan: number; // degrees
  argPerigee: number; // degrees
  trueAnomaly: number; // degrees
  period: number; // minutes
  altitude: number; // km
  speed: number; // km/s
}
