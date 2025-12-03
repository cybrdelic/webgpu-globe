
import { SatelliteMetadata, OrbitalElements } from '../types';

// Constants for Earth and Physics
const EARTH_RADIUS_KM = 6371.0;
const GM_KM3_S2 = 398600.4418;
const SIM_EARTH_RADIUS = 40.0;
const SCALE_POS = SIM_EARTH_RADIUS / EARTH_RADIUS_KM;
const SCALE_VEL = Math.sqrt(2000.0 / SIM_EARTH_RADIUS) / Math.sqrt(GM_KM3_S2 / EARTH_RADIUS_KM);

// Type IDs for GPU
const TYPE_EARTH = 1.0;
const TYPE_PAYLOAD = 2.0;
const TYPE_ROCKET_BODY = 3.0;
const TYPE_DEBRIS = 4.0;

export interface CartesianState {
  x: number; y: number; z: number;
  vx: number; vy: number; vz: number;
}

export interface SatelliteData {
    buffer: Float32Array;
    metadata: SatelliteMetadata[];
}

// --- ANALYTICAL SOLVERS ---

export const calculateOrbitalElements = (
    pos: { x: number, y: number, z: number }, // Sim Units
    vel: { x: number, y: number, z: number }  // Sim Units
): OrbitalElements => {
    // 1. Convert Sim Units back to Real World (KM and KM/S)
    const r_vec = [pos.x / SCALE_POS, pos.z / SCALE_POS, pos.y / SCALE_POS]; // Un-swizzle Y/Z
    const v_vec = [vel.x / SCALE_VEL, vel.z / SCALE_VEL, vel.y / SCALE_VEL]; 

    const r = Math.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2);
    const v = Math.sqrt(v_vec[0]**2 + v_vec[1]**2 + v_vec[2]**2);

    // Specific Angular Momentum (h = r x v)
    const h_vec = [
        r_vec[1]*v_vec[2] - r_vec[2]*v_vec[1],
        r_vec[2]*v_vec[0] - r_vec[0]*v_vec[2],
        r_vec[0]*v_vec[1] - r_vec[1]*v_vec[0]
    ];
    const h = Math.sqrt(h_vec[0]**2 + h_vec[1]**2 + h_vec[2]**2);

    // Node Line (n = k x h)
    const n_vec = [-h_vec[1], h_vec[0], 0];
    const n = Math.sqrt(n_vec[0]**2 + n_vec[1]**2 + n_vec[2]**2);

    // Eccentricity Vector
    const r_dot_v = r_vec[0]*v_vec[0] + r_vec[1]*v_vec[1] + r_vec[2]*v_vec[2];
    const e_vec = r_vec.map((rj, i) => (1/GM_KM3_S2) * ((v**2 - GM_KM3_S2/r)*rj - (r_dot_v)*v_vec[i]));
    const e = Math.sqrt(e_vec[0]**2 + e_vec[1]**2 + e_vec[2]**2);

    // Specific Energy & Semi-Major Axis
    const E = (v**2)/2 - GM_KM3_S2/r;
    const a = -GM_KM3_S2 / (2*E);

    // Inclination
    const i_rad = Math.acos(h_vec[2] / h);

    // RAAN (Omega)
    let omega_rad = Math.acos(n_vec[0] / n);
    if (n_vec[1] < 0) omega_rad = 2*Math.PI - omega_rad;
    if (isNaN(omega_rad)) omega_rad = 0; // Equatorial orbit

    // Argument of Perigee (w)
    let w_rad = Math.acos((n_vec[0]*e_vec[0] + n_vec[1]*e_vec[1] + n_vec[2]*e_vec[2]) / (n*e));
    if (e_vec[2] < 0) w_rad = 2*Math.PI - w_rad;
    if (isNaN(w_rad)) w_rad = 0;

    // True Anomaly (nu)
    let nu_rad = Math.acos((e_vec[0]*r_vec[0] + e_vec[1]*r_vec[1] + e_vec[2]*r_vec[2]) / (e*r));
    if (r_dot_v < 0) nu_rad = 2*Math.PI - nu_rad;
    if (isNaN(nu_rad)) nu_rad = 0;

    return {
        semiMajorAxis: a,
        eccentricity: e,
        inclination: i_rad * (180/Math.PI),
        raan: omega_rad * (180/Math.PI),
        argPerigee: w_rad * (180/Math.PI),
        trueAnomaly: nu_rad * (180/Math.PI),
        period: 2 * Math.PI * Math.sqrt((a**3)/GM_KM3_S2) / 60, // Minutes
        altitude: r - EARTH_RADIUS_KM,
        speed: v
    };
};

// Fetch active satellites from CelesTrak
export const fetchRealSatelliteData = async (): Promise<SatelliteData> => {
  try {
    const PROXY = 'https://corsproxy.io/?'; 
    const URL = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle';
    
    const response = await fetch(PROXY + encodeURIComponent(URL));
    if (!response.ok) throw new Error('Network response was not ok');
    const text = await response.text();
    return parseTLEToBuffer(text);
  } catch (error) {
    console.error("Failed to fetch TLE data", error);
    throw error;
  }
};

const parseTLEToBuffer = (tleData: string): SatelliteData => {
  const lines = tleData.split('\n').map(l => l.trim()).filter(l => l.length > 0);
  const states: CartesianState[] = [];
  const meta: SatelliteMetadata[] = [];

  let currentName = "UNKNOWN";

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    // Check if it's a name line (0 ...) or just a name preceding 1...
    if (!line.startsWith('1 ') && !line.startsWith('2 ')) {
        currentName = line;
        continue;
    }

    if (line.startsWith('1 ') && (i + 1 < lines.length) && lines[i+1].startsWith('2 ')) {
        const line1 = line;
        const line2 = lines[i+1];
        i++; // Skip next line as we consumed it

        try {
            const state = parseTLELine(line2);
            states.push(state);
            
            // Extract Metadata
            const noradId = line1.substring(2, 7).trim();
            const intDes = line1.substring(9, 17).trim();
            
            // Heuristic Type Detection
            let type: SatelliteMetadata['type'] = 'PAYLOAD';
            let typeId = TYPE_PAYLOAD;

            if (currentName.includes('DEB') || currentName.includes('COOLANT') || currentName.includes('SHIELD')) {
                type = 'DEBRIS';
                typeId = TYPE_DEBRIS;
            } else if (currentName.includes('R/B') || currentName.includes('ROCKET') || currentName.includes('STAGE')) {
                type = 'ROCKET BODY';
                typeId = TYPE_ROCKET_BODY;
            }

            // Store metadata with the mapped typeId for later filtering if needed
            meta.push({
                id: states.length - 1,
                name: currentName,
                noradId,
                intDesignator: intDes,
                type
            });
            
            // Attach typeId to state for buffer generation
            (state as any).typeId = typeId;

        } catch (e) {
            // Skip bad TLEs
        }
    }
  }

  // Convert to Float32Array for GPU [x, y, z, type, vx, vy, vz, padding]
  const buffer = new Float32Array(states.length * 8);
  states.forEach((sat, idx) => {
    const offset = idx * 8;
    buffer[offset] = sat.x;
    buffer[offset+1] = sat.y;
    buffer[offset+2] = sat.z;
    buffer[offset+3] = (sat as any).typeId; // Store Type ID
    buffer[offset+4] = sat.vx;
    buffer[offset+5] = sat.vy;
    buffer[offset+6] = sat.vz;
    buffer[offset+7] = 0.0; // Padding
  });

  return { buffer, metadata: meta };
};

const parseTLELine = (line2: string): CartesianState => {
  // Parse Line 2
  const i = parseFloat(line2.substring(8, 16)) * (Math.PI / 180.0); // Inclination
  const omega = parseFloat(line2.substring(17, 25)) * (Math.PI / 180.0); // RAAN
  const e = parseFloat("0." + line2.substring(26, 33)); // Eccentricity
  const w = parseFloat(line2.substring(34, 42)) * (Math.PI / 180.0); // Arg of Perigee
  const M = parseFloat(line2.substring(43, 51)) * (Math.PI / 180.0); // Mean Anomaly
  const n = parseFloat(line2.substring(52, 63)) * (2 * Math.PI / 86400.0); // Mean Motion (rad/s)

  // Semi-major axis (km)
  const a = Math.pow(GM_KM3_S2 / (n * n), 1.0 / 3.0);

  // Solve Kepler Equation for Eccentric Anomaly (E)
  let E = M;
  for (let k = 0; k < 10; k++) {
    E = M + e * Math.sin(E);
  }

  // Position/Velocity in Orbital Plane (Perifocal)
  const cosE = Math.cos(E);
  const sinE = Math.sin(E);
  const r = a * (1 - e * cosE);
  const vFactor = Math.sqrt(GM_KM3_S2 * a) / r;

  const x_orbit = a * (cosE - e);
  const y_orbit = a * Math.sqrt(1 - e * e) * sinE;
  
  const vx_orbit = -vFactor * sinE;
  const vy_orbit = vFactor * Math.sqrt(1 - e * e) * cosE;

  // Rotation Matrices elements
  const cos_omega = Math.cos(omega); // RAAN
  const sin_omega = Math.sin(omega);
  const cos_w = Math.cos(w); // Arg Perigee
  const sin_w = Math.sin(w);
  const cos_i = Math.cos(i); // Inclination
  const sin_i = Math.sin(i);

  // Rotate to ECI (Earth Centered Inertial)
  const xx = cos_omega * cos_w - sin_omega * sin_w * cos_i;
  const xy = -cos_omega * sin_w - sin_omega * cos_w * cos_i;
  const yx = sin_omega * cos_w + cos_omega * sin_w * cos_i;
  const yy = -sin_omega * sin_w + cos_omega * cos_w * cos_i;
  const zx = sin_w * sin_i;
  const zy = cos_w * sin_i;

  const x = xx * x_orbit + xy * y_orbit;
  const y = yx * x_orbit + yy * y_orbit; 
  const z = zx * x_orbit + zy * y_orbit;

  const vx = xx * vx_orbit + xy * vy_orbit;
  const vy = yx * vx_orbit + yy * vy_orbit;
  const vz = zx * vx_orbit + zy * vy_orbit;

  return {
    x: x * SCALE_POS,
    y: z * SCALE_POS, // Swizzle for sim: Z -> Y (Up)
    z: y * SCALE_POS, // Swizzle for sim: Y -> Z
    vx: vx * SCALE_VEL,
    vy: vz * SCALE_VEL,
    vz: vy * SCALE_VEL
  };
};
