
// Constants
const R_EARTH_SIM = 40.0;
const TILT_RAD = 11.5 * (Math.PI / 180); // Earth's magnetic tilt

export const generateMagneticField = (): Float32Array => {
    const points: number[] = [];
    
    // Textbook visualization: Dense lines
    // L-shells: Inner belts (blue) to Outer belts (purple)
    const lShells = [2.5, 3.5, 4.5, 6.0, 8.0, 10.0, 15.0];
    const segmentsPerLine = 300; // Ultra high density for continuous smooth lines
    const linesPerShell = 48; // Higher radial resolution

    for (const L_val of lShells) {
        const L = L_val * R_EARTH_SIM;
        
        for (let lon = 0; lon < 360; lon += (360 / linesPerShell)) {
            const lonRad = lon * (Math.PI / 180);

            // Avoid poles to prevent bunching artifacts
            const startTheta = 0.15;
            const endTheta = Math.PI - 0.15;

            for (let i = 0; i <= segmentsPerLine; i++) {
                const t = i / segmentsPerLine;
                const theta = startTheta + t * (endTheta - startTheta); 
                
                const rRaw = L * Math.pow(Math.sin(theta), 2);
                
                // Cartesian conversion (Standard Frame)
                // Sun is at +X
                let magX = rRaw * Math.sin(theta) * Math.cos(lonRad);
                let magY = rRaw * Math.sin(theta) * Math.sin(lonRad);
                let magZ = rRaw * Math.cos(theta);

                // --- Solar Wind Interaction (Bow Shock & Magnetotail) ---
                // We compress the day-side (+X) and elongate the night-side (-X)
                // The factor depends on the angle relative to the Sun.
                
                // 1. Calculate factor based on X position relative to distance
                const dist = Math.sqrt(magX*magX + magY*magY + magZ*magZ);
                const cosSunAngle = magX / dist; // 1.0 pointing at Sun, -1.0 away

                // 2. Apply deformation
                // Compression factor: 0.6 on day side, 1.0 neutral, 1.8 on night side
                let shapeFactor = 1.0;
                if (magX > 0) {
                    shapeFactor = 0.7 + 0.3 * (1.0 - cosSunAngle); // Compress
                } else {
                    shapeFactor = 1.0 + 0.8 * Math.abs(cosSunAngle); // Elongate tail
                }

                magX *= (magX > 0 ? 0.75 : 1.4); // Hard scale on X axis
                magY *= shapeFactor; // Flaring on Y/Z
                magZ *= shapeFactor;

                const r = Math.sqrt(magX*magX + magY*magY + magZ*magZ);
                
                // Stop if inside Earth surface
                if (r < R_EARTH_SIM * 0.98) continue;

                // --- Earth Tilt Rotation ---
                // Rotate around X-axis (since Earth spin axis is Y in Sim, but Z in math)
                // Standard Frame: Z is North. We tilt around X.
                
                const rotX = magX;
                const rotY = magY * Math.cos(TILT_RAD) - magZ * Math.sin(TILT_RAD);
                const rotZ = magY * Math.sin(TILT_RAD) + magZ * Math.cos(TILT_RAD);

                // --- Swizzle to WebGPU Sim Coordinates (Y-Up) ---
                // Math Z (North) -> Sim Y
                // Math Y -> Sim Z
                const simX = rotX;
                const simY = rotZ;
                const simZ = rotY;

                // Data Structure: [x, y, z, type, vx, vy, vz, life/padding]
                // Type 6.0 = Magnetic Field Particle
                // vx = Intensity (0.0 to 1.0) - used for Color Fade (L-Shell based)
                const intensity = Math.max(0, 1.0 - (L_val / 16.0)); 

                points.push(simX, simY, simZ, 6.0, intensity, 0, 0, 0);
            }
        }
    }

    return new Float32Array(points);
};