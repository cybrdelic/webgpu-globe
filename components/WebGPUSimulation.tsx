
import React, { useEffect, useRef, useState } from 'react';
import { SimulationStats, CameraState } from '../types';

// --- Matrix Math Utils ---
const mat4 = {
  perspective: (out: Float32Array, fovy: number, aspect: number, near: number, far: number) => {
    const f = 1.0 / Math.tan(fovy / 2);
    out.fill(0);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = (far + near) / (near - far);
    out[11] = -1;
    out[14] = (2 * far * near) / (near - far);
    out[15] = 0;
    return out;
  },
  lookAt: (out: Float32Array, eye: number[], center: number[], up: number[]) => {
    let x0, x1, x2, y0, y1, y2, z0, z1, z2, len;
    const eyex = eye[0], eyey = eye[1], eyez = eye[2];
    const upx = up[0], upy = up[1], upz = up[2];
    const centerx = center[0], centery = center[1], centerz = center[2];

    z0 = eyex - centerx; z1 = eyey - centery; z2 = eyez - centerz;
    len = 1 / Math.hypot(z0, z1, z2);
    z0 *= len; z1 *= len; z2 *= len;

    x0 = upy * z2 - upz * z1; x1 = upz * z0 - upx * z2; x2 = upx * z1 - upy * z0;
    len = Math.hypot(x0, x1, x2);
    if (!len) { x0 = 0; x1 = 0; x2 = 0; } else { len = 1 / len; x0 *= len; x1 *= len; x2 *= len; }

    y0 = z1 * x2 - z2 * x1; y1 = z2 * x0 - z0 * x2; y2 = z0 * x1 - z1 * x0;
    len = Math.hypot(y0, y1, y2);
    if (!len) { y0 = 0; y1 = 0; y2 = 0; } else { len = 1 / len; y0 *= len; y1 *= len; y2 *= len; }

    out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
    out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
    out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
    out[12] = -(x0 * eyex + x1 * eyey + x2 * eyez);
    out[13] = -(y0 * eyex + y1 * eyey + y2 * eyez);
    out[14] = -(z0 * eyex + z1 * eyey + z2 * eyez);
    out[15] = 1;
    return out;
  }
};

// --- Constants & Types ---
const GPU_BUFFER_USAGE = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
};

const GPU_TEXTURE_USAGE = {
  COPY_SRC: 0x01,
  COPY_DST: 0x02,
  TEXTURE_BINDING: 0x04,
  STORAGE_BINDING: 0x08,
  RENDER_ATTACHMENT: 0x10,
};

const GPUMapMode = { READ: 0x0001, WRITE: 0x0002 };
const GPU_SHADER_STAGE = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };

// --- SHADERS ---

// 1. COMPUTE SHADER (Orbital Physics) - Unchanged physics logic
const computeShaderCode = `
struct Particle {
  pos : vec4f, 
  vel : vec4f, 
};

struct SimParams {
  view : mat4x4f,   
  proj : mat4x4f,   
  gravity : f32,    
  dt : f32,         
  mouseX : f32,     
  mouseY : f32,     
  isMouseDown : f32,
  attractorStrength : f32,
  width : f32,
  height : f32,
  seed : f32, 
};

struct AtomicCounter {
  count : atomic<u32>,
};

@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<uniform> params : SimParams;
@group(0) @binding(2) var<storage, read_write> counter : AtomicCounter;

const J2 = 0.00108263;
const R_EARTH = 40.0;
const MU = 2000.0;
const RHO_0 = 0.0001; 
const H_SCALE = 5.0; 

fn calculate_accel(pos: vec3f, vel: vec3f) -> vec3f {
    let r2 = dot(pos, pos);
    let r = sqrt(r2);
    if (r < 1.0) { return vec3f(0.0); }
    let invR3 = 1.0 / (r * r * r);
    var acc = -MU * pos * invR3;
    let z = pos.y; 
    let z2 = z * z;
    let factor = (1.5 * J2 * MU * R_EARTH * R_EARTH) / (r2 * r2 * r);
    let j2_x = factor * pos.x * (5.0 * z2 / r2 - 1.0);
    let j2_z = factor * pos.z * (5.0 * z2 / r2 - 1.0);
    let j2_y = factor * pos.y * (5.0 * z2 / r2 - 3.0);
    acc.x += j2_x; acc.y += j2_y; acc.z += j2_z;
    let altitude = r - R_EARTH;
    if (altitude < 20.0 && altitude > 0.0) {
        let density = RHO_0 * exp(-altitude / H_SCALE);
        let speed = length(vel);
        let dragMag = 0.5 * density * speed * speed * 0.1; 
        acc -= normalize(vel) * dragMag;
    }
    return acc;
}

fn hash(p: u32) -> f32 {
    var p2 = p * 747796405u + 2891336453u;
    var p3 = ((p2 >> ((p2 >> 28u) + 4u)) ^ p2) * 277803737u;
    return f32((p3 >> 22u) ^ p3) / 4294967295.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let index = GlobalInvocationID.x;
  if (index >= arrayLength(&particles)) { return; }

  var p = particles[index];
  // Skip if Type is 0 (Dead) or 6 (Magnetic Field - static)
  if (p.pos.w == 0.0 || p.pos.w == 6.0) { return; }

  let dt = params.dt;
  let oldPos = p.pos.xyz;
  let oldVel = p.vel.xyz;

  var a1 = calculate_accel(oldPos, oldVel);
  if (params.isMouseDown > 0.5) {
     if (params.attractorStrength > 0.0) { a1 += -normalize(oldVel) * 2.0; } 
     else if (params.attractorStrength < 0.0) { a1 += normalize(oldPos) * 50.0; }
  }

  let v_half = oldVel + a1 * 0.5 * dt;
  let newPos = oldPos + v_half * dt;
  var a2 = calculate_accel(newPos, v_half);

  if (params.isMouseDown > 0.5) {
      if (params.attractorStrength > 0.0) { a2 += -normalize(v_half) * 2.0; } 
      else if (params.attractorStrength < 0.0) { a2 += normalize(newPos) * 50.0; }
  }

  let newVel = v_half + a2 * 0.5 * dt;

  p.pos = vec4f(newPos, p.pos.w);
  p.vel = vec4f(newVel, p.vel.w);

  if (p.pos.w == 5.0) { // Fragment decay
      p.vel.w -= dt * 0.5;
      if (p.vel.w <= 0.0) { p.pos.w = 0.0; }
  }
  
  // Kessler Trigger
  if ((p.pos.w == 2.0 || p.pos.w == 3.0) && params.attractorStrength < 0.0) {
      let rnd = hash(index + u32(params.seed));
      if (rnd > 0.99) {
          p.pos.w = 0.0; 
          for (var i = 0; i < 20; i++) {
             let newIdx = atomicAdd(&counter.count, 1u);
             let safeIdx = newIdx % arrayLength(&particles);
             var frag = particles[safeIdx];
             frag.pos = vec4f(newPos, 5.0);
             frag.vel = vec4f(newVel + vec3f(hash(newIdx*3u)-0.5, hash(newIdx*7u)-0.5, hash(newIdx*11u)-0.5)*20.0, 1.0);
             particles[safeIdx] = frag;
          }
      }
  }

  particles[index] = p;
}
`;

// 2. EARTH SHADER (Solid Mesh)
const earthShaderCode = `
struct SimParams {
  view : mat4x4f,
  proj : mat4x4f,
  gravity : f32,
  dt : f32,
  mouseX : f32,
  mouseY : f32,
  isMouseDown : f32,
  attractorStrength : f32,
  width : f32,
  height : f32,
  sunX : f32, sunY : f32, sunZ : f32,
};

@group(0) @binding(1) var<uniform> params : SimParams;
@group(1) @binding(0) var mySampler : sampler;
@group(1) @binding(1) var colorMap : texture_2d<f32>;
@group(1) @binding(2) var heightMap : texture_2d<f32>;
@group(1) @binding(3) var nightMap : texture_2d<f32>;
@group(1) @binding(4) var overlayMap : texture_2d<f32>;

struct VertexInput {
  @location(0) position : vec3f,
  @location(1) uv : vec2f,
  @location(2) normal : vec3f,
};

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) uv : vec2f,
  @location(1) normal : vec3f,
  @location(2) worldPos : vec3f,
  @location(3) distToCamera : f32,
};

const R_EARTH = 40.0;

fn hash21(p: vec2f) -> f32 {
    var p2 = fract(p * vec2f(123.34, 456.21));
    p2 += dot(p2, p2 + 45.32);
    return fract(p2.x * p2.y);
}

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
  var output : VertexOutput;
  output.uv = input.uv;
  output.normal = input.normal;
  
  // Calculate Camera Distance for LOD
  let camPos = vec3f(
      params.view[3].x * params.view[0].x + params.view[3].y * params.view[0].y + params.view[3].z * params.view[0].z,
      params.view[3].x * params.view[1].x + params.view[3].y * params.view[1].y + params.view[3].z * params.view[1].z,
      params.view[3].x * params.view[2].x + params.view[3].y * params.view[2].y + params.view[3].z * params.view[2].z
  ) * -1.0;
  
  let worldPosBase = input.position * R_EARTH;
  let dist = distance(camPos, worldPosBase);
  output.distToCamera = dist;

  // Displacement
  let heightData = textureSampleLevel(heightMap, mySampler, input.uv, 0.0).r;
  var displacement = heightData * 1.5;

  if (dist < 100.0) {
      let microNoise = hash21(input.uv * 5000.0);
      let zoomFactor = clamp((100.0 - dist) / 50.0, 0.0, 1.0);
      displacement += microNoise * 0.15 * zoomFactor;
  }

  let finalPos = normalize(input.position) * (R_EARTH + displacement);
  output.worldPos = finalPos;
  
  // Apply Matrices
  output.Position = params.proj * params.view * vec4f(finalPos, 1.0);
  
  return output;
}

@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4f {
  let normal = normalize(in.normal); // Approximate, real normal should use derivatives of height map
  let lightDir = normalize(vec3f(params.sunX, params.sunY, params.sunZ));
  let NdotL = dot(normal, lightDir);
  let terminator = smoothstep(-0.15, 0.15, NdotL);

  let dayColor = textureSample(colorMap, mySampler, in.uv).rgb;
  let nightColor = textureSample(nightMap, mySampler, in.uv).rgb;
  let overlayColor = textureSample(overlayMap, mySampler, in.uv).rgb;

  // City Lights logic
  let cityLights = pow(nightColor, vec3f(1.8)) * 3.0 * vec3f(1.0, 0.8, 0.5);
  var surfaceColor = mix(cityLights, dayColor * max(NdotL, 0.05), terminator);

  // Specular (Ocean)
  let isWater = (dayColor.b > dayColor.r * 1.1);
  if (isWater && NdotL > 0.0) {
     let viewDir = normalize(vec3f(params.view[3].x, params.view[3].y, params.view[3].z) * -1.0 - in.worldPos);
     let halfVec = normalize(lightDir + viewDir);
     let spec = pow(max(dot(normal, halfVec), 0.0), 60.0);
     surfaceColor += vec3f(0.8) * spec * terminator;
  }

  // Street Grid (Close Zoom)
  if (in.distToCamera < 60.0) {
      let streetScale = 6000.0;
      let grid = max(abs(fract(in.uv.x * streetScale) - 0.5), abs(fract(in.uv.y * streetScale) - 0.5));
      let isStreet = smoothstep(0.45, 0.48, grid);
      let mask = length(nightColor); // Only in cities
      surfaceColor = mix(surfaceColor, vec3f(1.0, 0.7, 0.3), isStreet * mask * smoothstep(60.0, 40.0, in.distToCamera));
  }

  // Borders
  if (overlayColor.g > 0.3 && overlayColor.r < 0.2) { // Heuristic for green lines on black map
      surfaceColor = mix(surfaceColor, vec3f(0.0, 0.8, 1.0), 0.5);
  }

  // Atmosphere Rim
  let viewDir = normalize(vec3f(params.view[3].x, params.view[3].y, params.view[3].z) * -1.0 - in.worldPos);
  let fresnel = 1.0 - max(dot(viewDir, normal), 0.0);
  let rim = pow(fresnel, 3.0) * max(NdotL, 0.0);
  surfaceColor += vec3f(0.2, 0.5, 1.0) * rim;

  return vec4f(surfaceColor, 1.0);
}
`;

// 3. PARTICLE SHADER (Debris / Satellites / Mag)
const particleShaderCode = `
struct Particle {
  pos : vec4f, 
  vel : vec4f, 
};

struct SimParams {
  view : mat4x4f,
  proj : mat4x4f,
  gravity : f32,
  dt : f32,
  mouseX : f32,
  mouseY : f32,
  isMouseDown : f32,
  attractorStrength : f32,
  width : f32,
  height : f32,
  sunX : f32, sunY : f32, sunZ : f32,
};

@group(0) @binding(0) var<storage, read> particles : array<Particle>;
@group(0) @binding(1) var<uniform> params : SimParams;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) color : vec4f,
  @location(1) uv : vec2f,
  @location(2) @interpolate(flat) instanceId : u32,
};

struct FragmentOutput {
  @location(0) color : vec4f,
  @location(1) objectId : u32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32, @builtin(instance_index) instanceIndex : u32) -> VertexOutput {
  var p = particles[instanceIndex];
  var output : VertexOutput;
  output.instanceId = instanceIndex;

  if (p.pos.w == 0.0 || p.pos.w == 1.0) { // Dead or Earth (Earth drawn in separate pass now)
      output.Position = vec4f(0.0);
      return output;
  }

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  output.uv = corner;

  var viewPos = params.view * vec4f(p.pos.xyz, 1.0);
  
  var scale = 0.15;
  if (p.pos.w == 6.0) { scale = 0.06; } // Mag field
  else if (p.pos.w == 5.0) { scale = 0.4 * p.vel.w; } // Fragment

  viewPos.x += corner.x * scale;
  viewPos.y += corner.y * scale;
  output.Position = params.proj * viewPos;

  // Color Logic
  if (p.pos.w == 2.0) { output.color = vec4f(0.0, 1.0, 1.0, 1.0); }
  else if (p.pos.w == 3.0) { output.color = vec4f(1.0, 0.6, 0.0, 1.0); }
  else if (p.pos.w == 4.0) { output.color = vec4f(1.0, 0.2, 0.2, 1.0); }
  else if (p.pos.w == 5.0) { output.color = vec4f(1.0, p.vel.w, 0.0, p.vel.w); }
  else if (p.pos.w == 6.0) { output.color = vec4f(0.2, 0.5, 1.0, p.vel.x * 0.3); } // Mag
  else { output.color = vec4f(1.0); }

  // Shadowing
  if (p.pos.w > 1.0 && p.pos.w < 6.0) {
      let lightDir = normalize(vec3f(params.sunX, params.sunY, params.sunZ));
      let sunDot = dot(p.pos.xyz, lightDir);
      if (sunDot < -5.0) {
         let distToAxis = length(cross(p.pos.xyz, lightDir));
         if (distToAxis < 40.0) { output.color = vec4f(output.color.rgb * 0.1, output.color.a); }
      }
  }

  return output;
}

@fragment
fn fs_main(in : VertexOutput) -> FragmentOutput {
  var output : FragmentOutput;
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  
  var alpha = 1.0 - smoothstep(0.8, 1.0, dist);
  output.color = vec4f(in.color.rgb, in.color.a * alpha);
  output.objectId = in.instanceId + 1u;
  return output;
}
`;

interface WebGPUSimulationProps {
  gravity: number;
  attractorStrength: number;
  onStatsUpdate: (stats: SimulationStats) => void;
  runSimulation: boolean;
  camera: CameraState;
  onError?: (msg: string) => void;
  earthTextures: { 
      colorMap: ImageBitmap; 
      heightMap: ImageBitmap; 
      nightMap: ImageBitmap;
      overlayMap: ImageBitmap;
  } | null;
  satelliteData: Float32Array | null;
  magneticData: Float32Array | null; 
  onHover?: (index: number | null, data?: Float32Array | null) => void;
  timeScale: number;
}

const MAX_CAPACITY = 400000;

export const WebGPUSimulation: React.FC<WebGPUSimulationProps> = ({
  gravity,
  attractorStrength,
  onStatsUpdate,
  runSimulation,
  camera,
  onError,
  earthTextures,
  satelliteData,
  magneticData,
  onHover,
  timeScale
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const contextRef = useRef<any | null>(null);
  const deviceRef = useRef<any | null>(null);
  const pipelineRef = useRef<any | null>(null);
  const mouseRef = useRef({ x: 0, y: 0, down: false });
  const [internalError, setInternalError] = useState<string | null>(null);
  
  const [deviceInitialized, setDeviceInitialized] = useState(false);
  const [pipelinesCreated, setPipelinesCreated] = useState(false);

  // Resources
  const depthTextureRef = useRef<any | null>(null);
  const pickingTextureRef = useRef<any | null>(null);
  const pickingBufferRef = useRef<any | null>(null);
  const uniformBufferRef = useRef<any | null>(null);
  const atomicCounterBufferRef = useRef<any | null>(null);
  const earthVertexBufferRef = useRef<any | null>(null);
  const earthIndexBufferRef = useRef<any | null>(null);
  const textureBindGroupRef = useRef<any | null>(null);

  // Stable Refs
  const cameraRef = useRef(camera);
  const onHoverRef = useRef(onHover);
  const onStatsUpdateRef = useRef(onStatsUpdate);
  const gravityRef = useRef(gravity);
  const attractorStrengthRef = useRef(attractorStrength);
  const timeScaleRef = useRef(timeScale);

  useEffect(() => { cameraRef.current = camera; }, [camera]);
  useEffect(() => { onHoverRef.current = onHover; }, [onHover]);
  useEffect(() => { onStatsUpdateRef.current = onStatsUpdate; }, [onStatsUpdate]);
  useEffect(() => { gravityRef.current = gravity; }, [gravity]);
  useEffect(() => { attractorStrengthRef.current = attractorStrength; }, [attractorStrength]);
  useEffect(() => { timeScaleRef.current = timeScale; }, [timeScale]);

  // --- MESH GENERATION (UV SPHERE) ---
  const createSphereMesh = (device: GPUDevice) => {
      const radius = 1.0;
      const latSegments = 128; // High res
      const lonSegments = 128;
      
      const vertices = [];
      const indices = [];

      for (let y = 0; y <= latSegments; y++) {
          const v = y / latSegments;
          const theta = v * Math.PI;
          for (let x = 0; x <= lonSegments; x++) {
              const u = x / lonSegments;
              const phi = u * 2 * Math.PI;

              const px = radius * Math.sin(theta) * Math.cos(phi);
              const py = radius * Math.cos(theta);
              const pz = radius * Math.sin(theta) * Math.sin(phi);

              // Sim coords swizzle: X->X, Y->Z, Z->Y
              vertices.push(px, pz, py); // Pos
              vertices.push(u, 1 - v); // UV
              vertices.push(px, pz, py); // Normal (same as pos for sphere)
          }
      }

      for (let y = 0; y < latSegments; y++) {
          for (let x = 0; x < lonSegments; x++) {
              const first = (y * (lonSegments + 1)) + x;
              const second = first + lonSegments + 1;
              indices.push(first, second, first + 1);
              indices.push(second, second + 1, first + 1);
          }
      }

      const vBuffer = device.createBuffer({
          size: vertices.length * 4,
          usage: GPU_BUFFER_USAGE.VERTEX | GPU_BUFFER_USAGE.COPY_DST,
          mappedAtCreation: true
      });
      new Float32Array(vBuffer.getMappedRange()).set(vertices);
      vBuffer.unmap();

      const iBuffer = device.createBuffer({
          size: indices.length * 4,
          usage: GPU_BUFFER_USAGE.INDEX | GPU_BUFFER_USAGE.COPY_DST,
          mappedAtCreation: true
      });
      new Uint32Array(iBuffer.getMappedRange()).set(indices);
      iBuffer.unmap();

      return { vBuffer, iBuffer, indexCount: indices.length };
  };

  // 1. Initialize Device & Pipelines
  useEffect(() => {
    let active = true;
    const initDevice = async () => {
      try {
        if (!(navigator as any).gpu) throw new Error("WebGPU not supported");
        const adapter = await (navigator as any).gpu.requestAdapter();
        if (!adapter) throw new Error("No WebGPU adapter");
        const device = await adapter.requestDevice();
        if (!active) return;
        deviceRef.current = device;
        
        const canvas = canvasRef.current;
        if (!canvas) return;
        const context = canvas.getContext('webgpu') as any;
        contextRef.current = context;

        const presentationFormat = (navigator as any).gpu.getPreferredCanvasFormat();
        context.configure({
            device,
            format: presentationFormat,
            alphaMode: 'premultiplied',
            usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | GPU_TEXTURE_USAGE.COPY_SRC
        });

        // Buffers
        const width = Math.max(1, canvas.clientWidth);
        const height = Math.max(1, canvas.clientHeight);
        canvas.width = width;
        canvas.height = height;

        depthTextureRef.current = device.createTexture({
            size: [width, height],
            format: 'depth24plus',
            usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT,
        });
        pickingTextureRef.current = device.createTexture({
            size: [width, height],
            format: 'r32uint',
            usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | GPU_TEXTURE_USAGE.COPY_SRC
        });
        pickingBufferRef.current = device.createBuffer({
            size: 16,
            usage: GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.MAP_READ
        });
        uniformBufferRef.current = device.createBuffer({
            size: 256,
            usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
        });
        atomicCounterBufferRef.current = device.createBuffer({
            size: 4,
            usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.COPY_SRC,
        });

        // Mesh
        const mesh = createSphereMesh(device);
        earthVertexBufferRef.current = mesh.vBuffer;
        earthIndexBufferRef.current = mesh.iBuffer;
        const earthIndexCount = mesh.indexCount;

        // Layouts
        const computeBindGroupLayout = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPU_SHADER_STAGE.COMPUTE, buffer: { type: 'storage' } },
              { binding: 1, visibility: GPU_SHADER_STAGE.COMPUTE, buffer: { type: 'uniform' } },
              { binding: 2, visibility: GPU_SHADER_STAGE.COMPUTE, buffer: { type: 'storage' } } 
            ]
        });
        
        const particleBindGroupLayout = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPU_SHADER_STAGE.VERTEX, buffer: { type: 'read-only-storage' } },
              { binding: 1, visibility: GPU_SHADER_STAGE.VERTEX, buffer: { type: 'uniform' } }
            ]
        });

        const earthBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 1, visibility: GPU_SHADER_STAGE.VERTEX | GPU_SHADER_STAGE.FRAGMENT, buffer: { type: 'uniform' } }
            ]
        });

        const textureBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPU_SHADER_STAGE.VERTEX | GPU_SHADER_STAGE.FRAGMENT, sampler: { type: 'filtering' } },
                { binding: 1, visibility: GPU_SHADER_STAGE.FRAGMENT, texture: {} },
                { binding: 2, visibility: GPU_SHADER_STAGE.VERTEX, texture: {} },
                { binding: 3, visibility: GPU_SHADER_STAGE.FRAGMENT, texture: {} },
                { binding: 4, visibility: GPU_SHADER_STAGE.FRAGMENT, texture: {} },
            ]
        });

        // Pipeline Creation
        const computeModule = device.createShaderModule({ code: computeShaderCode });
        const computePipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        // Earth Pipeline (Opaque Mesh)
        const earthModule = device.createShaderModule({ code: earthShaderCode });
        const earthPipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [earthBindGroupLayout, textureBindGroupLayout] }),
            vertex: {
                module: earthModule, entryPoint: 'vs_main',
                buffers: [{ arrayStride: 32, attributes: [
                    { shaderLocation: 0, offset: 0, format: 'float32x3' }, // Pos
                    { shaderLocation: 1, offset: 12, format: 'float32x2' }, // UV
                    { shaderLocation: 2, offset: 20, format: 'float32x3' }  // Normal
                ]}]
            },
            fragment: {
                module: earthModule, entryPoint: 'fs_main',
                targets: [{ format: presentationFormat }, { format: 'r32uint', writeMask: 0 }] 
            },
            primitive: { topology: 'triangle-list', cullMode: 'back' },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            }
        });

        // Particle Pipeline (Transparent Additive)
        const particleModule = device.createShaderModule({ code: particleShaderCode });
        const particlePipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [particleBindGroupLayout] }),
            vertex: { module: particleModule, entryPoint: 'vs_main' },
            fragment: {
                module: particleModule, entryPoint: 'fs_main',
                targets: [
                    { 
                        format: presentationFormat, 
                        blend: { 
                             color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                             alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
                        } 
                    },
                    { format: 'r32uint' }
                ]
            },
            primitive: { topology: 'triangle-list' },
            depthStencil: {
                depthWriteEnabled: false, // Don't write depth for transparent particles
                depthCompare: 'less',
                format: 'depth24plus',
            }
        });

        pipelineRef.current = {
            computePipeline,
            earthPipeline,
            particlePipeline,
            computeBindGroupLayout,
            particleBindGroupLayout,
            earthBindGroupLayout,
            textureBindGroupLayout,
            earthIndexCount
        };

        setDeviceInitialized(true);
      } catch (e: any) {
         setInternalError(e.message);
      }
    };
    initDevice();
    return () => { active = false; };
  }, []);

  // 2. Data Upload
  useEffect(() => {
      if (!deviceInitialized || !earthTextures) return;
      const device = deviceRef.current;

      // Textures
      const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'clamp-to-edge' });
      const createTex = (bmp: ImageBitmap) => {
          const tex = device.createTexture({ size: [bmp.width, bmp.height, 1], format: 'rgba8unorm', usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST | GPU_TEXTURE_USAGE.RENDER_ATTACHMENT });
          device.queue.copyExternalImageToTexture({ source: bmp }, { texture: tex }, [bmp.width, bmp.height]);
          return tex;
      };
      
      const t1 = createTex(earthTextures.colorMap);
      const t2 = createTex(earthTextures.heightMap);
      const t3 = createTex(earthTextures.nightMap);
      const t4 = createTex(earthTextures.overlayMap);

      textureBindGroupRef.current = device.createBindGroup({
          layout: pipelineRef.current.textureBindGroupLayout,
          entries: [
              { binding: 0, resource: sampler },
              { binding: 1, resource: t1.createView() },
              { binding: 2, resource: t2.createView() },
              { binding: 3, resource: t3.createView() },
              { binding: 4, resource: t4.createView() },
          ]
      });

      // Particle Buffers
      const magCount = magneticData ? magneticData.length / 8 : 0;
      const satCount = satelliteData ? satelliteData.length / 8 : 0;
      const totalCount = magCount + satCount;
      const bufferSize = Math.max(MAX_CAPACITY, totalCount + 1000) * 8 * 4;

      const bufferData = new Float32Array(bufferSize / 4);
      if (magneticData) bufferData.set(magneticData, 0);
      if (satelliteData) bufferData.set(satelliteData, magneticData ? magneticData.length : 0);

      const particleBuffer = device.createBuffer({
          size: bufferSize,
          usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.COPY_SRC,
          mappedAtCreation: true
      });
      new Float32Array(particleBuffer.getMappedRange()).set(bufferData);
      particleBuffer.unmap();

      const atomicInit = new Uint32Array([totalCount]);
      device.queue.writeBuffer(atomicCounterBufferRef.current, 0, atomicInit);

      // Create Bind Groups
      pipelineRef.current.computeBindGroup = device.createBindGroup({
          layout: pipelineRef.current.computeBindGroupLayout,
          entries: [
              { binding: 0, resource: { buffer: particleBuffer } },
              { binding: 1, resource: { buffer: uniformBufferRef.current } },
              { binding: 2, resource: { buffer: atomicCounterBufferRef.current } }
          ]
      });

      pipelineRef.current.particleBindGroup = device.createBindGroup({
          layout: pipelineRef.current.particleBindGroupLayout,
          entries: [
              { binding: 0, resource: { buffer: particleBuffer } },
              { binding: 1, resource: { buffer: uniformBufferRef.current } }
          ]
      });

      pipelineRef.current.earthBindGroup = device.createBindGroup({
          layout: pipelineRef.current.earthBindGroupLayout,
          entries: [
              { binding: 1, resource: { buffer: uniformBufferRef.current } }
          ]
      });

      pipelineRef.current.magEndIdx = magCount;
      pipelineRef.current.totalCount = totalCount;
      
      setPipelinesCreated(true);
  }, [deviceInitialized, earthTextures, satelliteData, magneticData]);

  // 3. Render Loop
  useEffect(() => {
      if (!pipelinesCreated || !runSimulation || internalError) return;
      let frameId = 0;
      let lastTime = performance.now();
      let accumTime = 0;
      let simTime = Date.now();
      const TIME_CALIBRATION = 0.007;

      const viewMatrix = new Float32Array(16);
      const projMatrix = new Float32Array(16);

      const render = (time: number) => {
          const dt = Math.min((time - lastTime) / 1000, 0.1);
          lastTime = time;
          const device = deviceRef.current;
          const canvas = canvasRef.current;

          // Resize
          const dpr = window.devicePixelRatio || 1;
          const cw = Math.floor(canvas.clientWidth * dpr);
          const ch = Math.floor(canvas.clientHeight * dpr);
          if (canvas.width !== cw || canvas.height !== ch) {
              canvas.width = cw; canvas.height = ch;
              if (depthTextureRef.current) depthTextureRef.current.destroy();
              if (pickingTextureRef.current) pickingTextureRef.current.destroy();
              depthTextureRef.current = device.createTexture({ size: [cw, ch], format: 'depth24plus', usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT });
              pickingTextureRef.current = device.createTexture({ size: [cw, ch], format: 'r32uint', usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | GPU_TEXTURE_USAGE.COPY_SRC });
          }

          // Sim Update
          simTime += dt * 1000 * timeScaleRef.current;
          accumTime += dt;
          if (accumTime > 0.5) {
              onStatsUpdateRef.current({
                  particleCount: pipelineRef.current.totalCount,
                  activeFragmentCount: 0,
                  fps: Math.round(1/dt),
                  riskLevel: 'LOW',
                  collisionEvents: 0,
                  energyTotal: pipelineRef.current.totalCount * 100,
                  simulatedTime: simTime
              });
              accumTime = 0;
          }

          // Uniforms
          const dayMillis = 86400000;
          const sunAngle = ((simTime % dayMillis) / dayMillis) * 2 * Math.PI;
          const sunX = Math.sin(sunAngle + Math.PI);
          const sunZ = Math.cos(sunAngle + Math.PI);

          const cam = cameraRef.current;
          const cx = Math.sin(cam.yaw) * Math.cos(cam.pitch) * cam.zoom;
          const cy = Math.sin(cam.pitch) * cam.zoom;
          const cz = Math.cos(cam.yaw) * Math.cos(cam.pitch) * cam.zoom;
          mat4.perspective(projMatrix, Math.PI / 4, cw/ch, 1, 2000);
          mat4.lookAt(viewMatrix, [cx, cy, cz], [0, 0, 0], [0, 1, 0]);

          device.queue.writeBuffer(uniformBufferRef.current, 0, viewMatrix);
          device.queue.writeBuffer(uniformBufferRef.current, 64, projMatrix);
          const params = new Float32Array([gravityRef.current, dt * TIME_CALIBRATION * timeScaleRef.current, mouseRef.current.x, mouseRef.current.y, mouseRef.current.down ? 1.0 : 0.0, attractorStrengthRef.current, cw, ch, sunX, 0.2, sunZ, 0]);
          device.queue.writeBuffer(uniformBufferRef.current, 128, params);

          // Render
          const cmd = device.createCommandEncoder();
          
          // Compute Pass
          const cp = cmd.beginComputePass();
          cp.setPipeline(pipelineRef.current.computePipeline);
          cp.setBindGroup(0, pipelineRef.current.computeBindGroup);
          cp.dispatchWorkgroups(Math.ceil(MAX_CAPACITY / 64));
          cp.end();

          const rp = cmd.beginRenderPass({
              colorAttachments: [
                  { view: contextRef.current.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' },
                  { view: pickingTextureRef.current.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }
              ],
              depthStencilAttachment: { view: depthTextureRef.current.createView(), depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' }
          });

          // Draw Earth (Solid Mesh)
          rp.setPipeline(pipelineRef.current.earthPipeline);
          rp.setBindGroup(0, pipelineRef.current.earthBindGroup);
          rp.setBindGroup(1, textureBindGroupRef.current);
          rp.setVertexBuffer(0, earthVertexBufferRef.current);
          rp.setIndexBuffer(earthIndexBufferRef.current, 'uint32');
          rp.drawIndexed(pipelineRef.current.earthIndexCount);

          // Draw Particles (Transparent)
          rp.setPipeline(pipelineRef.current.particlePipeline);
          rp.setBindGroup(0, pipelineRef.current.particleBindGroup);
          rp.draw(6, MAX_CAPACITY);

          rp.end();

          // Picking Readback
          if (mouseRef.current.x >= 0 && mouseRef.current.y >= 0) {
              cmd.copyTextureToBuffer({ texture: pickingTextureRef.current, origin: { x: mouseRef.current.x, y: mouseRef.current.y, z: 0 } }, { buffer: pickingBufferRef.current, bytesPerRow: 256 }, { width: 1, height: 1, depthOrArrayLayers: 1 });
          }

          device.queue.submit([cmd.finish()]);

          // Process Picking
          if (pickingBufferRef.current.mapState === 'unmapped' && onHoverRef.current) {
               pickingBufferRef.current.mapAsync(GPUMapMode.READ).then(() => {
                   const arr = new Uint32Array(pickingBufferRef.current.getMappedRange());
                   const id = arr[0];
                   pickingBufferRef.current.unmap();
                   if (id > 0) {
                       const idx = id - 1;
                       const magEnd = pipelineRef.current.magEndIdx;
                       if (idx < magEnd) onHoverRef.current(-3);
                       else onHoverRef.current(1000000 + (idx - magEnd));
                   } else {
                       onHoverRef.current(null);
                   }
               });
          }

          frameId = requestAnimationFrame(render);
      };
      frameId = requestAnimationFrame(render);
      return () => cancelAnimationFrame(frameId);
  }, [pipelinesCreated, runSimulation, internalError]);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      mouseRef.current.x = (e.clientX - rect.left) * dpr;
      mouseRef.current.y = (e.clientY - rect.top) * dpr;
    }
  };

  if (internalError) return <div className="text-red-500 bg-black p-4">GPU Error: {internalError}</div>;

  return (
    <canvas
      ref={canvasRef}
      className="block w-full h-full cursor-crosshair touch-none bg-black"
      onMouseMove={handleMouseMove}
      onMouseDown={() => mouseRef.current.down = true}
      onMouseUp={() => mouseRef.current.down = false}
      onMouseLeave={() => mouseRef.current.down = false}
    />
  );
};
