import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { WebGPUSimulation } from './components/WebGPUSimulation';
import { analyzeDebrisField } from './services/geminiService';
import { fetchRealSatelliteData, calculateOrbitalElements, SatelliteData } from './services/orbitalPhysics';
import { fetchEarthMap, EarthTextures } from './services/earthMap';
import { generateMagneticField } from './services/magneticField';
import { SimulationStats, AIAnalysis, CameraState, SatelliteMetadata, OrbitalElements } from './types';
import { 
  CpuChipIcon, 
  GlobeAltIcon, 
  PlayIcon, 
  PauseIcon, 
  SparklesIcon,
  ArrowsPointingOutIcon,
  SignalIcon,
  CloudArrowDownIcon,
  ClockIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
  FireIcon,
  BeakerIcon,
  BoltIcon
} from '@heroicons/react/24/outline';

function App() {
  const [timeScale, setTimeScale] = useState(1.0); 
  const [isRunning, setIsRunning] = useState(true);
  const [stats, setStats] = useState<SimulationStats>({
    particleCount: 0, 
    activeFragmentCount: 0,
    fps: 0, 
    riskLevel: 'LOW', 
    collisionEvents: 0, 
    energyTotal: 0,
    simulatedTime: Date.now()
  });

  const [earthTextures, setEarthTextures] = useState<EarthTextures | null>(null);
  const [magData, setMagData] = useState<Float32Array | null>(null);
  const [showMagField, setShowMagField] = useState(false);

  const [rawSatelliteBuffer, setRawSatelliteBuffer] = useState<Float32Array | null>(null);
  const [rawMetadata, setRawMetadata] = useState<SatelliteMetadata[]>([]);

  const [searchQuery, setSearchQuery] = useState("");
  const [activeFilters, setActiveFilters] = useState({
      'PAYLOAD': true,
      'ROCKET BODY': true,
      'DEBRIS': true,
      'UNKNOWN': true
  });
  
  const [isEarthLoading, setIsEarthLoading] = useState(true);
  const [isSatsLoading, setIsSatsLoading] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);

  const [hoveredData, setHoveredData] = useState<SatelliteMetadata | null>(null);
  const [orbitalElements, setOrbitalElements] = useState<OrbitalElements | null>(null);

  const [camera, setCamera] = useState<CameraState>({ pitch: 0.5, yaw: 0.0, zoom: 250.0 });
  const [aiAnalysis, setAiAnalysis] = useState<AIAnalysis | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  
  const [triggerKessler, setTriggerKessler] = useState(false);

  useEffect(() => {
    const loadEarth = async () => {
        try {
            const textures = await fetchEarthMap();
            setEarthTextures(textures);
            
            // Generate Mag Field
            const mag = generateMagneticField();
            setMagData(mag);

        } catch (e) {
            setSimError("Failed to load NASA Earth Textures.");
        } finally {
            setIsEarthLoading(false);
        }
    };
    loadEarth();
  }, []);

  const { filteredBuffer, filteredMetadata } = useMemo(() => {
      if (!rawSatelliteBuffer || rawMetadata.length === 0) {
          return { filteredBuffer: null, filteredMetadata: [] };
      }

      const allFiltersActive = Object.values(activeFilters).every(Boolean);
      if (searchQuery === "" && allFiltersActive) {
          return { filteredBuffer: rawSatelliteBuffer, filteredMetadata: rawMetadata };
      }

      const indices: number[] = [];
      const newMeta: SatelliteMetadata[] = [];
      const lowerSearch = searchQuery.toLowerCase();

      rawMetadata.forEach((meta, idx) => {
          const typeMatch = activeFilters[meta.type] || (meta.type === 'UNKNOWN' && activeFilters['UNKNOWN']);
          if (!typeMatch) return;

          const searchMatch = searchQuery === "" || 
                              meta.name.toLowerCase().includes(lowerSearch) || 
                              meta.noradId.includes(lowerSearch);
          
          if (searchMatch) {
              indices.push(idx);
              newMeta.push(meta);
          }
      });

      const newBuffer = new Float32Array(indices.length * 8);
      indices.forEach((originalIdx, i) => {
          const srcStart = originalIdx * 8;
          const destStart = i * 8;
          newBuffer.set(rawSatelliteBuffer.subarray(srcStart, srcStart + 8), destStart);
      });

      return { filteredBuffer: newBuffer, filteredMetadata: newMeta };

  }, [rawSatelliteBuffer, rawMetadata, searchQuery, activeFilters]);


  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    try {
      const result = await analyzeDebrisField(stats);
      setAiAnalysis(result);
    } catch (e) { console.error(e); } 
    finally { setIsAnalyzing(false); }
  };

  const toggleRealData = async () => {
    if (rawSatelliteBuffer) {
        setRawSatelliteBuffer(null);
        setRawMetadata([]);
        return;
    }

    setIsSatsLoading(true);
    try {
        const result = await fetchRealSatelliteData();
        setRawSatelliteBuffer(result.buffer);
        setRawMetadata(result.metadata);
        setSimError(null);
    } catch (e) {
        setSimError("Failed to fetch Live NORAD data. Connection refused or API unavailable.");
    } finally {
        setIsSatsLoading(false);
    }
  };

  const handleHover = useCallback((code: number | null) => {
      if (code === null || !earthTextures) {
          setHoveredData(null);
          setOrbitalElements(null);
          return;
      }
      // Note: Earth logic changed from count to index check inside WebGPUSim. 
      // But here we rely on the offset logic.
      // Earth is roughly index 0 to 150000.
      
      // Since we don't know exact earth count here easily without props,
      // we rely on the specific codes passed up.
      
      if (code < 400000 && code > -1 && code < 1000000) {
         setHoveredData({
             id: -1,
             name: "EARTH SURFACE (TERRAIN)",
             noradId: "PLANET-1",
             intDesignator: "SOL-3",
             type: "UNKNOWN"
         });
         setOrbitalElements(null);
      } else if (code === -3) {
         setHoveredData({
             id: -3,
             name: "VAN ALLEN RADIATION BELT",
             noradId: "MAGNETOSPHERE",
             intDesignator: "DIPOLE",
             type: "UNKNOWN"
         });
         setOrbitalElements(null);
      } else if (code >= 1000000) {
          // It's a satellite or fragment
          const satIndex = code - 1000000;

          if (filteredMetadata[satIndex]) {
              setHoveredData(filteredMetadata[satIndex]);
              
              if (filteredBuffer) {
                 const offset = satIndex * 8;
                 const pos = { x: filteredBuffer[offset], y: filteredBuffer[offset+1], z: filteredBuffer[offset+2] };
                 const vel = { x: filteredBuffer[offset+4], y: filteredBuffer[offset+5], z: filteredBuffer[offset+6] };
                 const elements = calculateOrbitalElements(pos, vel);
                 setOrbitalElements(elements);
              }

          } else {
              setHoveredData({
                  id: -2,
                  name: "DEBRIS CLOUD FRAGMENT",
                  noradId: "KESSLER-GEN",
                  intDesignator: "N/A",
                  type: "FRAGMENT"
              });
              setOrbitalElements(null);
          }
      }
  }, [earthTextures, filteredMetadata, filteredBuffer]);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    if (triggerKessler) setTriggerKessler(false); 
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setCamera(prev => ({
        ...prev,
        yaw: prev.yaw + e.movementX * 0.005,
        pitch: Math.max(-Math.PI/2 + 0.1, Math.min(Math.PI/2 - 0.1, prev.pitch + e.movementY * 0.005))
      }));
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    setCamera(prev => ({
      ...prev,
      zoom: Math.max(42, Math.min(1200, prev.zoom + e.deltaY * 0.1)) // Allow closer zoom (42) for terrain
    }));
  };

  const simDate = new Date(stats.simulatedTime).toUTCString();

  return (
    <div className="flex h-screen w-screen bg-neutral-900 text-white overflow-hidden font-mono">
      <aside className="w-96 flex flex-col border-r border-neutral-800 bg-neutral-900/95 backdrop-blur z-10 shadow-xl shrink-0">
        <div className="p-6 border-b border-neutral-800">
          <h1 className="text-xl font-bold flex items-center gap-2 text-blue-400">
            <GlobeAltIcon className="h-6 w-6" />
            KESSLER<span className="text-white">ZERO</span>
            <span className="text-[10px] bg-blue-500/20 px-2 py-0.5 rounded text-blue-300">SCIENTIFIC</span>
          </h1>
          <p className="text-xs text-neutral-500 mt-1">J2 Perturbation & Magnetosphere Solver</p>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-8">
          {/* Telemetry */}
          <section className="bg-neutral-800/50 p-4 rounded-lg border border-neutral-700">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-[10px] text-neutral-500">TRACKED OBJECTS</p>
                <p className="text-lg font-bold">{stats.particleCount.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-[10px] text-neutral-500">SYSTEM ENERGY</p>
                <p className="text-lg font-bold">{Math.floor(stats.energyTotal/1000)} TJ</p>
              </div>
              <div className="col-span-2 border-t border-neutral-700 pt-2 mt-2">
                <p className="text-[10px] text-neutral-500">SIMULATION UTC CLOCK</p>
                <p className="text-sm font-bold text-blue-200 font-mono tracking-tight">{simDate}</p>
              </div>
            </div>
          </section>

          {/* Time Controls */}
          <section className="space-y-4">
             <h2 className="text-xs font-semibold text-neutral-400 uppercase tracking-wider flex items-center gap-2">
                <ClockIcon className="w-3 h-3" /> Temporal Control
            </h2>
            <div className="bg-neutral-800/50 p-3 rounded border border-neutral-700">
                <div className="flex justify-between text-xs mb-2">
                    <span className="text-neutral-400">PROPAGATION SPEED</span>
                    <span className="text-blue-400 font-bold">{timeScale === 1 ? 'REAL-TIME (1x)' : `${timeScale.toFixed(0)}x`}</span>
                </div>
                <input 
                    type="range" 
                    min="1" 
                    max="5000" 
                    step="10"
                    value={timeScale} 
                    onChange={(e) => setTimeScale(parseFloat(e.target.value))}
                    className="w-full accent-blue-500 h-1 bg-neutral-700 rounded-lg appearance-none cursor-pointer"
                />
            </div>
          </section>

           {/* Physics */}
           <section className="space-y-3">
            <h2 className="text-xs font-semibold text-neutral-400 uppercase tracking-wider flex items-center gap-2">
                <FireIcon className="w-3 h-3" /> Kinetic Events
            </h2>
            <button
                onMouseDown={() => setTriggerKessler(true)}
                onMouseUp={() => setTriggerKessler(false)}
                className={`w-full py-4 rounded border font-bold flex flex-col items-center justify-center gap-1 transition-all ${triggerKessler ? 'bg-red-600 border-red-500 text-white scale-95 shadow-inner' : 'bg-red-900/20 border-red-500/50 text-red-400 hover:bg-red-900/40 hover:border-red-400'}`}
            >
                <span className="flex items-center gap-2">
                    <ArrowsPointingOutIcon className="w-5 h-5" /> 
                    TRIGGER FRAGMENTATION
                </span>
                <span className="text-[10px] font-normal opacity-70">INJECT RANDOM VELOCITY DELTAS</span>
            </button>
           </section>


          {/* Data Controls */}
          <section className="space-y-4">
            <h2 className="text-xs font-semibold text-neutral-400 uppercase tracking-wider flex items-center gap-2">
                <SignalIcon className="w-3 h-3" /> Data Uplink
            </h2>
            <div className="grid grid-cols-2 gap-2">
                <button 
                    onClick={toggleRealData}
                    disabled={isSatsLoading || !earthTextures}
                    className={`col-span-2 py-3 px-4 rounded border transition-all flex items-center justify-between group ${rawSatelliteBuffer ? 'bg-emerald-500/10 border-emerald-500 text-emerald-400' : 'bg-neutral-800 border-neutral-700 text-neutral-400 hover:border-neutral-500'}`}
                >
                    <span className="flex items-center gap-2">
                        {rawSatelliteBuffer ? <SignalIcon className="w-4 h-4 animate-pulse" /> : <CloudArrowDownIcon className="w-4 h-4" />}
                        <span className="text-xs font-bold">{rawSatelliteBuffer ? "GP DATA ACTIVE" : "INGEST TLE DATA"}</span>
                    </span>
                    {isSatsLoading && <span className="text-[10px] animate-pulse">SYNCING...</span>}
                </button>
                
                {/* Magnetic Field Toggle */}
                <button 
                    onClick={() => setShowMagField(!showMagField)}
                    className={`py-3 px-4 rounded border transition-all flex items-center justify-center gap-2 col-span-2 ${showMagField ? 'bg-purple-500/10 border-purple-500 text-purple-400' : 'bg-neutral-800 border-neutral-700 text-neutral-400 hover:border-neutral-500'}`}
                >
                    <BoltIcon className="w-4 h-4" />
                    <span className="text-xs font-bold">MAGNETOSPHERE</span>
                </button>
            </div>
          </section>

          {/* Search */}
          {rawSatelliteBuffer && (
            <section className="space-y-3 animate-in slide-in-from-left-2 fade-in">
                <h2 className="text-xs font-semibold text-neutral-400 uppercase tracking-wider flex items-center gap-2">
                    <MagnifyingGlassIcon className="w-3 h-3" /> Object Query
                </h2>
                <input 
                    type="text" 
                    placeholder="Filter by Name/ID..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full bg-neutral-900 border border-neutral-700 rounded p-2 text-xs focus:border-blue-500 outline-none transition-colors"
                />
            </section>
          )}

          {/* AI Strategy */}
          <section className="space-y-6">
            <div className="bg-gradient-to-br from-indigo-900/20 to-purple-900/20 border border-indigo-500/30 p-4 rounded-lg relative overflow-hidden">
                <div className="absolute top-0 right-0 p-2 opacity-20"><CpuChipIcon className="w-12 h-12" /></div>
                <h2 className="text-xs font-semibold text-indigo-300 uppercase tracking-wider mb-2 flex items-center gap-2">
                <SparklesIcon className="w-3 h-3" /> Gemini 2.5 Analysis
                </h2>
                {!aiAnalysis ? (
                <button onClick={handleAnalyze} disabled={isAnalyzing || !!simError} className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-xs font-bold transition-all disabled:opacity-50">
                    {isAnalyzing ? "PROCESSING..." : "ASSESS CONSTELLATION RISK"}
                </button>
                ) : (
                <div className="space-y-3 animate-in fade-in duration-500">
                    <p className="text-xs text-neutral-200">{aiAnalysis.analysis}</p>
                    <div className="border-l-2 border-orange-500 pl-3">
                        <p className="text-[10px] text-orange-400 uppercase">Action</p>
                        <p className="text-xs text-neutral-200">{aiAnalysis.recommendation}</p>
                    </div>
                </div>
                )}
            </div>
          </section>
        </div>
      </aside>

      <main 
        className="flex-1 relative bg-black cursor-crosshair overflow-hidden"
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onMouseMove={handleMouseMove}
        onWheel={handleWheel}
      >
         <WebGPUSimulation 
           gravity={1.0} 
           attractorStrength={triggerKessler ? -1.0 : 0.0}
           onStatsUpdate={setStats}
           runSimulation={isRunning}
           camera={camera}
           onError={setSimError}
           earthTextures={earthTextures}
           satelliteData={filteredBuffer} 
           magneticData={showMagField ? magData : null}
           onHover={handleHover}
           timeScale={timeScale}
         />
         
         {isEarthLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black z-50 pointer-events-none">
                <div className="text-center">
                    <GlobeAltIcon className="w-12 h-12 text-blue-500 animate-pulse mx-auto mb-4" />
                    <h2 className="text-xl font-bold text-blue-400">INITIALIZING PHYSICS ENGINE</h2>
                    <p className="text-xs text-neutral-500 mt-2">Loading High-Res Textures (NASA Blue Marble)...</p>
                </div>
            </div>
         )}
         
         {simError && (
             <div className="absolute bottom-6 left-6 right-6 bg-red-900/80 border border-red-500 p-4 rounded text-red-100 flex items-center justify-center pointer-events-none">
                 <p className="font-bold">{simError}</p>
             </div>
         )}

         {!isEarthLoading && !simError && (
          <div className="absolute top-6 right-6 pointer-events-none text-right select-none">
              <h1 className="text-4xl font-black text-neutral-800 tracking-tighter">ORBITAL VIEW</h1>
              <p className="text-neutral-600 text-xs">PITCH: {camera.pitch.toFixed(2)} | YAW: {camera.yaw.toFixed(2)} | ZOOM: {camera.zoom.toFixed(0)}</p>
          </div>
         )}

         {/* Scientific Dashboard Tooltip */}
         {hoveredData && (
             <div className="absolute top-6 left-6 pointer-events-none animate-in fade-in slide-in-from-left-4 duration-300 w-80">
                 <div className="bg-neutral-900/95 border border-blue-500/50 p-4 rounded-lg shadow-2xl backdrop-blur-md relative overflow-hidden">
                     <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-transparent" />
                     <div className="flex items-start justify-between mb-4">
                         <h3 className="text-blue-400 font-bold text-lg truncate pr-4">{hoveredData.name}</h3>
                         <span className="text-[10px] px-1.5 py-0.5 rounded border border-neutral-600 text-neutral-400">{hoveredData.type}</span>
                     </div>
                     
                     {/* Identity */}
                     <div className="grid grid-cols-2 gap-2 text-[10px] text-neutral-500 font-mono mb-4 border-b border-neutral-800 pb-4">
                         <div>NORAD ID: <span className="text-neutral-300">{hoveredData.noradId}</span></div>
                         <div>INTL: <span className="text-neutral-300">{hoveredData.intDesignator}</span></div>
                     </div>

                     {/* Orbital Elements Dashboard */}
                     {orbitalElements ? (
                        <div className="space-y-3">
                             <h4 className="text-xs font-bold text-neutral-400 flex items-center gap-1">
                                <BeakerIcon className="w-3 h-3" /> KEPLERIAN ELEMENTS
                             </h4>
                             <div className="grid grid-cols-2 gap-y-3 gap-x-2 text-xs font-mono">
                                <div className="bg-neutral-800/50 p-1.5 rounded">
                                    <div className="text-[9px] text-neutral-500 uppercase">Semi-Major Axis (a)</div>
                                    <div className="text-blue-200">{orbitalElements.semiMajorAxis.toFixed(2)} km</div>
                                </div>
                                <div className="bg-neutral-800/50 p-1.5 rounded">
                                    <div className="text-[9px] text-neutral-500 uppercase">Eccentricity (e)</div>
                                    <div className="text-blue-200">{orbitalElements.eccentricity.toFixed(5)}</div>
                                </div>
                                <div className="bg-neutral-800/50 p-1.5 rounded">
                                    <div className="text-[9px] text-neutral-500 uppercase">Inclination (i)</div>
                                    <div className="text-blue-200">{orbitalElements.inclination.toFixed(2)}°</div>
                                </div>
                                <div className="bg-neutral-800/50 p-1.5 rounded">
                                    <div className="text-[9px] text-neutral-500 uppercase">Period (P)</div>
                                    <div className="text-blue-200">{orbitalElements.period.toFixed(1)} min</div>
                                </div>
                                <div className="bg-neutral-800/50 p-1.5 rounded">
                                    <div className="text-[9px] text-neutral-500 uppercase">Altitude (h)</div>
                                    <div className="text-orange-200">{orbitalElements.altitude.toFixed(1)} km</div>
                                </div>
                                <div className="bg-neutral-800/50 p-1.5 rounded">
                                    <div className="text-[9px] text-neutral-500 uppercase">Velocity (v)</div>
                                    <div className="text-orange-200">{orbitalElements.speed.toFixed(3)} km/s</div>
                                </div>
                             </div>
                        </div>
                     ) : (
                        <div className="text-xs text-neutral-500 italic py-2">
                            {hoveredData.type === 'UNKNOWN' ? 'Terrain & Surface Data: Active' : 'Calculating State Vectors...'}
                        </div>
                     )}
                 </div>
             </div>
         )}
      </main>
    </div>
  );
}

export default App;