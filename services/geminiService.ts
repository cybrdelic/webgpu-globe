import { GoogleGenAI, Type } from "@google/genai";
import { SimulationStats, AIAnalysis } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export const analyzeDebrisField = async (stats: SimulationStats): Promise<AIAnalysis> => {
  const model = "gemini-2.5-flash";
  
  const prompt = `
    You are an orbital mechanics AI expert monitoring a real-time space debris field simulation (Kessler Syndrome scenario).
    
    Current Telemetry:
    - Active Debris Fragments: ${stats.particleCount}
    - Collision Events (Last Tick): ${stats.collisionEvents}
    - System Kinetic Energy: ${stats.energyTotal.toFixed(2)} units
    - Calculated Risk Level: ${stats.riskLevel}

    Based on this data, provide a structured status report. 
    1. Analyze the stability of the orbit.
    2. Recommend immediate mitigation strategies (e.g., laser ablation, orbit raising, foam capture).
    3. Define a formal protocol name for this situation.
    
    Keep the analysis concise, technical, and sci-fi flavored but grounded in physics.
  `;

  try {
    const response = await ai.models.generateContent({
      model,
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            analysis: { type: Type.STRING, description: "Technical analysis of the stability." },
            recommendation: { type: Type.STRING, description: "Actionable steps to reduce debris." },
            mitigationProtocol: { type: Type.STRING, description: "A cool sounding protocol name like 'OMEGA-CLEANSE'." }
          },
          required: ["analysis", "recommendation", "mitigationProtocol"]
        }
      }
    });

    if (response.text) {
      return JSON.parse(response.text) as AIAnalysis;
    }
    throw new Error("Empty response");
  } catch (error) {
    console.error("AI Analysis Failed:", error);
    return {
      analysis: "Telemetry link unstable. Unable to process orbital data.",
      recommendation: "Manual override required.",
      mitigationProtocol: "ERR-CONN-RESET"
    };
  }
};
