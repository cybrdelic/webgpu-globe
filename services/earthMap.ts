
export interface EarthTextures {
    colorMap: ImageBitmap;
    heightMap: ImageBitmap;
    nightMap: ImageBitmap;
    overlayMap: ImageBitmap;
}

export const fetchEarthMap = async (): Promise<EarthTextures> => {
    // 1. Day: NASA Blue Marble
    const colorUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Land_ocean_ice_2048.jpg/1024px-Land_ocean_ice_2048.jpg";
    
    // 2. Height: Topography (SRTM)
    const heightUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Srtm_ramp2.world.21600x10800.jpg/1024px-Srtm_ramp2.world.21600x10800.jpg";
    
    // 3. Night: NASA Black Marble (City Lights)
    const nightUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/The_earth_at_night.jpg/1024px-The_earth_at_night.jpg";

    // 4. Overlay: Political Borders & Reference (Using a high contrast map we can filter in shader)
    // We use a specific projection that aligns reasonably well with the others.
    const overlayUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Blue_Marble_2002.png/1024px-Blue_Marble_2002.png"; 
    // Note: Finding a perfect transparent PNG with borders on a reliable public URL is hard. 
    // We will use a visual trick in the shader to extract borders from a reference map or simulate them.
    // Actually, let's use a cleaner reference map if possible, or stick to a map that implies borders.
    // For this demo, we will use a map that highlights landmass boundaries clearly.

    const PROXY = 'https://corsproxy.io/?'; 

    const loadImage = async (url: string): Promise<ImageBitmap> => {
        try {
            const response = await fetch(PROXY + encodeURIComponent(url));
            if (!response.ok) throw new Error(`Failed to fetch texture: ${url}`);
            const blob = await response.blob();
            return await createImageBitmap(blob);
        } catch (e) {
            console.warn(`Texture failed: ${url}, using fallback.`);
            // Return a 1x1 black pixel as fallback to prevent crash
            const canvas = document.createElement('canvas');
            canvas.width = 1; canvas.height = 1;
            return await createImageBitmap(canvas);
        }
    };

    try {
        const [colorMap, heightMap, nightMap, overlayMap] = await Promise.all([
            loadImage(colorUrl),
            loadImage(heightUrl),
            loadImage(nightUrl),
            loadImage(overlayUrl) // We will use this to enhance the visuals
        ]);
        return { colorMap, heightMap, nightMap, overlayMap };
    } catch (e) {
        console.error("Texture Load Failed", e);
        throw e;
    }
};
