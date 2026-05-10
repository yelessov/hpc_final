#include <metal_stdlib>
using namespace metal;

// FUSED KERNEL: Sobel Edge Detection + Binarization Thresholding
kernel void vision_pipeline_fused(texture2d<float, access::read> inTexture [[texture(0)]],
                                  texture2d<float, access::write> outTexture [[texture(1)]],
                                  uint2 gridParams [[thread_position_in_grid]]) 
{
    uint x = gridParams.x;
    uint y = gridParams.y;

    // Boundary check
    if (x == 0 || y == 0 || x >= inTexture.get_width() - 1 || y >= inTexture.get_height() - 1) {
        return;
    }

    // 1. FETCH DATA (Loading surrounding pixels into ultra-fast thread registers)
    float tl = inTexture.read(uint2(x-1, y-1)).r;
    float tc = inTexture.read(uint2(x,   y-1)).r;
    float tr = inTexture.read(uint2(x+1, y-1)).r;
    float cl = inTexture.read(uint2(x-1, y)).r;
    float cr = inTexture.read(uint2(x+1, y)).r;
    float bl = inTexture.read(uint2(x-1, y+1)).r;
    float bc = inTexture.read(uint2(x,   y+1)).r;
    float br = inTexture.read(uint2(x+1, y+1)).r;

    // 2. SOBEL MATH (Executing convolution directly in registers)
    float gx = -tl + tr - 2.0*cl + 2.0*cr - bl + br;
    float gy = -tl - 2.0*tc - tr + bl + 2.0*bc + br;
    float magnitude = sqrt(gx*gx + gy*gy);

    // 3. THRESHOLDING (Binarization on the fly without writing to RAM first)
    // 0.2 is roughly 50/255 in normalized float space
    float final_pixel = (magnitude > 0.2) ? 1.0 : 0.0;

    // 4. WRITE-BACK (Only write to global memory once)
    outTexture.write(float4(final_pixel, final_pixel, final_pixel, 1.0), gridParams);
}