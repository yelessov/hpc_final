#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    @autoreleasepool {
        // 1. Инициализация видеокарты Apple (M2)
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal is not supported on this device." << std::endl;
            return -1;
        }
        std::cout << "Using GPU: " << [[device name] UTF8String] << std::endl;

        // 2. Загрузка картинки
        int width, height, channels;
        std::string input_path = "input_images/img4.jpg"; // Возьмем самую большую картинку
        unsigned char* image_data = stbi_load(input_path.c_str(), &width, &height, &channels, 1);
        
        if(!image_data) {
            std::cerr << "Failed to load image." << std::endl;
            return -1;
        }

        // 3. Подготовка текстур для GPU
        MTLTextureDescriptor* textureDesc = [[MTLTextureDescriptor alloc] init];
        textureDesc.pixelFormat = MTLPixelFormatR8Unorm; // 8-bit grayscale
        textureDesc.width = width;
        textureDesc.height = height;
        textureDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

        id<MTLTexture> inTexture = [device newTextureWithDescriptor:textureDesc];
        id<MTLTexture> outTexture = [device newTextureWithDescriptor:textureDesc];

        MTLRegion region = MTLRegionMake2D(0, 0, width, height);
        [inTexture replaceRegion:region mipmapLevel:0 withBytes:image_data bytesPerRow:width];

        // 4. Компиляция шейдера
        NSError* error = nil;
        NSString* shaderSource = [NSString stringWithContentsOfFile:@"sobel.metal" encoding:NSUTF8StringEncoding error:&error];
        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        id<MTLFunction> kernelFunc = [library newFunctionWithName:@"vision_pipeline_fused"];
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunc error:&error];

        // 5. Запуск на видеокарте
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipelineState];
        [encoder setTexture:inTexture atIndex:0];
        [encoder setTexture:outTexture atIndex:1];

        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        MTLSize threadGroups = MTLSizeMake((width + 15) / 16, (height + 15) / 16, 1);
        [encoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        NSDate* startTime = [NSDate date];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted]; // Ждем, пока GPU закончит работу
        NSTimeInterval executionTime = [[NSDate date] timeIntervalSinceDate:startTime];

        std::cout << "[GPU Compute] Processed 4000x6000 image in: " << executionTime << " seconds." << std::endl;

        // 6. Скачиваем результат с GPU и сохраняем
        std::vector<unsigned char> result_data(width * height);
        [outTexture getBytes:result_data.data() bytesPerRow:width fromRegion:region mipmapLevel:0];
        stbi_write_jpg("output_images/gpu_final.jpg", width, height, 1, result_data.data(), 100);

        stbi_image_free(image_data);
    }
    return 0;
}