#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <iomanip>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

const int TAG_META = 1;
const int TAG_NAME = 2;
const int TAG_DATA = 3;

// ============================================================================
// EXECUTION MODE TRACKING
// ============================================================================
enum ExecutionMode {
    PURE_OPENMP,
    PURE_MPI,
    HYBRID_MPI_OPENMP
};

struct ExecutionConfig {
    ExecutionMode mode;
    int world_size;
    int world_rank;
    int num_threads;
    int num_images_processed;
    double total_time;
    double blur_time;
    double sobel_time;
    double threshold_time;
    double io_time;
};

// ============================================================================
// OPENMP MATHEMATICAL FILTERS WITH TIMING
// ============================================================================

std::vector<unsigned char> apply_blur(const unsigned char* input, int w, int h) {
    std::vector<unsigned char> output(w * h, 0);
    
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int sum = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    sum += input[(y + dy) * w + (x + dx)];
                }
            }
            output[y * w + x] = sum / 9;
        }
    }
    return output;
}

std::vector<unsigned char> apply_sobel(const std::vector<unsigned char>& input, int w, int h) {
    std::vector<unsigned char> output(w * h, 0);
    
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int gx = -1 * input[(y-1)*w + (x-1)] + 1 * input[(y-1)*w + (x+1)]
                     -2 * input[y*w + (x-1)]     + 2 * input[y*w + (x+1)]
                     -1 * input[(y+1)*w + (x-1)] + 1 * input[(y+1)*w + (x+1)];
            
            int gy = -1 * input[(y-1)*w + (x-1)] - 2 * input[(y-1)*w + x] - 1 * input[(y-1)*w + (x+1)]
                     +1 * input[(y+1)*w + (x-1)] + 2 * input[(y+1)*w + x] + 1 * input[(y+1)*w + (x+1)];
            
            int mag = (int)std::sqrt((float)(gx * gx + gy * gy));
            output[y * w + x] = std::min(255, mag);
        }
    }
    return output;
}

std::vector<unsigned char> apply_threshold(const std::vector<unsigned char>& input, int w, int h) {
    std::vector<unsigned char> output(w * h, 0);
    
    #pragma omp parallel for
    for (int i = 0; i < w * h; i++) {
        output[i] = (input[i] > 50) ? 255 : 0;
    }
    return output;
}

// ============================================================================
// PURE OPENMP EXECUTION PATH
// ============================================================================

void execute_pure_openmp(ExecutionConfig& config) {
    int num_threads = omp_get_max_threads();
    config.num_threads = num_threads;
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "[EXECUTION MODE] PURE OPENMP (Single Node, " << num_threads << " Threads)" << std::endl;
    std::cout << std::string(80, '=') << std::endl << std::endl;
    
    double global_start = MPI_Wtime();
    double blur_accum = 0.0, sobel_accum = 0.0, threshold_accum = 0.0, io_accum = 0.0;
    int image_count = 0;
    
    try {
        if (!fs::exists("input_images")) {
            std::cerr << "[ERROR] Directory 'input_images' not found." << std::endl;
            return;
        }
        
        for (const auto& entry : fs::directory_iterator("input_images")) {
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".JPG") {
                std::string filename = entry.path().filename().string();
                int w, h, c;
                
                unsigned char* image = stbi_load(entry.path().c_str(), &w, &h, &c, 1);
                if (!image) {
                    std::cerr << "[WARNING] Failed to load " << filename << std::endl;
                    continue;
                }
                
                image_count++;
                std::cout << "[OPENMP] Processing image " << image_count << ": " << filename 
                          << " (" << w << "x" << h << ")" << std::endl;
                
                double t_blur_start = MPI_Wtime();
                std::vector<unsigned char> blurred = apply_blur(image, w, h);
                double t_blur_end = MPI_Wtime();
                double blur_time = t_blur_end - t_blur_start;
                blur_accum += blur_time;
                
                double t_sobel_start = MPI_Wtime();
                std::vector<unsigned char> edges = apply_sobel(blurred, w, h);
                double t_sobel_end = MPI_Wtime();
                double sobel_time = t_sobel_end - t_sobel_start;
                sobel_accum += sobel_time;
                
                double t_threshold_start = MPI_Wtime();
                std::vector<unsigned char> final_img = apply_threshold(edges, w, h);
                double t_threshold_end = MPI_Wtime();
                double threshold_time = t_threshold_end - t_threshold_start;
                threshold_accum += threshold_time;
                
                double t_io_start = MPI_Wtime();
                std::string out_path = "output_images/final_" + filename;
                stbi_write_jpg(out_path.c_str(), w, h, 1, final_img.data(), 100);
                double t_io_end = MPI_Wtime();
                double io_time = t_io_end - t_io_start;
                io_accum += io_time;
                
                double per_image_total = blur_time + sobel_time + threshold_time + io_time;
                std::cout << "  |-- Blur:      " << std::fixed << std::setprecision(4) << blur_time * 1000 << " ms" << std::endl;
                std::cout << "  |-- Sobel:     " << sobel_time * 1000 << " ms" << std::endl;
                std::cout << "  |-- Threshold: " << threshold_time * 1000 << " ms" << std::endl;
                std::cout << "  |-- I/O:       " << io_time * 1000 << " ms" << std::endl;
                std::cout << "  \\-- Total:     " << per_image_total * 1000 << " ms" << std::endl << std::endl;
                
                stbi_image_free(image);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception during execution: " << e.what() << std::endl;
        return;
    }
    
    double global_end = MPI_Wtime();
    double total_time = global_end - global_start;
    
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "[SUMMARY] Pure OpenMP Execution" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Total images processed:    " << image_count << std::endl;
    std::cout << "Total execution time:      " << std::fixed << std::setprecision(4) << total_time << " seconds" << std::endl;
    std::cout << "Average time per image:    " << (image_count > 0 ? total_time / image_count : 0) << " seconds" << std::endl;
    std::cout << "Throughput:                " << (image_count > 0 ? image_count / total_time : 0) << " images/sec" << std::endl;
    std::cout << "\nAggregate filter timings:" << std::endl;
    std::cout << "  |-- Total Blur:      " << blur_accum << " seconds (" << (blur_accum/total_time)*100 << "%)" << std::endl;
    std::cout << "  |-- Total Sobel:     " << sobel_accum << " seconds (" << (sobel_accum/total_time)*100 << "%)" << std::endl;
    std::cout << "  |-- Total Threshold: " << threshold_accum << " seconds (" << (threshold_accum/total_time)*100 << "%)" << std::endl;
    std::cout << "  \\-- Total I/O:       " << io_accum << " seconds (" << (io_accum/total_time)*100 << "%)" << std::endl;
    std::cout << std::string(80, '=') << std::endl << std::endl;
    
    config.mode = PURE_OPENMP;
    config.num_images_processed = image_count;
    config.total_time = total_time;
    config.blur_time = blur_accum;
    config.sobel_time = sobel_accum;
    config.threshold_time = threshold_accum;
    config.io_time = io_accum;
}

// ============================================================================
// HYBRID MPI + OPENMP EXECUTION PATH (3-STAGE ASYNC PIPELINE)
// ============================================================================

void execute_hybrid_mpi_openmp(ExecutionConfig& config) {
    int world_size = config.world_size;
    int world_rank = config.world_rank;
    int num_threads = omp_get_max_threads();
    
    if (world_rank == 0) {
        std::string mode_str = (num_threads > 1) ? "HYBRID (MPI + OpenMP)" : "PURE MPI";
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "[EXECUTION MODE] " << mode_str << " (" << world_size << " Ranks, " << num_threads << " Threads/Rank)" << std::endl;
        std::cout << std::string(80, '=') << std::endl << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    if (world_rank == 0) {
        MPI_Request send_reqs[3] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        std::vector<unsigned char> async_buffer;
        std::string async_name;
        int async_meta[3];
        int image_count = 0;
        double blur_accum = 0.0;
        
        try {
            if (!fs::exists("input_images")) {
                std::cerr << "[ERROR] Directory 'input_images' not found." << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            for (const auto& entry : fs::directory_iterator("input_images")) {
                if (entry.path().extension() == ".jpg" || entry.path().extension() == ".JPG") {
                    std::string filename = entry.path().filename().string();
                    int w, h, c;
                    unsigned char* image = stbi_load(entry.path().c_str(), &w, &h, &c, 1);
                    
                    if (image) {
                        MPI_Waitall(3, send_reqs, MPI_STATUSES_IGNORE);
                        
                        double t1 = MPI_Wtime();
                        async_buffer = apply_blur(image, w, h);
                        double t2 = MPI_Wtime();
                        blur_accum += (t2 - t1);
                        
                        image_count++;
                        std::cout << "[Rank 0] Blurred " << filename << " (" << w << "x" << h << ") in " 
                                  << std::fixed << std::setprecision(4) << (t2-t1)*1000 << " ms" << std::endl;
                        
                        async_meta[0] = w;
                        async_meta[1] = h;
                        async_meta[2] = filename.length();
                        async_name = filename;
                        
                        MPI_Isend(async_meta, 3, MPI_INT, 1, TAG_META, MPI_COMM_WORLD, &send_reqs[0]);
                        MPI_Isend(async_name.c_str(), async_meta[2], MPI_CHAR, 1, TAG_NAME, MPI_COMM_WORLD, &send_reqs[1]);
                        MPI_Isend(async_buffer.data(), w * h, MPI_UNSIGNED_CHAR, 1, TAG_DATA, MPI_COMM_WORLD, &send_reqs[2]);
                        
                        stbi_image_free(image);
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Rank 0 exception: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        MPI_Waitall(3, send_reqs, MPI_STATUSES_IGNORE);
        int kill_meta[3] = {-1, -1, -1};
        MPI_Send(kill_meta, 3, MPI_INT, 1, TAG_META, MPI_COMM_WORLD);
        
        config.num_images_processed = image_count;
        config.blur_time = blur_accum;
    }
    else if (world_rank == 1) {
        MPI_Request send_reqs[3] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        std::vector<unsigned char> out_buffer;
        std::string out_name;
        int out_meta[3];
        double sobel_accum = 0.0;
        
        while (true) {
            int meta[3];
            MPI_Recv(meta, 3, MPI_INT, 0, TAG_META, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            if (meta[0] == -1) {
                MPI_Waitall(3, send_reqs, MPI_STATUSES_IGNORE);
                int kill_meta[3] = {-1, -1, -1};
                MPI_Send(kill_meta, 3, MPI_INT, 2, TAG_META, MPI_COMM_WORLD);
                break;
            }
            
            int w = meta[0], h = meta[1], name_len = meta[2];
            std::vector<char> name_buf(name_len + 1, '\0');
            MPI_Recv(name_buf.data(), name_len, MPI_CHAR, 0, TAG_NAME, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::string filename(name_buf.data());
            
            std::vector<unsigned char> blurred_img(w * h);
            MPI_Recv(blurred_img.data(), w * h, MPI_UNSIGNED_CHAR, 0, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            double t1 = MPI_Wtime();
            std::vector<unsigned char> edges = apply_sobel(blurred_img, w, h);
            double t2 = MPI_Wtime();
            sobel_accum += (t2 - t1);
            
            std::cout << "[Rank 1] Sobel applied to " << filename << " in " 
                      << std::fixed << std::setprecision(4) << (t2-t1)*1000 << " ms" << std::endl;
            
            MPI_Waitall(3, send_reqs, MPI_STATUSES_IGNORE);
            
            out_buffer = std::move(edges);
            out_name = filename;
            out_meta[0] = w;
            out_meta[1] = h;
            out_meta[2] = name_len;
            
            MPI_Isend(out_meta, 3, MPI_INT, 2, TAG_META, MPI_COMM_WORLD, &send_reqs[0]);
            MPI_Isend(out_name.c_str(), out_meta[2], MPI_CHAR, 2, TAG_NAME, MPI_COMM_WORLD, &send_reqs[1]);
            MPI_Isend(out_buffer.data(), w * h, MPI_UNSIGNED_CHAR, 2, TAG_DATA, MPI_COMM_WORLD, &send_reqs[2]);
        }
        
        config.sobel_time = sobel_accum;
    }
    else if (world_rank == 2) {
        double threshold_accum = 0.0, io_accum = 0.0;
        
        while (true) {
            int meta[3];
            MPI_Recv(meta, 3, MPI_INT, 1, TAG_META, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            if (meta[0] == -1) break;
            
            int w = meta[0], h = meta[1], name_len = meta[2];
            std::vector<char> name_buf(name_len + 1, '\0');
            MPI_Recv(name_buf.data(), name_len, MPI_CHAR, 1, TAG_NAME, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::string filename(name_buf.data());
            
            std::vector<unsigned char> edges_img(w * h);
            MPI_Recv(edges_img.data(), w * h, MPI_UNSIGNED_CHAR, 1, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            double t1 = MPI_Wtime();
            std::vector<unsigned char> final_img = apply_threshold(edges_img, w, h);
            double t2 = MPI_Wtime();
            threshold_accum += (t2 - t1);
            
            double t_io_start = MPI_Wtime();
            std::string out_path = "output_images/final_" + filename;
            stbi_write_jpg(out_path.c_str(), w, h, 1, final_img.data(), 100);
            double t_io_end = MPI_Wtime();
            io_accum += (t_io_end - t_io_start);
            
            std::cout << "[Rank 2] Threshold & saved " << filename << " in " 
                      << std::fixed << std::setprecision(4) << (t2-t1)*1000 << " ms (I/O: " 
                      << (t_io_end - t_io_start)*1000 << " ms)" << std::endl;
        }
        
        config.threshold_time = threshold_accum;
        config.io_time = io_accum;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;
    
    if (world_rank == 0) {
        std::cout << std::string(80, '-') << std::endl;
        std::cout << "[SUMMARY] Hybrid MPI+OpenMP Pipeline Execution" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        std::cout << "Total images processed:    " << config.num_images_processed << std::endl;
        std::cout << "Total execution time:      " << std::fixed << std::setprecision(4) << total_time << " seconds" << std::endl;
        std::cout << "Average time per image:    " << (config.num_images_processed > 0 ? total_time / config.num_images_processed : 0) << " seconds" << std::endl;
        std::cout << "Throughput:                " << (config.num_images_processed > 0 ? config.num_images_processed / total_time : 0) << " images/sec" << std::endl;
        std::cout << "\nAggregate filter timings:" << std::endl;
        std::cout << "  |-- Total Blur:      " << config.blur_time << " seconds (" << (config.blur_time/total_time)*100 << "%)" << std::endl;
        std::cout << "  |-- Total Sobel:     " << config.sobel_time << " seconds (" << (config.sobel_time/total_time)*100 << "%)" << std::endl;
        std::cout << "  |-- Total Threshold: " << config.threshold_time << " seconds (" << (config.threshold_time/total_time)*100 << "%)" << std::endl;
        std::cout << "  \\-- Total I/O:       " << config.io_time << " seconds (" << (config.io_time/total_time)*100 << "%)" << std::endl;
        std::cout << std::string(80, '=') << std::endl << std::endl;
    }
    
    config.total_time = total_time;
    config.mode = (num_threads > 1) ? HYBRID_MPI_OPENMP : PURE_MPI;
}

// ============================================================================
// MAIN ENTRY POINT WITH MODE DETECTION
// ============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    int num_threads = omp_get_max_threads();
    
    ExecutionConfig config;
    config.world_size = world_size;
    config.world_rank = world_rank;
    config.num_threads = num_threads;
    
    ExecutionMode mode;
    if (world_size == 1) {
        mode = PURE_OPENMP;
    } else if (world_size == 3) {
        mode = (num_threads > 1) ? HYBRID_MPI_OPENMP : PURE_MPI;
    } else {
        if (world_rank == 0) {
            std::cerr << "[ERROR] Unsupported MPI configuration. Requires 1 rank (Pure OpenMP) or 3 ranks (MPI/Hybrid)." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    if (mode == PURE_OPENMP) {
        execute_pure_openmp(config);
    } else {
        execute_hybrid_mpi_openmp(config);
    }
    
    MPI_Finalize();
    return 0;
}
