#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>
#include <mpi.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

const int TAG_META = 1;
const int TAG_NAME = 2;
const int TAG_DATA = 3;

// --- OPENMP MATHEMATICAL FILTERS ---
std::vector<unsigned char> apply_blur(const unsigned char* input, int w, int h) {
    std::vector<unsigned char> output(w * h, 0);
    #pragma omp parallel for
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
    #pragma omp parallel for
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int gx = -1 * input[(y-1)*w + (x-1)] + 1 * input[(y-1)*w + (x+1)]
                     -2 * input[y*w + (x-1)]     + 2 * input[y*w + (x+1)]
                     -1 * input[(y+1)*w + (x-1)] + 1 * input[(y+1)*w + (x+1)];
            int gy = -1 * input[(y-1)*w + (x-1)] - 2 * input[(y-1)*w + x] - 1 * input[(y-1)*w + (x+1)]
                     +1 * input[(y+1)*w + (x-1)] + 2 * input[(y+1)*w + x] + 1 * input[(y+1)*w + (x+1)];
            int mag = sqrt(gx * gx + gy * gy);
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

// --- MAIN ASYNC PIPELINE ---
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size != 3) {
        if (world_rank == 0) std::cerr << "Requires exactly 3 MPI Ranks." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double start_time = MPI_Wtime();

    // ==========================================
    // STAGE 1: RANK 0 (LOAD & BLUR - ASYNC SEND)
    // ==========================================
    if (world_rank == 0) {
        MPI_Request send_reqs[3] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        std::vector<unsigned char> async_buffer; // Double buffer concept: hold data until network is done
        std::string async_name;
        int async_meta[3];

        for (const auto& entry : fs::directory_iterator("input_images")) {
            if (entry.path().extension() == ".jpg") {
                std::string filename = entry.path().filename().string();
                int w, h, c;
                unsigned char* image = stbi_load(entry.path().c_str(), &w, &h, &c, 1);
                
                if (image) {
                    // 1. Wait ONLY if the PREVIOUS background send is still flying over the network
                    MPI_Waitall(3, send_reqs, MPI_STATUSES_IGNORE);

                    // 2. Compute the current image
                    double t1 = MPI_Wtime();
                    async_buffer = apply_blur(image, w, h);
                    double t2 = MPI_Wtime();
                    std::cout << "[Rank 0] Blurred " << filename << " in " << (t2-t1) << "s" << std::endl;

                    // 3. Prepare metadata
                    async_meta[0] = w; async_meta[1] = h; async_meta[2] = filename.length();
                    async_name = filename;

                    // 4. NON-BLOCKING SENDS (MPI_Isend). The CPU instantly moves to the next file!
                    MPI_Isend(async_meta, 3, MPI_INT, 1, TAG_META, MPI_COMM_WORLD, &send_reqs[0]);
                    MPI_Isend(async_name.c_str(), async_meta[2], MPI_CHAR, 1, TAG_NAME, MPI_COMM_WORLD, &send_reqs[1]);
                    MPI_Isend(async_buffer.data(), w * h, MPI_UNSIGNED_CHAR, 1, TAG_DATA, MPI_COMM_WORLD, &send_reqs[2]);
                    
                    stbi_image_free(image);
                }
            }
        }
        // Wait for the final image to finish sending before sending the kill signal
        MPI_Waitall(3, send_reqs, MPI_STATUSES_IGNORE);
        int kill_meta[3] = {-1, -1, -1};
        MPI_Send(kill_meta, 3, MPI_INT, 1, TAG_META, MPI_COMM_WORLD);
    }

    // ==========================================
    // STAGE 2: RANK 1 (SOBEL - ASYNC RECV & SEND)
    // ==========================================
    else if (world_rank == 1) {
        MPI_Request send_reqs[3] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        std::vector<unsigned char> out_buffer;
        std::string out_name;
        int out_meta[3];

        while (true) {
            int meta[3];
            MPI_Recv(meta, 3, MPI_INT, 0, TAG_META, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (meta[0] == -1) {
                MPI_Waitall(3, send_reqs, MPI_STATUSES_IGNORE); // Ensure last item sent
                MPI_Send(meta, 3, MPI_INT, 2, TAG_META, MPI_COMM_WORLD);
                break;
            }

            int w = meta[0], h = meta[1], name_len = meta[2];
            std::vector<char> name_buf(name_len + 1, '\0');
            MPI_Recv(name_buf.data(), name_len, MPI_CHAR, 0, TAG_NAME, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::string filename(name_buf.data());

            std::vector<unsigned char> blurred_img(w * h);
            MPI_Recv(blurred_img.data(), w * h, MPI_UNSIGNED_CHAR, 0, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Compute Sobel
            double t1 = MPI_Wtime();
            std::vector<unsigned char> edges = apply_sobel(blurred_img, w, h);
            double t2 = MPI_Wtime();
            std::cout << "[Rank 1] Sobel applied to " << filename << " in " << (t2-t1) << "s" << std::endl;

            // Wait for previous background send to finish before overwriting the out_buffer
            MPI_Waitall(3, send_reqs, MPI_STATUSES_IGNORE);
            
            // Move computed data into double buffer variables
            out_buffer = std::move(edges);
            out_name = filename;
            out_meta[0] = w; out_meta[1] = h; out_meta[2] = name_len;

            // NON-BLOCKING SENDS to Rank 2
            MPI_Isend(out_meta, 3, MPI_INT, 2, TAG_META, MPI_COMM_WORLD, &send_reqs[0]);
            MPI_Isend(out_name.c_str(), out_meta[2], MPI_CHAR, 2, TAG_NAME, MPI_COMM_WORLD, &send_reqs[1]);
            MPI_Isend(out_buffer.data(), w * h, MPI_UNSIGNED_CHAR, 2, TAG_DATA, MPI_COMM_WORLD, &send_reqs[2]);
        }
    }

    // ==========================================
    // STAGE 3: RANK 2 (THRESHOLD & SAVE)
    // ==========================================
    else if (world_rank == 2) {
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
            std::cout << "[Rank 2] Threshold applied to " << filename << " in " << (t2-t1) << "s. Saving..." << std::endl;

            std::string out_path = "output_images/final_" + filename;
            stbi_write_jpg(out_path.c_str(), w, h, 1, final_img.data(), 100);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        std::cout << "\n>>> Async Assembly Line Complete! Total Time: " << (MPI_Wtime() - start_time) << " seconds. <<<" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
