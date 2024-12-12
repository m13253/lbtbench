#include <atomic>
#include <cblas.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <latch>
#include <libblastrampoline.h>
#include <omp.h>
#include <random>
#include <stdlib.h>
#include <string_view>

using namespace std::literals;

namespace {

struct DenormGuard final {
    explicit DenormGuard() noexcept {
        std::uint32_t mxcsr;
        asm("stmxcsr %0" : "=m"(mxcsr));
        std::uint32_t new_mxcsr = mxcsr | 0x8040;
        asm("ldmxcsr %0" : : "m"(new_mxcsr));
        saved_mxcsr = mxcsr & 0x8040;
    }
    explicit DenormGuard(const DenormGuard &) = delete;
    explicit DenormGuard(DenormGuard &&) = default;
    ~DenormGuard() noexcept {
        std::uint32_t mxcsr;
        asm("stmxcsr %0" : "=m"(mxcsr));
        mxcsr = (mxcsr & ~(std::uint32_t) 0x8040) | saved_mxcsr;
        asm("ldmxcsr %0" : : "m"(mxcsr));
    }

private:
    std::uint32_t saved_mxcsr;
};

void validate_blas() noexcept {
    DenormGuard _denorm_guard;
    static const float A[3][2] = {{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}};
    static const float B[3][2] = {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
    static const float C_ref[4] = {58.0, 139.0, 64.0, 154.0};
    float C[4] = {0.0, 0.0, 0.0, 0.0};
    lbt_set_num_threads(1);
    cblas_sgemm64_(
        /* Order = */ CblasColMajor,
        /* TransA = */ CblasNoTrans,
        /* TransB = */ CblasTrans,
        /* M = */ 2,
        /* N = */ 2,
        /* K = */ 3,
        /* alpha = */ 1.0f,
        /* A = */ &A[0][0],
        /* lda = */ 2,
        /* B = */ &B[0][0],
        /* ldb = */ 2,
        /* beta = */ 1.0f,
        /* C = */ &C[0],
        /* ldc = */ 2
    );
    if (lbt_get_num_threads() != 1) {
        std::cout << "Error: lbt_set_num_threads(1) failed\n"sv;
    }
    std::cout << std::format("BLAS validation: expect [58, 64, 139, 154], got [{:.0f}, {:.0f}, {:.0f}, {:.0f}]. "sv, C[0], C[2], C[1], C[3]);
    for (std::size_t i = 0; i != 4; i++) {
        if (std::abs(C[i] - C_ref[i]) < 0.5) {
            continue;
        }
        std::cout << "Failed!\n"sv;
        std::exit(1);
    }
    std::cout << "Passed!\n"sv;
}

void generate_random_numbers(float random_buffer[]) noexcept {
    std::default_random_engine gen{std::random_device{}()};
    std::normal_distribution d;
    for (std::size_t i = 0; i != 2 * 64 * 64; i++) {
        random_buffer[i] = d(gen);
    }
}

std::uint64_t perform_benchmark(const std::chrono::high_resolution_clock::time_point &deadline, const float A[], const float B[], float C[]) noexcept {
    DenormGuard _denorm_guard;
    std::uint64_t flop = 0;
    do {
        cblas_sgemm64_(
            /* Order = */ CblasColMajor,
            /* TransA = */ CblasNoTrans,
            /* TransB = */ CblasTrans,
            /* M = */ 64,
            /* N = */ 64,
            /* K = */ 64,
            /* alpha = */ 1.0f,
            /* A = */ A,
            /* lda = */ 64,
            /* B = */ B,
            /* ldb = */ 64,
            /* beta = */ 1.0f,
            /* C = */ C,
            /* ldc = */ 64
        );
        flop += 64 * 64 * (2 * 64 + 2);
    } while (std::chrono::high_resolution_clock::now() < deadline);
    return flop;
}

} // namespace

int main() {
    std::cout << "BLAS backend:"sv;
    for (const lbt_library_info_t *const *i = lbt_get_config()->loaded_libs; *i; i++) {
        std::cout << ' ' << i[0]->libname;
    }
    std::cout << '\n';
    auto sgemm_ptr = lbt_get_forward("sgemm_", LBT_INTERFACE_ILP64, LBT_F2C_UNKNOWN);
    std::cout << std::format("BLAS validation: sgemm_ = {}\n"sv, sgemm_ptr);
    if (!sgemm_ptr) {
        std::cout << "Error: libblastrampoline has not loaded any BLAS backend for sgemm_. Use environment variable LBT_DEFAULT_LIBS to specify one.\n"sv;
        std::exit(1);
    }
    validate_blas();

    std::atomic_uint_fast64_t flop{0};
#pragma omp parallel
    {
        alignas(64) thread_local float random_buffer[2 * 64 * 64];
        alignas(64) thread_local float C[64 * 64] = {0.0f};
        generate_random_numbers(random_buffer);
#pragma omp single
        std::cout << std::format("Running 64\303\22764\303\22764 GEMM benchmark on {} threads.\n"sv, omp_get_num_threads());
        for (;;) {
            auto begin = std::chrono::high_resolution_clock::now();
            auto deadline = begin + std::chrono::nanoseconds(10'000'000'000);
#pragma omp barrier
            flop.fetch_add(perform_benchmark(deadline, &random_buffer[0], &random_buffer[64 * 64], C), std::memory_order_relaxed);
#pragma omp barrier
#pragma omp single
            {
                std::chrono::duration<double, std::nano> duration = std::chrono::high_resolution_clock::now() - begin;
                std::cout << std::format("Total performance: {:.9f} Gflop/s\n"sv, flop.load(std::memory_order_relaxed) / duration.count());
                flop.store(0, std::memory_order_relaxed);
            }
        }
    }

    return 0;
}
