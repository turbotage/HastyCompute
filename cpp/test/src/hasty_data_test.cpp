#include <configure_file_settings.hpp>

#include <nlohmann/json.hpp>

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_io_mod;
import hasty_io_mod_blosc;
import hasty_generic_value_mod;
import hasty_viz_mod;

void load_hdf5_test()
{
    auto img_path = std::string(HASTY_DATA_DIR) + "/imgs/images_1.h5";
    std::cout << "Loading image from " << img_path << std::endl;

    hasty::GenericValue gv = hasty::io::hdf5::read_generic_value_entry(img_path, "astronaut_luma_512x512", false);

    hasty::Tensor t = gv.as_tensor().contiguous();

    hasty::viz::default_heatmap(hasty::viz::DefaultHeatmapOptions<1,1>{
        .z = {{t.spanning_view()}},
        .titles = {{"Astronaut Luma 512x512"}}
    }).show();
}

void test_hdf5_blosc()
{
    using namespace hasty;
    using namespace hasty::io;

    const auto img_path  = std::string(HASTY_DATA_DIR) + "/imgs/images_1.h5";
    const auto blosc_dir = std::string(HASTY_DATA_DIR) + "/imgs";

    // ── 1. Read from HDF5 ─────────────────────────────────────────────────────
    std::cout << "test_hdf5_blosc: reading '" << img_path << "'\n";
    GenericValue gv_ref = hdf5::read_generic_value(img_path, /*ignore_nongv=*/false);

    // Recursive equality check: returns "" on success, error description otherwise
    std::function<std::string(const GenericValue&, const GenericValue&, const std::string&)>
    check_equal = [&](const GenericValue& ref, const GenericValue& got,
                      const std::string& path) -> std::string
    {
        if (ref.type() != got.type())
            return path + ": type mismatch\n";

        switch (ref.type()) {
        case GenericValue::eType::NONE:
            return "";

        case GenericValue::eType::TENSOR: {
            const auto& rt = ref.as_tensor();
            const auto& gt = got.as_tensor();
            if (rt.scalar_type() != gt.scalar_type())
                return path + ": dtype mismatch\n";
            if (rt.sizes() != gt.sizes())
                return path + ": shape mismatch\n";
            double err = rt.equal(gt);
            if (!rt.equal(gt))
                return path + ": tensor.equal failed\n";
            return "";
        }

        case GenericValue::eType::STRING:
            if (ref.as_string() != got.as_string())
                return path + ": string mismatch\n";
            return "";

        case GenericValue::eType::VECTOR:
        case GenericValue::eType::TUPLE: {
            const auto& rv = ref.as_vector();
            const auto& gv = got.as_vector();
            if (rv.size() != gv.size())
                return path + ": vector size mismatch\n";
            std::string errs;
            for (std::size_t i = 0; i < rv.size(); ++i)
                errs += check_equal(rv[i], gv[i], path + "[" + std::to_string(i) + "]");
            return errs;
        }

        case GenericValue::eType::DICT: {
            const auto& rd = ref.as_dict();
            const auto& gd = got.as_dict();
            if (rd.size() != gd.size())
                return path + ": dict size mismatch\n";
            std::string errs;
            for (const auto& [key, val] : rd) {
                auto it = gd.find(key);
                if (it == gd.end()) { errs += path + "['" + key + "']: missing in got\n"; continue; }
                errs += check_equal(val, it->second, path + "['" + key + "']");
            }
            return errs;
        }
        }
        return "";
    };

    // ── 2. Compression matrix ─────────────────────────────────────────────────
    struct ComboCase {
        blosc::BloscCompressionStrategy strategy;
        const char* name;
        int level;
    };

    const std::array cases{
        ComboCase{ blosc::BloscCompressionStrategy::LZ4,    "LZ4",    1 },
        ComboCase{ blosc::BloscCompressionStrategy::LZ4,    "LZ4",    5 },
        ComboCase{ blosc::BloscCompressionStrategy::LZ4,    "LZ4",    9 },
        ComboCase{ blosc::BloscCompressionStrategy::LZ4_HC, "LZ4_HC", 1 },
        ComboCase{ blosc::BloscCompressionStrategy::LZ4_HC, "LZ4_HC", 5 },
        ComboCase{ blosc::BloscCompressionStrategy::LZ4_HC, "LZ4_HC", 7 },
        ComboCase{ blosc::BloscCompressionStrategy::LZ4_HC, "LZ4_HC", 9 },
        ComboCase{ blosc::BloscCompressionStrategy::ZSTD,   "ZSTD",   1 },
        ComboCase{ blosc::BloscCompressionStrategy::ZSTD,   "ZSTD",   5 },
        ComboCase{ blosc::BloscCompressionStrategy::ZSTD,   "ZSTD",   7 },
        ComboCase{ blosc::BloscCompressionStrategy::ZSTD,   "ZSTD",   9 },
    };

    bool all_passed = true;

    for (const auto& c : cases) {
        std::string tag = std::string(c.name) + "_L" + std::to_string(c.level);
        std::string blosc_path = blosc_dir + "/images_gv_" + tag + ".b2";

        // Write
        std::cout << "\n[test] Writing " << tag << " -> " << blosc_path << "\n" << std::flush;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        blosc::write_generic_value(gv_ref, blosc_path, c.level, c.strategy);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "[test] Write complete for " << tag << " ("
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms)\n" << std::flush;

        // Read back
        std::cout << "[test] Reading back " << blosc_path << "\n" << std::flush;
        start = std::chrono::steady_clock::now();
        GenericValue gv_back = blosc::read_generic_value(blosc_path);
        end = std::chrono::steady_clock::now();
        std::cout << "[test] Read complete for " << tag << " ("
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms)\n" << std::flush;

        // Compare
        std::cout << "[test] Comparing " << tag << "...\n" << std::flush;
        std::string errors = check_equal(gv_ref, gv_back, "root");

        if (errors.empty()) {
            std::cout << "  [PASS] " << tag << "\n" << std::flush;
        } else {
            std::cout << "  [FAIL] " << tag << ":\n" << errors << std::flush;
            all_passed = false;
        }
    }

    std::cout << "test_hdf5_blosc: " << (all_passed ? "ALL PASSED" : "SOME FAILED") << "\n";
}

void compress_hasty_data() {
using namespace hasty;
    using namespace hasty::io;

    {
        const auto inpath = std::string(HASTY_DATA_DIR) + "/rawdata/dataset.h5";
    
        std::cout << "Parsing hdf5 file '" << inpath << "' as GenericValue...\n";
        auto rettup = hdf5::parse_hdf5_as_generic_value(inpath, true);
    
        GenericValue gv = std::move(std::get<0>(rettup));
        nlohmann::json j = std::move(std::get<1>(rettup));
        std::vector<std::string> unparseable = std::move(std::get<2>(rettup));
    
        std::cout << "json: " << j.dump(2) << std::endl;
    
        for (const auto& s : unparseable) {
            std::cout << "unparseable: " << s << std::endl;
        }
    
        const auto outpath = std::string(HASTY_DATA_DIR) + "/rawdata/dataset.blosc";
    
        std::cout << "Writing GenericValue to '" << outpath << "' with ZSTD compression...\n";
        blosc::write_generic_value(gv, outpath, 9, blosc::BloscCompressionStrategy::ZSTD);
    }

    {
        const auto inpath = std::string(HASTY_DATA_DIR) + "/rawdata/image_320.h5";
    
        std::cout << "Parsing hdf5 file '" << inpath << "' as GenericValue...\n";
        auto rettup = hdf5::parse_hdf5_as_generic_value(inpath, true);
    
        GenericValue gv = std::move(std::get<0>(rettup));
        nlohmann::json j = std::move(std::get<1>(rettup));
        std::vector<std::string> unparseable = std::move(std::get<2>(rettup));
    
        std::cout << "json: " << j.dump(2) << std::endl;
    
        for (const auto& s : unparseable) {
            std::cout << "unparseable: " << s << std::endl;
        }
    
        const auto outpath = std::string(HASTY_DATA_DIR) + "/rawdata/image_320.blosc";
    
        std::cout << "Writing GenericValue to '" << outpath << "' with ZSTD compression...\n";
        blosc::write_generic_value(gv, outpath, 9, blosc::BloscCompressionStrategy::ZSTD);
    }

    {
        const auto inpath = std::string(HASTY_DATA_DIR) + "/rawdata/MRI_Raw.h5";
    
        std::cout << "Parsing hdf5 file '" << inpath << "' as GenericValue...\n";
        auto rettup = hdf5::parse_hdf5_as_generic_value(inpath, true);
    
        GenericValue gv = std::move(std::get<0>(rettup));
        nlohmann::json j = std::move(std::get<1>(rettup));
        std::vector<std::string> unparseable = std::move(std::get<2>(rettup));
    
        std::cout << "json: " << j.dump(2) << std::endl;
    
        for (const auto& s : unparseable) {
            std::cout << "unparseable: " << s << std::endl;
        }
    
        const auto outpath = std::string(HASTY_DATA_DIR) + "/rawdata/MRI_Raw.blosc";
    
        std::cout << "Writing GenericValue to '" << outpath << "' with ZSTD compression...\n";
        blosc::write_generic_value(gv, outpath, 9, blosc::BloscCompressionStrategy::ZSTD);
    }

    {
        const auto inpath = std::string(HASTY_DATA_DIR) + "/rawdata/smaps_320.h5";
    
        std::cout << "Parsing hdf5 file '" << inpath << "' as GenericValue...\n";
        auto rettup = hdf5::parse_hdf5_as_generic_value(inpath, true);
    
        GenericValue gv = std::move(std::get<0>(rettup));
        nlohmann::json j = std::move(std::get<1>(rettup));
        std::vector<std::string> unparseable = std::move(std::get<2>(rettup));
    
        std::cout << "json: " << j.dump(2) << std::endl;
    
        for (const auto& s : unparseable) {
            std::cout << "unparseable: " << s << std::endl;
        }
    
        const auto outpath = std::string(HASTY_DATA_DIR) + "/rawdata/smaps_320.blosc";
    
        std::cout << "Writing GenericValue to '" << outpath << "' with ZSTD compression...\n";
        blosc::write_generic_value(gv, outpath, 9, blosc::BloscCompressionStrategy::ZSTD);
    }

    {
        const auto inpath = std::string(HASTY_DATA_DIR) + "/rawdata/smaps_320.h5";
    
        std::cout << "Parsing hdf5 file '" << inpath << "' as GenericValue...\n";
        auto rettup = hdf5::parse_hdf5_as_generic_value(inpath, true);
    
        GenericValue gv = std::move(std::get<0>(rettup));
        nlohmann::json j = std::move(std::get<1>(rettup));
        std::vector<std::string> unparseable = std::move(std::get<2>(rettup));
    
        std::cout << "json: " << j.dump(2) << std::endl;
    
        for (const auto& s : unparseable) {
            std::cout << "unparseable: " << s << std::endl;
        }
    
        const auto outpath = std::string(HASTY_DATA_DIR) + "/rawdata/smaps_320.blosc";
    
        std::cout << "Writing GenericValue to '" << outpath << "' with ZSTD compression...\n";
        blosc::write_generic_value(gv, outpath, 9, blosc::BloscCompressionStrategy::ZSTD);
    }

    {
        const auto inpath = std::string(HASTY_DATA_DIR) + "/rawdata/smaps_true.h5";
    
        std::cout << "Parsing hdf5 file '" << inpath << "' as GenericValue...\n";
        auto rettup = hdf5::parse_hdf5_as_generic_value(inpath, true);
    
        GenericValue gv = std::move(std::get<0>(rettup));
        nlohmann::json j = std::move(std::get<1>(rettup));
        std::vector<std::string> unparseable = std::move(std::get<2>(rettup));
    
        std::cout << "json: " << j.dump(2) << std::endl;
    
        for (const auto& s : unparseable) {
            std::cout << "unparseable: " << s << std::endl;
        }
    
        const auto outpath = std::string(HASTY_DATA_DIR) + "/rawdata/smaps_true.blosc";
    
        std::cout << "Writing GenericValue to '" << outpath << "' with ZSTD compression...\n";
        blosc::write_generic_value(gv, outpath, 9, blosc::BloscCompressionStrategy::ZSTD);
    }
}

int main() {
    

    return 0;
}