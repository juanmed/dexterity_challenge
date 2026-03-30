#include "algorithm_base.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class GreedyAlgorithm : public AlgorithmBase {
public:
    TruckInfo truck_{};

    void setup(const TruckInfo& truck) override {
        truck_ = truck;
    }

    PlacementDecision decide(
        const BoxInfo& current_box,
        const std::vector<PlacedBoxInfo>& placed_boxes,
        int boxes_remaining,
        double current_density
    ) override {
        py::gil_scoped_release release;
        // Naive: place at origin with identity orientation
        PlacementDecision d;
        d.position = {0.0, 0.0, 0.0};
        d.orientation = {1.0, 0.0, 0.0, 0.0};
        d.stop = false;
        return d;
    }

    void teardown(double final_density, const std::string& termination_reason) override {}
};

PYBIND11_MODULE(dex_algorithm, m) {
    py::class_<GreedyAlgorithm>(m, "Algorithm")
        .def(py::init<>())
        .def("setup", [](GreedyAlgorithm& self, double depth, double width, double height) {
            self.setup(TruckInfo{depth, width, height});
        })
        .def("decide", [](GreedyAlgorithm& self, py::dict box, py::list placed, int remaining, double density) -> py::dict {
            BoxInfo bi;
            bi.id = box["id"].cast<std::string>();
            auto dims = box["dimensions"].cast<std::vector<double>>();
            bi.dimensions = {dims[0], dims[1], dims[2]};
            bi.weight = box["weight"].cast<double>();

            std::vector<PlacedBoxInfo> placed_boxes;
            // (omit full parsing for brevity)

            auto result = self.decide(bi, placed_boxes, remaining, density);
            py::dict out;
            out["position"] = py::list({result.position.x, result.position.y, result.position.z});
            out["orientation_wxyz"] = py::list({result.orientation.w, result.orientation.x, result.orientation.y, result.orientation.z});
            out["stop"] = result.stop;
            return out;
        })
        .def("teardown", [](GreedyAlgorithm& self, double density, std::string reason) {
            self.teardown(density, reason);
        });
}
