#pragma once
#include <array>
#include <string>
#include <vector>

struct Vec3 { double x, y, z; };
struct Quat { double w, x, y, z; };

struct BoxInfo {
    std::string id;
    std::array<double, 3> dimensions;  // [length, width, height]
    double weight;
};

struct PlacedBoxInfo : BoxInfo {
    Vec3 settled_position;
    Quat settled_orientation;
};

struct PlacementDecision {
    Vec3 position;
    Quat orientation;
    bool stop = false;
};

struct TruckInfo {
    double depth, width, height;
};

class AlgorithmBase {
public:
    virtual void setup(const TruckInfo& truck) = 0;
    virtual PlacementDecision decide(
        const BoxInfo& current_box,
        const std::vector<PlacedBoxInfo>& placed_boxes,
        int boxes_remaining,
        double current_density
    ) = 0;
    virtual void teardown(double final_density, const std::string& termination_reason) {}
    virtual ~AlgorithmBase() = default;
};
