from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geotoken import load_config, print_config
from geotoken.geometry import BEVAnchorGrid, CameraProjector, LidarRasterizer, RelationGraphBuilder, SparseEdgeSampler


def parse_args():
    parser = argparse.ArgumentParser(description="Build GeoToken relation cache.")
    parser.add_argument("--config", default="configs/geotoken/geotoken_nuscenes.yaml")
    parser.add_argument("--print-config", action="store_true", help="Print resolved experiment config and exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.print_config:
        print_config(config)
        return
    grid = BEVAnchorGrid.from_config(config)
    rasterizer = LidarRasterizer.from_config(config, grid=grid)
    projector = CameraProjector.from_config(config)
    relation_builder = RelationGraphBuilder.from_config(config, grid=grid)
    edge_sampler = SparseEdgeSampler.from_config(config)
    print_config(config)
    print(
        "Step 6 sparse edge sampler ready: "
        f"anchors={grid.num_anchors}, grid_shape={grid.grid_shape}, "
        f"lidar_z_range={rasterizer.config.z_range}, image_size={projector.config.image_size}, "
        f"distance_bins={relation_builder.spec.distance_bins}, "
        f"edges_per_anchor={edge_sampler.config.edges_per_anchor}, "
        f"max_edges={grid.config.active_anchor_count * edge_sampler.config.edges_per_anchor}"
    )


if __name__ == "__main__":
    main()
