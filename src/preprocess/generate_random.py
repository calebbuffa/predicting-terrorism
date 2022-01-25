import arcpy
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("workspace", type=str)
parser.add_argument("count", type=str)
parser.add_argument("distance", type=str)
args = parser.parse_args()


def main(workspace: str, count: str, distance: str):
    with arcpy.EnvManager(workspace=workspace):
        extent = "-4077201.58048806 -247892.603949694 3062429.90123652 4981239.14419199"

        print(f"Creating {count} Random points...")
        random = arcpy.management.CreateRandomPoints(
            workspace,
            f"Random_{count}",
            "Europe_DissolveBoundaries",
            extent,
            number_of_points_or_field=count,
            minimum_allowed_distance=distance,
        )
        print("Clipping tesselations...")
        clipped = arcpy.analysis.Clip(
            random, "Europe_DissolveBoundaries", f"Random_{count}_clipped"
        )
        print("Creating centroids...")
        centroids = arcpy.management.FeatureToPoint(
            clipped, f"Random_{count}_centroids", "CENTROID"
        )
        print("Buffering gtd points...")
        buffer = arcpy.analysis.PairwiseBuffer(
            "gtd_original",
            "buffer",
            "2.5 Kilometers",
            "ALL",
            None,
            "PLANAR",
            "0 DecimalDegrees",
        )
        print("Erasing centroids in gtd buffer...")
        arcpy.analysis.PairwiseErase(
            f"Random_{count}_centroids",
            buffer,
            f"Random_{count}_centroids_erased",
            None,
        )
        print("Merging data...")
        merged = arcpy.management.Merge(
            f"Random_{count}_centroids_erased;gtd_original", f"Random_{count}_data"
        )
        print("Adding xy coordinates...")
        data = arcpy.management.AddXY(f"Random_{count}_data")
        print("Extracting pixel values...")
        data = arcpy.sa.ExtractMultiValuesToPoints(
            data,
            r"Rasters\ppp_2018_1km_Aggregated.tif pop__density;Rasters\acled_raster civil_unrest;Rasters\GDP_grid_flt.tif gdp;Rasters\europe_nighttime_lights nighttime_lights;Rasters\f_15_49_2015.tif f_15_49_50;Rasters\elevation.tif elevation;Rasters\EstISA_final.tif impervious_land;Rasters\distance_major_waterway.tif dist_maj_waterway;Rasters\population_density_2018.tif pop_density;Rasters\landcover.tif landcover;Rasters\slope.tif slope;Rasters\built_settlement_growth_2018.tif built_settlement_growth;Rasters\distance_inland_water.tif dist_inland_water;Rasters\distance_major_road.tif dist_maj_road;Rasters\distance_major_road_intersection.tif dist_maj_road_int",
            "NONE",
        )
        print("Exporting csv...")
        arcpy.conversion.TableToTable(
            data, os.getcwd(), f"Random_{count}_data.csv",
        )


if __name__ == "__main__":
    arcpy.env.Workspace = True

    main(args.workspace, args.count, args.distance)
