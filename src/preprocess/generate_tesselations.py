<<<<<<< HEAD
import arcpy
import sys
import os


def main(workspace: str, shape: str, size: str):
    with arcpy.env.Manager(workspace=workspace):
        extent = "-4077201.58048806 -247892.603949694 3062429.90123652 4981239.14419199"
        projection = "PROJCS['Europe_Albers_Equal_Area_Conic',GEOGCS['GCS_European_1950',DATUM['D_European_1950',SPHEROID['International_1924',6378388.0,297.0]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],PROJECTION['Albers'],PARAMETER['False_Easting',0.0],PARAMETER['False_Northing',0.0],PARAMETER['Central_Meridian',10.0],PARAMETER['Standard_Parallel_1',43.0],PARAMETER['Standard_Parallel_2',62.0],PARAMETER['Latitude_Of_Origin',30.0],UNIT['Meter',1.0]];-14490700 -7106100 10000;-100000 10000;-100000 10000;0.001;0.001;0.001;IsHighPrecision"

        print(f"Creating {shape} tesselations of {size} km...")
        tesselations = arcpy.management.GenerateTessellation(
            f"{shape}_{size}",
            extent,
            shape.upper(),
            f"{size} SquareKilometers",
            projection,
        )
        print("Clipping tesselations...")
        clipped = arcpy.analysis.Clip(
            tesselations, "Europe_DissolveBoundaries", f"{shape}_{size}_clipped"
        )
        print("Creating centroids...")
        centroids = arcpy.management.FeatureToPoint(
            clipped, f"{shape}_{size}_centroids", "CENTROID"
        )
        print("Buffering gtd points...")
        arcpy.analysis.PairwiseBuffer(
            "gtd_europe",
            "buffer",
            "2.5 Kilometers",
            "ALL",
            None,
            "PLANAR",
            "0 DecimalDegrees",
        )
        print("Erasing centroids in gtd buffer...")
        arcpy.analysis.PairwiseErase(
            f"{shape}_{size}_centroids",
            "gtd_europe_PairwiseBuffer",
            f"{shape}_{size}_centroids_erased",
            None,
        )
        arcpy.management.Merge(
            f"{shape}_{size}_centroids_erased;gtd_europe",
            f"{shape}_{size}_data",
            f'GRID_ID "GRID_ID" true true false 12 Text 0 0,First,#,{shape}_{size}_centroids_erased,GRID_ID,0,12;ORIG_FID "ORIG_FID" true true false 4 Long 0 0,First,#,{shape}_{size}_centroids_erased,ORIG_FID,-1,-1;POINT_X "POINT_X" true true false 8 Double 0 0,First,#,{shape}_{size}_centroids_erased,POINT_X,-1,-1;POINT_Y "POINT_Y" true true false 8 Double 0 0,First,#,{shape}_{size}_centroids_erased,POINT_Y,-1,-1;Attack "Attack" true true false 2 Short 0 0,First,#,{shape}_{size}_centroids_erased,Attack,-1,-1,gtd_europe,Attack,-1,-1;eventid "eventid" true true false 8 Double 0 0,First,#,gtd_europe,eventid,-1,-1;iyear "iyear" true true false 4 Long 0 0,First,#,gtd_europe,iyear,-1,-1;imonth "imonth" true true false 4 Long 0 0,First,#,gtd_europe,imonth,-1,-1;iday "iday" true true false 4 Long 0 0,First,#,gtd_europe,iday,-1,-1;approxdate "approxdate" true true false 8000 Text 0 0,First,#,gtd_europe,approxdate,0,8000;extended "extended" true true false 4 Long 0 0,First,#,gtd_europe,extended,-1,-1;resolution "resolution" true true false 8000 Text 0 0,First,#,gtd_europe,resolution,0,8000;country "country" true true false 4 Long 0 0,First,#,gtd_europe,country,-1,-1;country_txt "country_txt" true true false 8000 Text 0 0,First,#,gtd_europe,country_txt,0,8000;region "region" true true false 4 Long 0 0,First,#,gtd_europe,region,-1,-1;region_txt "region_txt" true true false 8000 Text 0 0,First,#,gtd_europe,region_txt,0,8000;provstate "provstate" true true false 8000 Text 0 0,First,#,gtd_europe,provstate,0,8000;city "city" true true false 8000 Text 0 0,First,#,gtd_europe,city,0,8000;latitude "latitude" true true false 8 Double 0 0,First,#,gtd_europe,latitude,-1,-1;longitude "longitude" true true false 8 Double 0 0,First,#,gtd_europe,longitude,-1,-1;specificity "specificity" true true false 4 Long 0 0,First,#,gtd_europe,specificity,-1,-1;vicinity "vicinity" true true false 4 Long 0 0,First,#,gtd_europe,vicinity,-1,-1;location "location" true true false 8000 Text 0 0,First,#,gtd_europe,location,0,8000;summary "summary" true true false 8000 Text 0 0,First,#,gtd_europe,summary,0,8000;crit1 "crit1" true true false 4 Long 0 0,First,#,gtd_europe,crit1,-1,-1;crit2 "crit2" true true false 4 Long 0 0,First,#,gtd_europe,crit2,-1,-1;crit3 "crit3" true true false 4 Long 0 0,First,#,gtd_europe,crit3,-1,-1;doubtterr "doubtterr" true true false 4 Long 0 0,First,#,gtd_europe,doubtterr,-1,-1;alternative "alternative" true true false 4 Long 0 0,First,#,gtd_europe,alternative,-1,-1;alternative_txt "alternative_txt" true true false 8000 Text 0 0,First,#,gtd_europe,alternative_txt,0,8000;multiple "multiple" true true false 4 Long 0 0,First,#,gtd_europe,multiple,-1,-1;success "success" true true false 4 Long 0 0,First,#,gtd_europe,success,-1,-1;suicide "suicide" true true false 4 Long 0 0,First,#,gtd_europe,suicide,-1,-1;attacktype1 "attacktype1" true true false 4 Long 0 0,First,#,gtd_europe,attacktype1,-1,-1;attacktype1_txt "attacktype1_txt" true true false 8000 Text 0 0,First,#,gtd_europe,attacktype1_txt,0,8000;attacktype2 "attacktype2" true true false 4 Long 0 0,First,#,gtd_europe,attacktype2,-1,-1;attacktype2_txt "attacktype2_txt" true true false 8000 Text 0 0,First,#,gtd_europe,attacktype2_txt,0,8000;attacktype3 "attacktype3" true true false 8000 Text 0 0,First,#,gtd_europe,attacktype3,0,8000;attacktype3_txt "attacktype3_txt" true true false 8000 Text 0 0,First,#,gtd_europe,attacktype3_txt,0,8000;targtype1 "targtype1" true true false 4 Long 0 0,First,#,gtd_europe,targtype1,-1,-1;targtype1_txt "targtype1_txt" true true false 8000 Text 0 0,First,#,gtd_europe,targtype1_txt,0,8000;targsubtype1 "targsubtype1" true true false 4 Long 0 0,First,#,gtd_europe,targsubtype1,-1,-1;targsubtype1_txt "targsubtype1_txt" true true false 8000 Text 0 0,First,#,gtd_europe,targsubtype1_txt,0,8000;corp1 "corp1" true true false 8000 Text 0 0,First,#,gtd_europe,corp1,0,8000;target1 "target1" true true false 8000 Text 0 0,First,#,gtd_europe,target1,0,8000;natlty1 "natlty1" true true false 4 Long 0 0,First,#,gtd_europe,natlty1,-1,-1;natlty1_txt "natlty1_txt" true true false 8000 Text 0 0,First,#,gtd_europe,natlty1_txt,0,8000;targtype2 "targtype2" true true false 8000 Text 0 0,First,#,gtd_europe,targtype2,0,8000;targtype2_txt "targtype2_txt" true true false 8000 Text 0 0,First,#,gtd_europe,targtype2_txt,0,8000;targsubtype2 "targsubtype2" true true false 8000 Text 0 0,First,#,gtd_europe,targsubtype2,0,8000;targsubtype2_txt "targsubtype2_txt" true true false 8000 Text 0 0,First,#,gtd_europe,targsubtype2_txt,0,8000;corp2 "corp2" true true false 8000 Text 0 0,First,#,gtd_europe,corp2,0,8000;target2 "target2" true true false 8000 Text 0 0,First,#,gtd_europe,target2,0,8000;natlty2 "natlty2" true true false 8000 Text 0 0,First,#,gtd_europe,natlty2,0,8000;natlty2_txt "natlty2_txt" true true false 8000 Text 0 0,First,#,gtd_europe,natlty2_txt,0,8000;targtype3 "targtype3" true true false 8000 Text 0 0,First,#,gtd_europe,targtype3,0,8000;targtype3_txt "targtype3_txt" true true false 8000 Text 0 0,First,#,gtd_europe,targtype3_txt,0,8000;targsubtype3 "targsubtype3" true true false 8000 Text 0 0,First,#,gtd_europe,targsubtype3,0,8000;targsubtype3_txt "targsubtype3_txt" true true false 8000 Text 0 0,First,#,gtd_europe,targsubtype3_txt,0,8000;corp3 "corp3" true true false 8000 Text 0 0,First,#,gtd_europe,corp3,0,8000;target3 "target3" true true false 8000 Text 0 0,First,#,gtd_europe,target3,0,8000;natlty3 "natlty3" true true false 8000 Text 0 0,First,#,gtd_europe,natlty3,0,8000;natlty3_txt "natlty3_txt" true true false 8000 Text 0 0,First,#,gtd_europe,natlty3_txt,0,8000;gname "gname" true true false 8000 Text 0 0,First,#,gtd_europe,gname,0,8000;gsubname "gsubname" true true false 8000 Text 0 0,First,#,gtd_europe,gsubname,0,8000;gname2 "gname2" true true false 8000 Text 0 0,First,#,gtd_europe,gname2,0,8000;gsubname2 "gsubname2" true true false 8000 Text 0 0,First,#,gtd_europe,gsubname2,0,8000;gname3 "gname3" true true false 8000 Text 0 0,First,#,gtd_europe,gname3,0,8000;gsubname3 "gsubname3" true true false 8000 Text 0 0,First,#,gtd_europe,gsubname3,0,8000;motive "motive" true true false 8000 Text 0 0,First,#,gtd_europe,motive,0,8000;guncertain1 "guncertain1" true true false 4 Long 0 0,First,#,gtd_europe,guncertain1,-1,-1;guncertain2 "guncertain2" true true false 8000 Text 0 0,First,#,gtd_europe,guncertain2,0,8000;guncertain3 "guncertain3" true true false 8000 Text 0 0,First,#,gtd_europe,guncertain3,0,8000;individual "individual" true true false 4 Long 0 0,First,#,gtd_europe,individual,-1,-1;nperps "nperps" true true false 4 Long 0 0,First,#,gtd_europe,nperps,-1,-1;nperpcap "nperpcap" true true false 8 Double 0 0,First,#,gtd_europe,nperpcap,-1,-1;claimed "claimed" true true false 4 Long 0 0,First,#,gtd_europe,claimed,-1,-1;claimmode "claimmode" true true false 4 Long 0 0,First,#,gtd_europe,claimmode,-1,-1;claimmode_txt "claimmode_txt" true true false 8000 Text 0 0,First,#,gtd_europe,claimmode_txt,0,8000;claim2 "claim2" true true false 8000 Text 0 0,First,#,gtd_europe,claim2,0,8000;claimmode2 "claimmode2" true true false 8000 Text 0 0,First,#,gtd_europe,claimmode2,0,8000;claimmode2_txt "claimmode2_txt" true true false 8000 Text 0 0,First,#,gtd_europe,claimmode2_txt,0,8000;claim3 "claim3" true true false 8000 Text 0 0,First,#,gtd_europe,claim3,0,8000;claimmode3 "claimmode3" true true false 8000 Text 0 0,First,#,gtd_europe,claimmode3,0,8000;claimmode3_txt "claimmode3_txt" true true false 8000 Text 0 0,First,#,gtd_europe,claimmode3_txt,0,8000;compclaim "compclaim" true true false 8000 Text 0 0,First,#,gtd_europe,compclaim,0,8000;weaptype1 "weaptype1" true true false 4 Long 0 0,First,#,gtd_europe,weaptype1,-1,-1;weaptype1_txt "weaptype1_txt" true true false 8000 Text 0 0,First,#,gtd_europe,weaptype1_txt,0,8000;weapsubtype1 "weapsubtype1" true true false 4 Long 0 0,First,#,gtd_europe,weapsubtype1,-1,-1;weapsubtype1_txt "weapsubtype1_txt" true true false 8000 Text 0 0,First,#,gtd_europe,weapsubtype1_txt,0,8000;weaptype2 "weaptype2" true true false 4 Long 0 0,First,#,gtd_europe,weaptype2,-1,-1;weaptype2_txt "weaptype2_txt" true true false 8000 Text 0 0,First,#,gtd_europe,weaptype2_txt,0,8000;weapsubtype2 "weapsubtype2" true true false 4 Long 0 0,First,#,gtd_europe,weapsubtype2,-1,-1;weapsubtype2_txt "weapsubtype2_txt" true true false 8000 Text 0 0,First,#,gtd_europe,weapsubtype2_txt,0,8000;weaptype3 "weaptype3" true true false 8000 Text 0 0,First,#,gtd_europe,weaptype3,0,8000;weaptype3_txt "weaptype3_txt" true true false 8000 Text 0 0,First,#,gtd_europe,weaptype3_txt,0,8000;weapsubtype3 "weapsubtype3" true true false 8000 Text 0 0,First,#,gtd_europe,weapsubtype3,0,8000;weapsubtype3_txt "weapsubtype3_txt" true true false 8000 Text 0 0,First,#,gtd_europe,weapsubtype3_txt,0,8000;weaptype4 "weaptype4" true true false 8000 Text 0 0,First,#,gtd_europe,weaptype4,0,8000;weaptype4_txt "weaptype4_txt" true true false 8000 Text 0 0,First,#,gtd_europe,weaptype4_txt,0,8000;weapsubtype4 "weapsubtype4" true true false 8000 Text 0 0,First,#,gtd_europe,weapsubtype4,0,8000;weapsubtype4_txt "weapsubtype4_txt" true true false 8000 Text 0 0,First,#,gtd_europe,weapsubtype4_txt,0,8000;weapdetail "weapdetail" true true false 8000 Text 0 0,First,#,gtd_europe,weapdetail,0,8000;nkill "nkill" true true false 4 Long 0 0,First,#,gtd_europe,nkill,-1,-1;nkillus "nkillus" true true false 4 Long 0 0,First,#,gtd_europe,nkillus,-1,-1;nkillter "nkillter" true true false 4 Long 0 0,First,#,gtd_europe,nkillter,-1,-1;nwound "nwound" true true false 8 Double 0 0,First,#,gtd_europe,nwound,-1,-1;nwoundus "nwoundus" true true false 4 Long 0 0,First,#,gtd_europe,nwoundus,-1,-1;nwoundte "nwoundte" true true false 4 Long 0 0,First,#,gtd_europe,nwoundte,-1,-1;property "property" true true false 4 Long 0 0,First,#,gtd_europe,property,-1,-1;propextent "propextent" true true false 4 Long 0 0,First,#,gtd_europe,propextent,-1,-1;propextent_txt "propextent_txt" true true false 8000 Text 0 0,First,#,gtd_europe,propextent_txt,0,8000;propvalue "propvalue" true true false 8 Double 0 0,First,#,gtd_europe,propvalue,-1,-1;propcomment "propcomment" true true false 8000 Text 0 0,First,#,gtd_europe,propcomment,0,8000;ishostkid "ishostkid" true true false 4 Long 0 0,First,#,gtd_europe,ishostkid,-1,-1;nhostkid "nhostkid" true true false 4 Long 0 0,First,#,gtd_europe,nhostkid,-1,-1;nhostkidus "nhostkidus" true true false 4 Long 0 0,First,#,gtd_europe,nhostkidus,-1,-1;nhours "nhours" true true false 8000 Text 0 0,First,#,gtd_europe,nhours,0,8000;ndays "ndays" true true false 8000 Text 0 0,First,#,gtd_europe,ndays,0,8000;divert "divert" true true false 8000 Text 0 0,First,#,gtd_europe,divert,0,8000;kidhijcountry "kidhijcountry" true true false 8000 Text 0 0,First,#,gtd_europe,kidhijcountry,0,8000;ransom "ransom" true true false 4 Long 0 0,First,#,gtd_europe,ransom,-1,-1;ransomamt "ransomamt" true true false 8 Double 0 0,First,#,gtd_europe,ransomamt,-1,-1;ransomamtus "ransomamtus" true true false 8000 Text 0 0,First,#,gtd_europe,ransomamtus,0,8000;ransompaid "ransompaid" true true false 8 Double 0 0,First,#,gtd_europe,ransompaid,-1,-1;ransompaidus "ransompaidus" true true false 8000 Text 0 0,First,#,gtd_europe,ransompaidus,0,8000;ransomnote "ransomnote" true true false 8000 Text 0 0,First,#,gtd_europe,ransomnote,0,8000;hostkidoutcome "hostkidoutcome" true true false 4 Long 0 0,First,#,gtd_europe,hostkidoutcome,-1,-1;hostkidoutcome_txt "hostkidoutcome_txt" true true false 8000 Text 0 0,First,#,gtd_europe,hostkidoutcome_txt,0,8000;nreleased "nreleased" true true false 4 Long 0 0,First,#,gtd_europe,nreleased,-1,-1;addnotes "addnotes" true true false 8000 Text 0 0,First,#,gtd_europe,addnotes,0,8000;scite1 "scite1" true true false 8000 Text 0 0,First,#,gtd_europe,scite1,0,8000;scite2 "scite2" true true false 8000 Text 0 0,First,#,gtd_europe,scite2,0,8000;scite3 "scite3" true true false 8000 Text 0 0,First,#,gtd_europe,scite3,0,8000;dbsource "dbsource" true true false 8000 Text 0 0,First,#,gtd_europe,dbsource,0,8000;INT_LOG "INT_LOG" true true false 4 Long 0 0,First,#,gtd_europe,INT_LOG,-1,-1;INT_IDEO "INT_IDEO" true true false 4 Long 0 0,First,#,gtd_europe,INT_IDEO,-1,-1;INT_MISC "INT_MISC" true true false 4 Long 0 0,First,#,gtd_europe,INT_MISC,-1,-1;INT_ANY "INT_ANY" true true false 4 Long 0 0,First,#,gtd_europe,INT_ANY,-1,-1;related "related" true true false 8000 Text 0 0,First,#,gtd_europe,related,0,8000',
            "NO_SOURCE_INFO",
        )
        print("Adding xy coordinates...")
        data = arcpy.management.AddXY(f"{shape}_{size}_data")
        print("Extracting pixel values...")

        print("Exporting csv...")
        arcpy.conversion.TableToTable(
            data, os.getcwd(), f"{shape}_{size}_data.csv",
        )


if __name__ == "__main__":
    arcpy.env.Workspace = True

    args = sys.argv[1:]

    workspace = args[0]
    shape = args[1]
    size = args[2]

    main(workspace, shape, size)
=======
import arcpy
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("workspace", type=str)
parser.add_argument("size", type=str)
args = parser.parse_args()


def main(workspace: str, size: str):
    with arcpy.EnvManager(workspace=workspace):
        extent = "-4077201.58048806 -247892.603949694 3062429.90123652 4981239.14419199"
        sr = arcpy.SpatialReference("Albers Equal Area Conic")

        print(f"Creating Hexagon tesselations of {size} km...")
        tesselations = arcpy.management.GenerateTessellation(
            f"Hexagon_{size}", extent, "HEXAGON", f"{size} SquareKilometers", sr,
        )
        print("Clipping tesselations...")
        clipped = arcpy.analysis.Clip(
            tesselations, "Europe_DissolveBoundaries", f"Hexagon_{size}_clipped"
        )
        print("Creating centroids...")
        centroids = arcpy.management.FeatureToPoint(
            clipped, f"Hexagon_{size}_centroids", "CENTROID"
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
            f"Hexagon_{size}_centroids",
            buffer,
            f"Hexagon_{size}_centroids_erased",
            None,
        )
        print("Merging data...")
        merged = arcpy.management.Merge(
            f"Hexagon_{size}_centroids_erased;gtd_original", f"Hexagon_{size}_data"
        )
        print("Adding xy coordinates...")
        data = arcpy.management.AddXY(f"Hexagon_{size}_data")
        print("Extracting pixel values...")
        data = arcpy.sa.ExtractMultiValuesToPoints(
            data,
            r"Rasters\ppp_2018_1km_Aggregated.tif pop__density;Rasters\acled_raster civil_unrest;Rasters\GDP_grid_flt.tif gdp;Rasters\europe_nighttime_lights nighttime_lights;Rasters\f_15_49_2015.tif f_15_49_50;Rasters\elevation.tif elevation;Rasters\EstISA_final.tif impervious_land;Rasters\distance_major_waterway.tif dist_maj_waterway;Rasters\population_density_2018.tif pop_density;Rasters\landcover.tif landcover;Rasters\slope.tif slope;Rasters\built_settlement_growth_2018.tif built_settlement_growth;Rasters\distance_inland_water.tif dist_inland_water;Rasters\distance_major_road.tif dist_maj_road;Rasters\distance_major_road_intersection.tif dist_maj_road_int",
            "NONE",
        )
        print("Exporting csv...")
        arcpy.conversion.TableToTable(
            data, os.getcwd(), f"Hexagon_{size}_data.csv",
        )


if __name__ == "__main__":
    arcpy.env.Workspace = True

    main(args.workspace, args.size)
>>>>>>> c7ce61c3dca9c198f5e92d9ca9f61eaf6e241aab
