# Summer practice on Semi-Global Matching algorithm.

The goal is to create an estimation of dense disparity map given two photos of the same subject from slightly different angles. This implementation uses darker colors for smaller pixel differences, i.e. parts closer to photographer.

## Sample results:

### Aerial photography
<p float="middle">
  <img src="https://github.com/sanityseeker/sgm_project/blob/3720b12f90f80d15aa5ecba01288e8cca285e00b/dpth_map_dataset/aero/source_right.jpg" alt="Left" width="32%" />
  <img src="https://github.com/sanityseeker/sgm_project/blob/3720b12f90f80d15aa5ecba01288e8cca285e00b/dpth_map_dataset/aero/source_right.jpg" alt="Right" width="32%" /> 
  <img src="https://raw.githubusercontent.com/LokoDenis/sgm_project/master/results/aero/base_source%2B%2B_-10_30_5_80.jpg" alt="Result disparity map" width="32%" />
</p>

### Aloe scene
<p float="middle">
  <img src="https://github.com/sanityseeker/sgm_project/blob/3720b12f90f80d15aa5ecba01288e8cca285e00b/dpth_map_dataset/aloe_left.jpg" alt="Left" width="32%"/>
  <img src="https://github.com/sanityseeker/sgm_project/blob/3720b12f90f80d15aa5ecba01288e8cca285e00b/dpth_map_dataset/aloe_right.jpg" alt="Right" width="32%"/> 
  <img src="https://raw.githubusercontent.com/LokoDenis/sgm_project/master/results/800_600/final_aloe%2B%2B_-80_90_5_70.jpg" alt="Result disparity map" width="32%" />
</p>

