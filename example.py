import slm_3dpointcloud
import numpy
import time

# Function generating points coordinates for a regularly spaced grid of n by n points, with a given size in micrometers
def grid_parameters(n,size):
    cx, cy = numpy.meshgrid(numpy.linspace(-size/2, size/2, n), numpy.linspace(-size/2, size/2, n))
    coords = numpy.ones((int(n**2), 3))
    coords[:, 0] = cx.flatten()
    coords[:, 1] = cy.flatten()
    coords[:, 2] = 0.0
    ints=numpy.ones(int(n**2))

    return coords, ints

# Function generating a rotation transform matrix for given pitch, yaw and roll angles
def rotation_matrix(a, b, g):
    from numpy import cos, sin
    return numpy.array([[cos(a)*cos(b), cos(a)*sin(b)*sin(g)-sin(a)*cos(g), cos(a)*sin(b)*cos(g)+sin(a)*sin(g)],
                        [sin(a)*cos(b), sin(a)*sin(b)*sin(g)+cos(a)*cos(g), sin(a)*sin(b)*cos(g)-cos(a)*sin(g)],
                        [-sin(b)      , cos(b)*sin(g)                     , cos(b)*cos(g)                     ]])


# Magnification of the telescope(s) between SLM and the back aperture of the objective lens of the system.
# set to one if only using the SLM and one lens
slm_to_pupil_magnification = 5.0/3.0

# wavelength of the light source, in nanometers
wavelength_nm=800

# slm pixel dimension, in micrometers
slm_pixel_pitch_um = 9.2

# screen number of the SLM
screenID = 1

# Radius of the gaussian laser beam at the SLM surface, in millimeters. Set to None to consider uniform intensity
beam_radius_slm_mm = 6.0

# focal length of the objective lens, in millimeters (equal to the tube lens focal length of the objective manufacturer,
# divided by the objective magnification)
focal = 180.0/20.0

# The two 8 bit values equivalent to 0 and 2 pi for the grayscale to phase conversion of the SLM. The SLM
# inputs are assumed to be linear in phase.
lut_limits=[0,255]

SLM = slm_3dpointcloud.SlmControl(wavelength_nm,
                                  slm_to_pupil_magnification*slm_pixel_pitch_um,
                                  focal,
                                  slm_to_pupil_magnification*beam_radius_slm_mm,
                                  screenID,
                                  None, #use [x_offset,y_offset,diameter] instead of none to define the pupil in a
                                        #subregion of the slm surface. Sizes must be in pixels.
                                  lut_limits)

# Define coordinates for a regular grid of spots
spots_coords, ints = grid_parameters(10, 300.0)

# Test the RS performance
performance = SLM.rs(spots_coords, ints, get_perf = True)
print("RS performance:")
print("Computation time: "+str(performance["Time"]))
print("Efficiency: "+str(performance["Efficiency"]))
print("Uniformity: "+str(performance["Uniformity"]))
print("\n")

# Test the WGS performance
performance = SLM.wgs(spots_coords, ints, 10, get_perf = True)
print("WGS performance, 10 iterations:")
print("Computation time: "+str(performance["Time"]))
print("Efficiency: "+str(performance["Efficiency"]))
print("Uniformity: "+str(performance["Uniformity"]))
print("\n")

# Test the CS-WGS performance
performance = SLM.cs_wgs(spots_coords, ints, 10, 1/16.0, get_perf = True)
print("CS-WGS performance, 10 iterations, 16x compression:")
print("Computation time: "+str(performance["Time"]))
print("Efficiency: "+str(performance["Efficiency"]))
print("Uniformity: "+str(performance["Uniformity"]))
print("\n")

# Show a live rotating grid. get_perf is set to false for maximum live performance

# Define rotation rates in pitch, yaw and roll
rotation_rates_rad_per_s = numpy.array([1.0, 1.5, 2.0])

#create an empty array with the rotated coordinates
rot_coordinates=numpy.zeros_like(spots_coords)

print("Live grid rotation, RS algorithm")
t_0 = time.perf_counter()
while time.perf_counter()-t_0 < 10.0:
    t = time.perf_counter()-t_0
    angles = rotation_rates_rad_per_s*t
    rot_mat = rotation_matrix(angles[0], angles[1], angles[2])
    for i in range(spots_coords.shape[0]):
        rot_coordinates[i, :] = numpy.dot(rot_mat, spots_coords[i, :])
    SLM.rs(rot_coordinates, ints)

print("Live grid rotation, WGS algorithm, 10 iterations")
t_0 = time.perf_counter()
while time.perf_counter()-t_0 < 10.0:
    t = time.perf_counter()-t_0
    angles = rotation_rates_rad_per_s*t
    rot_mat = rotation_matrix(angles[0], angles[1], angles[2])
    for i in range(spots_coords.shape[0]):
        rot_coordinates[i, :] = numpy.dot(rot_mat, spots_coords[i, :])
    SLM.wgs(rot_coordinates, ints, 10)

print("Live grid rotation, WGS algorithm, 1 iteration")
t_0 = time.perf_counter()
while time.perf_counter()-t_0 < 10.0:
    t = time.perf_counter()-t_0
    angles = rotation_rates_rad_per_s*t
    rot_mat = rotation_matrix(angles[0], angles[1], angles[2])
    for i in range(spots_coords.shape[0]):
        rot_coordinates[i, :] = numpy.dot(rot_mat, spots_coords[i, :])
    SLM.wgs(rot_coordinates, ints, 1)

print("Live grid rotation, CS-WGS algorithm, 10 iterations, 16x compression")
t_0 = time.perf_counter()
while time.perf_counter()-t_0 < 10.0:
    t = time.perf_counter()-t_0
    angles = rotation_rates_rad_per_s*t
    rot_mat = rotation_matrix(angles[0], angles[1], angles[2])
    for i in range(spots_coords.shape[0]):
        rot_coordinates[i, :] = numpy.dot(rot_mat, spots_coords[i, :])
    SLM.cs_wgs(rot_coordinates, ints, 10, 1/16.0)
