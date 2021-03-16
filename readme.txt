Python library for generating 3d point cloud holograms, with phase only spatial light modulators, in real time through a GPU implementation of 5 algorithms (Random superposition, Gerchberg-Saxton, weighted Gerchberg-Saxton, compressed sensing Gerchberg-Saxton, compressed sensing weighted Gerchberg-Saxton).

The GPU implementation of the algorithm is discussed in our publication  "Real time generation of three dimensional patterns for multiphoton stimulation" (https://doi.org/10.3389/fncel.2021.609505 , https://www.frontiersin.org/articles/10.3389/fncel.2021.609505/full). A previous publication describes the algorithms in detail ((https://doi.org/10.3390/mps2010002).

Extremely quick summary of the publications:

- RS: fast computation, low quality holograms
- GS: slow computation, high efficiency holograms
- WGS: slow computation, high uniformity holograms
- CS-GS: faster version of GS, may require some tweaking of the compression parameter for optimal performance.
- CS-WGS: faster version of WGS, may require some tweaking of the compression parameter for optimal performance.

If you use this library for scientific research, please consider citing our work. Any use of this code for distribution of closed-source commercial software is prohibited by the license.

Hardware requirements:

Requires a GPU supporting CUDA 3.0 or newer. Only supports SLMs directly connected to the GPU through VGA/DVI/HDMI/Displayport connection. Phase encoding must be linearly encoded on 8-bit grayscale.
 This unfortunately breaks compatibility with early Meadowlark SLMs, which encoded pixel voltages as a 16-bit numbers over the green and red pixel values without hardware linearization of phase. We do not have access to one of these SLMs for testing, if you are interested in helping us implementing this functionality please contact us.
The library is tested only on Windows 10, 64-bit, but should work with minor tweaks to the installation procedure on other windows versions and linux.

Software requirements:

numpy (install through "pip install numpy")
pyopengl (install through "pip install pyopengl")
screeninfo (install through "pip install screeninfo")
pycuda (further description below)
pyglfw (further description below)

Installation guide:

Quick guide to the installation and setup of CUDA/pyCUDA:

- Install the cuda toolkit (https://developer.nvidia.com/cuda-downloads)
- Install a recent version of visual studio (https://visualstudio.microsoft.com/it/downloads/)
- Add to the system PATH environmental variable the location of the file "cl.exe" within the visual studio installation folder. (beware, the path to the file itself is needed, not just the path to the folder containing the file.)
- Install pyCUDA either through pip (pip install pycuda), or in case of failure, through Christoph Gohlke's pre-compiled binary (https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycuda)

Quick guide to the installation and setup of pyglfw:

- install pyglfw through pip (pip install pyglfw)
- download recent glfw binaries from the official repositiory (https://www.glfw.org/download.html)
- copy the most appropriate binary dll either in the folder with the python code running, or in a folder from the system PATH environmental variable.

Class reference:

class py3dpointcloudslm.SlmControl (wavelength_nm, pixel_size_um, focal_mm, beam_radius_mm= None, screenID=None, active_area_coords=None, lut_edges=[0, 255])

parameters:

wavelength_nm : float
	wavelength of the laser source employed.

pixel_size_um : float
	pixel pitch of the slm in micrometers. If using a system of more than one lens, report the size at the actual pupil of the final focusing lens (e.g. the microscope objective for optogenetics or optical trapping applications), considering  the magnification of other lenses.

focal_mm : float
	focal length of the focusing lens, in mm.

beam_radius_mm = None : float, optional
	expected beam diameter of the laser at the aperture of the focusing lens. The library will assume a gaussian beam of the given radius, centered on the focusing lens pupil. If None, a uniform intensity distribution at the pupil is assumed.

screenID = None : int
	The identification number of the computer screen output used for the SLM. If None, a dummy SLM window is opened on the main computer screen.

active_area_coords=None : list of three ints, optional
	optional pixel coordinates of the active subregion of the SLM, in case not the whole surface is to be used. the list must have structure [y,x,resolution], with y and x being the y and x coordinates of the top left corner of the subregion on the screen, and resolution being the diameter in pixels of the system pupil.

lut_edges=[0, 255] : list of ints, optional
	screen grayscale values for pixel phase modulations of 0 and 2 pi. The library assumes the screen is calibrated to have a grayscale output linearly modulating the phase. these values can be used to set the grayscale interval between 0 and 2 pi.

methods:

rs(spots_coords, spots_ints, get_perf=False):
	compute and project an hologram generating a 3d point cloud using the random superposition algorithm.
	
	parameters:

	spots_coords: Nx3 numpy array. Desired x,y and z coordinates, in micrometers of the N spots in the point cloud
	spots_ints: numpy array. Target intensities of the point cloud spots
	get_perf=False: boolean. If True, the function returns the computation time, efficiency, uniformity and variance of the hologram. Computation of the performance will require extra time, only set to True for diagnostics/benchmarking. 


gs(spots_coords, spots_ints, iterations, get_perf=False):
	compute and project an hologram generating a 3d point cloud using the Gerchberg-Saxton algorithm.
	
	parameters:

	spots_coords: Nx3 numpy array. Desired x,y and z coordinates, in micrometers of the N spots in the point cloud
	spots_ints: numpy array. Target intensities of the point cloud spots
	iterations: int. number of algorithm iterations
	get_perf=False: boolean. If True, the function returns the computation time, efficiency, uniformity and variance of the hologram. Computation of the performance will require extra time, only set to True for diagnostics/benchmarking. 


wgs(spots_coords, spots_ints, iterations, get_perf=False):
	compute and project an hologram generating a 3d point cloud using the weighted Gerchberg-Saxton algorithm.
	
	parameters:

	spots_coords: Nx3 numpy array. Desired x,y and z coordinates, in micrometers of the N spots in the point cloud
	spots_ints: numpy array. Target intensities of the point cloud spots
	iterations: int. number of algorithm iterations
	get_perf=False: boolean. If True, the function returns the computation time, efficiency, uniformity and variance of the hologram. Computation of the performance will require extra time, only set to True for diagnostics/benchmarking. 


cs_gs(spots_coords, spots_ints, iterations, comression, get_perf=False):
	compute and project an hologram generating a 3d point cloud using the compressed sensing Gerchberg-Saxton algorithm.
	
	parameters:

	spots_coords: Nx3 numpy array. Desired x,y and z coordinates, in micrometers of the N spots in the point cloud
	spots_ints: numpy array. Target intensities of the point cloud spots
	iterations: int. number of algorithm iterations
	compression: float. Compression factor in pupil sensing. Must be >0.0 and <1.0.
	get_perf=False: boolean. If True, the function returns the computation time, efficiency, uniformity and variance of the hologram. Computation of the performance will require extra time, only set to True for diagnostics/benchmarking. 

cs_wgs(spots_coords, spots_ints, iterations, comression, get_perf=False):
	compute and project an hologram generating a 3d point cloud using the compressed sensing weighted Gerchberg-Saxton algorithm.
	
	parameters:

	spots_coords: Nx3 numpy array. Desired x,y and z coordinates, in micrometers of the N spots in the point cloud
	spots_ints: numpy array. Target intensities of the point cloud spots
	iterations: int. number of algorithm iterations
	compression: float. Compression factor in pupil sensing. Must be >0.0 and <1.0.
	get_perf=False: boolean. If True, the function returns the computation time, efficiency, uniformity and variance of the hologram. Computation of the performance will require extra time, only set to True for diagnostics/benchmarking. 

get_phase():
        returns the hologram phase as a 2D numpy array.

set_phase(phase):
	set on the hologram a known phase, instead of generating it from points coordinates computation.
        
	parameters:
	
	phase: 2-dimensional numpy array of floats. Input phase distribution, must have size equal to the pupil pixel diameter.
