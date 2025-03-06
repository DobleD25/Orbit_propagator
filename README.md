
# Orbit propagator

Simple orbit propagator with maneuvers and mission comparison. It supports every orbital environment excepting low orbits.
It includes options to customize the central body, adding impulsive or continuous maneuvers and visualization of several trajectories with different initial properties.


## Overview

This Python program is designed to simulate and propagate spacecraft orbits around a central body, considering various perturbations and maneuvers. It utilizes numerical integration methods for orbit propagation and generates both graphical plots and CSV files containing ephemeris data. The program's configuration, including initial orbit parameters, spacecraft properties, central body characteristics, and perturbation settings, is defined through a JSON input file (`Input.json`).





## Features



* **Orbit Propagation:**
    * Numerical integration using Runge-Kutta 4th order (RK4) and Runge-Kutta 5th/6th order (RK56) methods.
    * Propagation in J2000 Cartesian coordinates.
    * Customizable time step and total time span.
* **Perturbation Modeling:**
    * **Non-Spherical Gravity:** Incorporates the Earth's non-spherical gravity field using the EGM96 model or custom J2, J3, C22, and S22 coefficients.
    * **N-Body Perturbations:** Accounts for gravitational influences from other celestial bodies like the Moon and the Sun.
    * **Solar Radiation Pressure (SRP):** Models the perturbation due to solar radiation pressure acting on the spacecraft.
* **Maneuver Capabilities:**
    * **Chemical Maneuvers:** Impulsive maneuvers defined by epoch and Delta-V components in the VNB (Velocity-Normal-Binormal) frame.
    * **Electrical Maneuvers:** Continuous thrust maneuvers with options for defining thrust and time, thrust and Delta-V, or Delta-V and time, also defined in the VNB frame.
* **Eclipse Calculation:** Determines eclipse status (No eclipse, Partial eclipse, Total eclipse) with respect to user-defined eclipsing bodies (e.g., Earth, Moon).
* **Output Generation:**
    * **CSV Ephemeris Files:** Saves propagation results in CSV format, including Cartesian states, Keplerian elements, and eclipse statuses.
    * **Graphical Plots:** Generates plots of:
        * 2D and 3D Orbit Trajectories.
        * Evolution of Keplerian elements over time.
        * Latitude and Longitude in the body-fixed frame with groundtrack visualization.
* **Configurable via JSON:** All simulation parameters are defined in a `Input.json` file.



## Getting Started

### Prerequisites

* **Python 3.x:**  Make sure you have Python 3 installed on your system.
* **Python Libraries:**  Install the required Python libraries using pip:
    ```bash
    pip install numpy spiceypy matplotlib pandas astropy
    ```

### Installation

1. **Clone the Repository (or download the script files):**
   If you have access to the code repository, clone it using Git:
   ```bash
   git clone [repository_url]
   cd [repository_directory]
Alternatively, download the Python script files and place them in a directory of your choice.

Navigate to the Program Directory: Open a terminal or command prompt and navigate to the directory where you have saved the program files.
## Usage/Examples

Input JSON File (Input.json):

The program's behavior is controlled by the Input.json file located in the main directory.
Important: Edit Input.json to define your desired simulation parameters, orbits, spacecraft, perturbations, and maneuvers. See the "Configuration Details (Input.json)" section below for a detailed explanation of the JSON structure.

Usage
Run the Program:
Execute the main Python script (main.py) from your terminal:
```bash
python start_prop.py
```

Program Execution:

The program will read the Input.json file.
It will propagate the orbit(s) as defined in the JSON configuration.

During propagation, it will apply defined maneuvers and calculate eclipse statuses at each time step.

After propagation, it will:

-Save ephemeris data in CSV files (Cartesian and Keplerian states with eclipse status).

-Generate plots of the orbit trajectories and Keplerian elements and save them in the "Plots" subdirectory.

#### Output Files:

CSV Files:

* *ephemerides_cartesians_orbit_[mission_index].csv*: CSV file containing time, Cartesian position (x, y, z), velocity (vx, vy, vz), and eclipse status for each time step.
* *ephemerides_keplerians_orbit_[mission_index].csv*: CSV file containing time, Keplerian elements (a, e, i, Omega_AN, omega_PER, nu), and eclipse status for each time step.
Plot Files:

* *Orbit_Trajectory_2D_Orbit_[mission_index].png*: 2D plot of the orbit trajectory in the XY plane.
* *Orbit_Trajectory_3D_Orbit_[mission_index].png*: 3D plot of the orbit trajectory.
* *Keplerian_Elements_Evolution_Orbit_[mission_index].png*: Plots showing the evolution of Semi-major axis (a), Eccentricity (e), and Inclination (i), argument of periapsis (aop), righ ascension of ascending node (raan) over time.
* *Grountrack_[mission_index].png*: Plot of Latitude and Longitude in the body-fixed frame over time.

## Documentation

The initial properties of the mission are introduced in and JSON file.

### Example: 

{

  Missions { 

    {
      "system": "cartesians",
      "Keplerian_coordinates": [
        {
          "a": 0,
          "e": 0,
          "i": 0,
          "Omega_AN": 0,
          "omega_PER": 0,
          "nu": 0
        }
      ],
      "Cartesian_coordinates": [
        {
          "x": 41264.2396,
          "y": -8655.6642909999,
          "z": 0.829481,
          "vx": 0.631648,
          "vy": 3.009276,
          "vz": -0.001085
        }
      ],
      "color": "yellow",
      "step_size": 1000,
      "time_span": 259200,
      "init_epoch": "2025-01-14T18:00:00",
      "Maneouvers": [
        {
          "Chemical": [
            {
              "Epoch": "2025-01-14T18:02:30",
              "DeltaV (VNB)": [0, 0, 500]
            }
          ],
          "Electrical": [
            {
              "Epoch": "2025-01-19T18:00:00",
              "Trust and Time (TT) or Thrust and DeltaV (TD) or DeltaV and Time (DT)": "TT",
              "Thrust(VNB)": [0, 0, 0],
              "Time": 86400,
              "DeltaV (VNB)": [0, 0, 0]
            }
          ]
        }
      ]
    }
  ],

  "Spacecraft": [

    {
      "name": "AG1",
      "mass": 1895.546,
      "area": 44.75,
      "Cr": 1
    }
  ],
  "Central_body": [

    {
      "name": "Earth",
      "mass": 5.972e+24,
      "radius": 6378.137,
      "eclipsing_bodies": ["Earth", "Moon"]
    }
  ],
  "Perturbations": [

    {
      "Non_spherical_body": [
        {
          "value": true,
          "coefficients": [
            {
              "J2": 0.0010826,
              "J3": -2.53e-06,
              "C22": 1.57e-06,
              "S22": -9e-07
            }
          ],
          "EGM96_model": true
        }
      ],
      "N-body": [
        {
          "value": true,
          "list": ["moon", "sun"]
        }
      ],
      "SRP": [
        {
          "value": true
        }
      ]
    }
  ]
}

### Sections and Parameters:

* **Missions**  :  Defines a list of missions to propagate. Each mission configuration is a dictionary:

    * **"system"** : Coordinate system for initial state ("cartesians" is currently used).
    * **"Keplerian_coordinates"**: Array containing Keplerian elements (not actively used in the example as initial state is Cartesian).
        * "a": Semi-major axis.
        * "e": Eccentricity.
        * "i": Inclination (degrees).
        * "raan": Right ascension of the ascending node (degrees).
        * "aop": Argument of periapsis (degrees).
        * "nu": True anomaly (degrees).
    * **"Cartesian_coordinates"**: Array containing Cartesian initial state.
        * "x", "y", "z": Initial position coordinates.
        * "vx", "vy", "vz": Initial velocity components.
    * **"color"** : Color for plotting the orbit trajectory.
    * **"step_size"**: Time step for propagation (seconds).
    * **"time_span"**: Total propagation time (seconds).
    * **"init_epoch"**: Initial epoch in UTC Gregorian format (e.g., "2025-01-14T18:00:00").
* **"Maneouvers"**: Array defining maneuvers for this mission.

  * **"Chemical"**: Array of chemical (impulsive) maneuvers.
    * "Epoch": Maneuver epoch.
    * "DeltaV (VNB)": Delta-V components in VNB frame [dV_V, dV_N, dV_B] (in m/s).

  * **"Electrical"**: Array of electrical (continuous thrust) maneuvers.
    * "Epoch": Start epoch of electrical maneuver.
    * "Trust and Time (TT) or Thrust and DeltaV (TD) or DeltaV and Time (DT)": Method for defining electrical maneuver.
    * "Thrust(VNB)": Thrust vector components in VNB frame.
    * "Time": Maneuver duration (seconds).
    * "DeltaV (VNB)": Delta-V vector components (used if "TD" or  "DT" is selected).
* **Spacecraft**: Defines spacecraft properties.

  * **"name"** : Spacecraft name (optional).
  * **"mass"**: Spacecraft mass (kg).
  * **"area"**: Spacecraft area (m^2) for SRP calculation.
  * **"Cr"**: Coefficient of reflectivity for SRP calculation.
* **Central_body** : Defines the central body.

  **"name"**: Central body name (e.g., "Earth").
  **"mass"**: Central body mass (kg).
  **"radius"** : Central body radius (km).
  **"eclipsing_bodies"**: List of bodies that can cause eclipses (e.g., ["Earth", "Moon"]).
* **Perturbations**: Defines perturbation settings.

  * **"Non_spherical_body"**: Settings for non-spherical gravity perturbation.
    * "value": true to enable, false to disable.
    * "coefficients": Custom gravity coefficients (J2, J3, C22, S22). Optional if EGM96_model is enable.
    * "EGM96_model": true to use EGM96 model (overrides custom coefficients if true).
  * **"N-body"**: Settings for N-body perturbations.
    * "value": true to enable, false to disable.
    * "list": List of bodies to include in N-body perturbation (e.g., ["moon", "sun"]).
  * **"SRP"** : Settings for Solar Radiation Pressure.
    * "value": true to enable, false to disable.
