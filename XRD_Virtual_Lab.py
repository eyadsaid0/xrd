
import numpy as np

# --- Physical Constants ---
# X-ray wavelength for Cu K-alpha (Å)
CU_K_ALPHA = 1.5406

# --- Crystal Structures and (hkl) reflections ---
# Define common (hkl) reflections for FCC, BCC, and Diamond cubic structures
# These are based on selection rules for cubic systems
CRYSTAL_STRUCTURES = {
    "FCC": {
        "name": "Face-Centered Cubic",
        "reflections": [
            (1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2), (4, 0, 0),
            (3, 3, 1), (4, 2, 0), (4, 2, 2), (5, 1, 1), (3, 3, 3), (4, 4, 0)
        ]
    },
    "BCC": {
        "name": "Body-Centered Cubic",
        "reflections": [
            (1, 1, 0), (2, 0, 0), (2, 1, 1), (2, 2, 0), (3, 1, 0), (2, 2, 2),
            (3, 2, 1), (4, 0, 0), (4, 1, 1), (3, 3, 0), (4, 2, 0), (4, 2, 2)
        ]
    },
    "Diamond Cubic": {
        "name": "Diamond Cubic", # e.g., Si, Ge
        "reflections": [
            (1, 1, 1), (2, 2, 0), (3, 1, 1), (4, 0, 0), (3, 3, 1), (4, 2, 2),
            (5, 1, 1), (3, 3, 3), (4, 4, 0), (5, 3, 1), (5, 3, 3), (6, 2, 0)
        ]
    }
}

# --- Material Properties (Lattice Constants in Å) ---
MATERIALS = {
    "Silicon (Si)": {
        "crystal_structure": "Diamond Cubic",
        "lattice_constant": 5.431, # Å
        "density": 2.329, # g/cm^3
        "atomic_weight": 28.0855 # g/mol
    },
    "Copper (Cu)": {
        "crystal_structure": "FCC",
        "lattice_constant": 3.615, # Å
        "density": 8.96, # g/cm^3
        "atomic_weight": 63.546 # g/mol
    },
    "Aluminum (Al)": {
        "crystal_structure": "FCC",
        "lattice_constant": 4.049, # Å
        "density": 2.70, # g/cm^3
        "atomic_weight": 26.9815 # g/mol
    }
}

# --- Physics Functions ---

def calculate_d_spacing(h, k, l, a):
    """Calculates d-spacing for cubic crystals."""
    return a / np.sqrt(h**2 + k**2 + l**2)

def bragg_angle(d_spacing, wavelength, n=1):
    """Calculates Bragg angle (theta in radians) from Bragg's Law."""
    # n*lambda = 2*d*sin(theta)
    # sin(theta) = n*lambda / (2*d)
    sin_theta = n * wavelength / (2 * d_spacing)
    # Ensure sin_theta is within valid range [-1, 1] for arcsin
    if sin_theta > 1 or sin_theta < -1:
        return np.nan # No reflection possible
    return np.arcsin(sin_theta)

def scherrer_broadening(crystallite_size_nm, wavelength_A, bragg_angle_rad, K=0.9):
    """Calculates peak broadening (FWHM in radians) using Scherrer equation.
    crystallite_size_nm: crystallite size in nanometers
    wavelength_A: X-ray wavelength in Angstroms
    bragg_angle_rad: Bragg angle in radians
    K: Scherrer constant (typically 0.9)
    Returns: FWHM in radians
    """
    # Convert crystallite size from nm to Å
    crystallite_size_A = crystallite_size_nm * 10
    # Beta = K * lambda / (L * cos(theta))
    beta_rad = (K * wavelength_A) / (crystallite_size_A * np.cos(bragg_angle_rad))
    return beta_rad

def gaussian_peak(x, center, amplitude, fwhm):
    """Generates a Gaussian peak shape.
    x: 2theta values
    center: 2theta peak position
    amplitude: peak intensity
    fwhm: Full Width at Half Maximum
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))

def simulate_xrd_pattern(
    material_name,
    x_ray_wavelength=CU_K_ALPHA,
    scan_range_2theta=(10, 90),
    step_size=0.02,
    crystallite_size_nm=50,
    noise_level=0.05,
    instrumental_broadening_fwhm_deg=0.1 # Example instrumental broadening
):
    """Simulates an XRD pattern for a given material and experimental parameters.
    Returns: 2theta values, intensity values, and a list of detected peaks.
    """
    material_props = MATERIALS[material_name]
    crystal_structure_type = material_props["crystal_structure"]
    lattice_constant = material_props["lattice_constant"]
    reflections = CRYSTAL_STRUCTURES[crystal_structure_type]["reflections"]

    two_theta_deg = np.arange(scan_range_2theta[0], scan_range_2theta[1] + step_size, step_size)
    intensity = np.zeros_like(two_theta_deg)
    detected_peaks = []

    # Convert instrumental broadening from degrees to radians
    instrumental_broadening_fwhm_rad = np.deg2rad(instrumental_broadening_fwhm_deg)

    # Simple relative intensity for common reflections (can be improved with structure factors)
    # For now, higher hkl sum generally means lower intensity, but (111) and (200) are strong
    # This is a simplification and should be replaced by proper structure factor calculations for accuracy
    hkl_intensity_map = {
        (1, 1, 1): 100, (2, 0, 0): 80, (2, 2, 0): 60, (3, 1, 1): 40, (2, 2, 2): 20,
        (1, 1, 0): 100, (2, 0, 0): 70, (2, 1, 1): 50, (2, 2, 0): 30, (3, 1, 0): 20
    }

    for hkl in reflections:
        h, k, l = hkl
        d = calculate_d_spacing(h, k, l, lattice_constant)
        theta_rad = bragg_angle(d, x_ray_wavelength)

        if not np.isnan(theta_rad):
            two_theta_peak_deg = np.rad2deg(2 * theta_rad)

            if scan_range_2theta[0] <= two_theta_peak_deg <= scan_range_2theta[1]:
                # Calculate broadening from crystallite size (Scherrer)
                scherrer_fwhm_rad = scherrer_broadening(crystallite_size_nm, x_ray_wavelength, theta_rad)
                scherrer_fwhm_deg = np.rad2deg(scherrer_fwhm_rad)

                # Combine instrumental and Scherrer broadening (simple sum of FWHM squared)
                # More accurately, use Voigt function, but sum of squares is a good approximation
                total_fwhm_deg = np.sqrt(scherrer_fwhm_deg**2 + instrumental_broadening_fwhm_deg**2)

                # Determine peak amplitude (simplified)
                # Use a default if not in map, or calculate based on hkl sum
                amplitude = hkl_intensity_map.get(hkl, 100 / (h**2 + k**2 + l**2 + 1))
                if amplitude > 100: amplitude = 100 # Cap max intensity

                peak_profile = gaussian_peak(two_theta_deg, two_theta_peak_deg, amplitude, total_fwhm_deg)
                intensity += peak_profile

                detected_peaks.append({
                    "hkl": hkl,
                    "2theta_deg": two_theta_peak_deg,
                    "d_spacing_A": d,
                    "intensity": amplitude, # Store base amplitude for reference
                    "fwhm_deg": total_fwhm_deg
                })

    # Add background noise
    max_intensity = np.max(intensity) if np.max(intensity) > 0 else 1
    intensity += np.random.normal(0, noise_level * max_intensity, len(intensity))
    intensity[intensity < 0] = 0 # Ensure no negative intensities

    return two_theta_deg, intensity, detected_peaks


# Example Usage (for testing the physics engine)
if __name__ == "__main__":
    print("Testing XRD Physics Engine...")
    two_theta, intensities, peaks = simulate_xrd_pattern(
        material_name="Silicon (Si)",
        x_ray_wavelength=CU_K_ALPHA,
        scan_range_2theta=(20, 80),
        step_size=0.05,
        crystallite_size_nm=30,
        noise_level=0.02
    )

    print(f"Simulated {len(two_theta)} data points.")
    print(f"Detected {len(peaks)} peaks.")
    for peak in peaks:
        print(f"  (hkl): {peak['hkl']}, 2theta: {peak['2theta_deg']:.2f}°, d: {peak['d_spacing_A']:.3f} Å, FWHM: {peak['fwhm_deg']:.3f}°")

    # Basic plot for verification
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(two_theta, intensities)
    plt.title("Simulated XRD Pattern for Silicon")
    plt.xlabel("2θ (degrees)")
    plt.ylabel("Intensity (arbitrary units)")
    plt.grid(True)
    plt.show()

