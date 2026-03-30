#!/usr/bin/env python3
"""
🔬 X-Ray Diffraction (XRD) Virtual Lab
A professional interactive simulation of a real university XRD experiment
Built with Streamlit + Plotly for authentic lab experience
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime
import io

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="XRD Virtual Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for scientific styling
st.markdown("""
<style>
    .main {
        padding-top: 0rem;
    }
    .lab-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .status-ready {
        color: #2ca02c;
        font-weight: bold;
    }
    .status-scanning {
        color: #ff7f0e;
        font-weight: bold;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PHYSICS ENGINE (Imported from XRD_Virtual_Lab.py)
# ============================================================================

CU_K_ALPHA = 1.5406

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
        "name": "Diamond Cubic",
        "reflections": [
            (1, 1, 1), (2, 2, 0), (3, 1, 1), (4, 0, 0), (3, 3, 1), (4, 2, 2),
            (5, 1, 1), (3, 3, 3), (4, 4, 0), (5, 3, 1), (5, 3, 3), (6, 2, 0)
        ]
    }
}

MATERIALS = {
    "Silicon (Si)": {
        "crystal_structure": "Diamond Cubic",
        "lattice_constant": 5.431,
        "density": 2.329,
        "atomic_weight": 28.0855
    },
    "Copper (Cu)": {
        "crystal_structure": "FCC",
        "lattice_constant": 3.615,
        "density": 8.96,
        "atomic_weight": 63.546
    },
    "Aluminum (Al)": {
        "crystal_structure": "FCC",
        "lattice_constant": 4.049,
        "density": 2.70,
        "atomic_weight": 26.9815
    }
}

def calculate_d_spacing(h, k, l, a):
    """Calculates d-spacing for cubic crystals."""
    return a / np.sqrt(h**2 + k**2 + l**2)

def bragg_angle(d_spacing, wavelength, n=1):
    """Calculates Bragg angle (theta in radians) from Bragg's Law."""
    sin_theta = n * wavelength / (2 * d_spacing)
    if sin_theta > 1 or sin_theta < -1:
        return np.nan
    return np.arcsin(sin_theta)

def scherrer_broadening(crystallite_size_nm, wavelength_A, bragg_angle_rad, K=0.9):
    """Calculates peak broadening using Scherrer equation."""
    crystallite_size_A = crystallite_size_nm * 10
    beta_rad = (K * wavelength_A) / (crystallite_size_A * np.cos(bragg_angle_rad))
    return beta_rad

def gaussian_peak(x, center, amplitude, fwhm):
    """Generates a Gaussian peak shape."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))

def simulate_xrd_pattern(
    material_name,
    x_ray_wavelength=CU_K_ALPHA,
    scan_range_2theta=(10, 90),
    step_size=0.02,
    crystallite_size_nm=50,
    noise_level=0.05,
    instrumental_broadening_fwhm_deg=0.1
):
    """Simulates an XRD pattern for a given material."""
    material_props = MATERIALS[material_name]
    crystal_structure_type = material_props["crystal_structure"]
    lattice_constant = material_props["lattice_constant"]
    reflections = CRYSTAL_STRUCTURES[crystal_structure_type]["reflections"]

    two_theta_deg = np.arange(scan_range_2theta[0], scan_range_2theta[1] + step_size, step_size)
    intensity = np.zeros_like(two_theta_deg)
    detected_peaks = []

    instrumental_broadening_fwhm_rad = np.deg2rad(instrumental_broadening_fwhm_deg)

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
                scherrer_fwhm_rad = scherrer_broadening(crystallite_size_nm, x_ray_wavelength, theta_rad)
                scherrer_fwhm_deg = np.rad2deg(scherrer_fwhm_rad)
                total_fwhm_deg = np.sqrt(scherrer_fwhm_deg**2 + instrumental_broadening_fwhm_deg**2)

                amplitude = hkl_intensity_map.get(hkl, 100 / (h**2 + k**2 + l**2 + 1))
                if amplitude > 100: amplitude = 100

                peak_profile = gaussian_peak(two_theta_deg, two_theta_peak_deg, amplitude, total_fwhm_deg)
                intensity += peak_profile

                detected_peaks.append({
                    "hkl": hkl,
                    "2theta_deg": two_theta_peak_deg,
                    "d_spacing_A": d,
                    "intensity": amplitude,
                    "fwhm_deg": total_fwhm_deg
                })

    max_intensity = np.max(intensity) if np.max(intensity) > 0 else 1
    intensity += np.random.normal(0, noise_level * max_intensity, len(intensity))
    intensity[intensity < 0] = 0

    return two_theta_deg, intensity, detected_peaks

def estimate_lattice_constant(peaks, material_name):
    """Estimates lattice constant from detected peaks using Bragg's Law."""
    material_props = MATERIALS[material_name]
    estimates = []
    
    for peak in peaks:
        h, k, l = peak['hkl']
        d = peak['d_spacing_A']
        a_est = d * np.sqrt(h**2 + k**2 + l**2)
        estimates.append(a_est)
    
    if estimates:
        return np.mean(estimates), np.std(estimates)
    return None, None

def estimate_crystallite_size(peaks, x_ray_wavelength=CU_K_ALPHA, K=0.9):
    """Estimates crystallite size from peak broadening using Scherrer equation."""
    sizes = []
    
    for peak in peaks:
        fwhm_deg = peak['fwhm_deg']
        fwhm_rad = np.deg2rad(fwhm_deg)
        
        h, k, l = peak['hkl']
        d = peak['d_spacing_A']
        theta_rad = bragg_angle(d, x_ray_wavelength)
        
        if not np.isnan(theta_rad):
            # Scherrer: L = K * lambda / (beta * cos(theta))
            # Assuming beta is the Scherrer broadening only (not including instrumental)
            # For simplicity, use the total FWHM as an upper estimate
            L_A = (K * x_ray_wavelength) / (fwhm_rad * np.cos(theta_rad))
            L_nm = L_A / 10
            if L_nm > 0:
                sizes.append(L_nm)
    
    if sizes:
        return np.mean(sizes), np.std(sizes)
    return None, None

# ============================================================================
# STREAMLIT SESSION STATE MANAGEMENT
# ============================================================================

if 'scan_completed' not in st.session_state:
    st.session_state.scan_completed = False
if 'scan_data' not in st.session_state:
    st.session_state.scan_data = None
if 'peaks_data' not in st.session_state:
    st.session_state.peaks_data = None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="lab-header">
        <h1>🔬 X-Ray Diffraction Virtual Lab</h1>
        <p>Professional Interactive Simulation of a Real University XRD Experiment</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    with st.sidebar:
        st.markdown("## 🎛️ Lab Controls")
        
        # Step 1: Material Selection
        st.markdown("### Step 1: Select Material")
        material = st.selectbox(
            "Choose your sample:",
            list(MATERIALS.keys()),
            help="Select the crystalline material to analyze"
        )
        
        mat_props = MATERIALS[material]
        crystal_type = mat_props["crystal_structure"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Crystal System", crystal_type.split()[0])
        with col2:
            st.metric("Lattice (Å)", f"{mat_props['lattice_constant']:.3f}")
        
        st.divider()
        
        # Step 2: Experiment Parameters
        st.markdown("### Step 2: Set Parameters")
        
        wavelength = st.slider(
            "X-ray Wavelength (Å):",
            min_value=0.5,
            max_value=2.5,
            value=CU_K_ALPHA,
            step=0.01,
            help="Cu Kα = 1.5406 Å (default)"
        )
        
        scan_start = st.number_input(
            "2θ Start (°):",
            min_value=5,
            max_value=30,
            value=10,
            step=1
        )
        
        scan_end = st.number_input(
            "2θ End (°):",
            min_value=scan_start + 10,
            max_value=150,
            value=90,
            step=1
        )
        
        step_size = st.slider(
            "Step Size (°):",
            min_value=0.01,
            max_value=0.5,
            value=0.05,
            step=0.01,
            help="Smaller step = higher resolution but longer scan"
        )
        
        st.divider()
        
        # Step 3: Sample Properties
        st.markdown("### Step 3: Sample Properties")
        
        crystallite_size = st.slider(
            "Crystallite Size (nm):",
            min_value=5,
            max_value=500,
            value=50,
            step=5,
            help="Smaller size → broader peaks (Scherrer broadening)"
        )
        
        noise_level = st.slider(
            "Noise Level (%):",
            min_value=0,
            max_value=20,
            value=5,
            step=1,
            help="Simulates experimental noise"
        )
        
        st.divider()
        
        # Step 4: Run Experiment
        st.markdown("### Step 4: Run Experiment")
        
        if st.button("▶️ START SCAN", use_container_width=True, type="primary"):
            st.session_state.scan_completed = False
            st.session_state.scan_data = None
            st.session_state.peaks_data = None
            
            # Simulate scanning process
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.markdown('<p class="status-scanning">🔄 Scanning in progress...</p>', unsafe_allow_html=True)
            
            # Simulate gradual scan
            for i in range(100):
                time.sleep(0.02)  # Simulate measurement time
                progress_bar.progress(i / 100)
            
            # Run actual simulation
            two_theta, intensity, peaks = simulate_xrd_pattern(
                material_name=material,
                x_ray_wavelength=wavelength,
                scan_range_2theta=(scan_start, scan_end),
                step_size=step_size,
                crystallite_size_nm=crystallite_size,
                noise_level=noise_level/100
            )
            
            progress_bar.progress(1.0)
            status_text.markdown('<p class="status-ready">✅ Scan Complete!</p>', unsafe_allow_html=True)
            
            # Store results in session state
            st.session_state.scan_data = {
                'two_theta': two_theta,
                'intensity': intensity,
                'material': material,
                'wavelength': wavelength,
                'crystallite_size': crystallite_size,
                'noise_level': noise_level
            }
            st.session_state.peaks_data = peaks
            st.session_state.scan_completed = True
            
            time.sleep(1)
            st.rerun()

    # Main Content Area
    if not st.session_state.scan_completed:
        st.info("👈 Configure your experiment in the sidebar and click **START SCAN** to begin!")
        
        # Display crystal structure info
        st.markdown("## 📚 Crystal Structure Information")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Material:** {material}")
        with col2:
            st.markdown(f"**System:** {MATERIALS[material]['crystal_structure']}")
        with col3:
            st.markdown(f"**Density:** {MATERIALS[material]['density']} g/cm³")
        
        # Show expected peaks
        st.markdown("## 📊 Expected Reflections")
        
        crystal_type = MATERIALS[material]["crystal_structure"]
        lattice_const = MATERIALS[material]["lattice_constant"]
        reflections = CRYSTAL_STRUCTURES[crystal_type]["reflections"]
        
        expected_peaks = []
        for hkl in reflections[:8]:  # Show first 8
            h, k, l = hkl
            d = calculate_d_spacing(h, k, l, lattice_const)
            theta_rad = bragg_angle(d, CU_K_ALPHA)
            if not np.isnan(theta_rad):
                two_theta = np.rad2deg(2 * theta_rad)
                expected_peaks.append({
                    "(hkl)": f"({h}{k}{l})",
                    "d-spacing (Å)": f"{d:.4f}",
                    "2θ (°)": f"{two_theta:.2f}"
                })
        
        df_expected = pd.DataFrame(expected_peaks)
        st.dataframe(df_expected, use_container_width=True)
    
    else:
        # Display scan results
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Experiment", "🔍 Analysis", "📈 Peak Analysis", "📥 Export"])
        
        with tab1:
            st.markdown("## XRD Scan Results")
            
            # Display scan parameters
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Material", st.session_state.scan_data['material'])
            with col2:
                st.metric("Wavelength (Å)", f"{st.session_state.scan_data['wavelength']:.4f}")
            with col3:
                st.metric("Crystallite Size (nm)", f"{st.session_state.scan_data['crystallite_size']}")
            with col4:
                st.metric("Noise Level (%)", f"{st.session_state.scan_data['noise_level']}")
            
            st.divider()
            
            # Interactive plot
            two_theta = st.session_state.scan_data['two_theta']
            intensity = st.session_state.scan_data['intensity']
            peaks = st.session_state.peaks_data
            
            fig = go.Figure()
            
            # Add main XRD pattern
            fig.add_trace(go.Scatter(
                x=two_theta,
                y=intensity,
                mode='lines',
                name='XRD Pattern',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                hovertemplate='<b>2θ:</b> %{x:.2f}°<br><b>Intensity:</b> %{y:.2f}<extra></extra>'
            ))
            
            # Mark detected peaks
            peak_2theta = [p['2theta_deg'] for p in peaks]
            peak_intensity = [np.max(intensity[np.argmin(np.abs(two_theta - p['2theta_deg']))]) for p in peaks]
            
            fig.add_trace(go.Scatter(
                x=peak_2theta,
                y=peak_intensity,
                mode='markers+text',
                name='Detected Peaks',
                marker=dict(size=8, color='red', symbol='star'),
                text=[f"({p['hkl'][0]}{p['hkl'][1]}{p['hkl'][2]})" for p in peaks],
                textposition='top center',
                hovertemplate='<b>(hkl):</b> %{text}<br><b>2θ:</b> %{x:.2f}°<extra></extra>'
            ))
            
            fig.update_layout(
                title="X-Ray Diffraction Pattern",
                xaxis_title="2θ (degrees)",
                yaxis_title="Intensity (counts)",
                template="plotly_white",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("## Crystallographic Analysis")
            
            material = st.session_state.scan_data['material']
            peaks = st.session_state.peaks_data
            
            # Estimate lattice constant
            a_est, a_std = estimate_lattice_constant(peaks, material)
            a_true = MATERIALS[material]['lattice_constant']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Measured Lattice (Å)", f"{a_est:.4f}" if a_est else "N/A")
            with col2:
                st.metric("Theoretical Lattice (Å)", f"{a_true:.4f}")
            with col3:
                error_pct = abs(a_est - a_true) / a_true * 100 if a_est else 0
                st.metric("Error (%)", f"{error_pct:.2f}%")
            
            st.divider()
            
            # Estimate crystallite size
            L_est, L_std = estimate_crystallite_size(peaks)
            L_true = st.session_state.scan_data['crystallite_size']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Estimated Size (nm)", f"{L_est:.1f}" if L_est else "N/A")
            with col2:
                st.metric("Input Size (nm)", f"{L_true}")
            with col3:
                if L_est:
                    size_error = abs(L_est - L_true) / L_true * 100
                    st.metric("Error (%)", f"{size_error:.1f}%")
        
        with tab3:
            st.markdown("## Detected Peaks")
            
            peaks = st.session_state.peaks_data
            
            peak_data = []
            for p in peaks:
                peak_data.append({
                    "(hkl)": f"({p['hkl'][0]}{p['hkl'][1]}{p['hkl'][2]})",
                    "2θ (°)": f"{p['2theta_deg']:.2f}",
                    "d-spacing (Å)": f"{p['d_spacing_A']:.4f}",
                    "FWHM (°)": f"{p['fwhm_deg']:.3f}",
                    "Intensity": f"{p['intensity']:.1f}"
                })
            
            df_peaks = pd.DataFrame(peak_data)
            st.dataframe(df_peaks, use_container_width=True)
        
        with tab4:
            st.markdown("## Export Data")
            
            two_theta = st.session_state.scan_data['two_theta']
            intensity = st.session_state.scan_data['intensity']
            peaks = st.session_state.peaks_data
            
            # Create export dataframe
            df_export = pd.DataFrame({
                '2θ (degrees)': two_theta,
                'Intensity (counts)': intensity
            })
            
            # Download CSV
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="📊 Download XRD Data (CSV)",
                data=csv,
                file_name=f"xrd_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Generate report
            st.markdown("### Experiment Report")
            
            material = st.session_state.scan_data['material']
            a_est, _ = estimate_lattice_constant(peaks, material)
            L_est, _ = estimate_crystallite_size(peaks)
            
            report = f"""
# XRD Experiment Report
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experimental Parameters
- **Material:** {material}
- **X-ray Wavelength:** {st.session_state.scan_data['wavelength']:.4f} Å
- **2θ Range:** {st.session_state.scan_data['two_theta'][0]:.1f}° - {st.session_state.scan_data['two_theta'][-1]:.1f}°
- **Crystallite Size:** {st.session_state.scan_data['crystallite_size']} nm
- **Noise Level:** {st.session_state.scan_data['noise_level']}%

## Results
- **Measured Lattice Constant:** {a_est:.4f} Å
- **Theoretical Lattice Constant:** {MATERIALS[material]['lattice_constant']:.4f} Å
- **Estimated Crystallite Size:** {L_est:.1f} nm
- **Number of Detected Peaks:** {len(peaks)}

## Detected Reflections
| (hkl) | 2θ (°) | d-spacing (Å) |
|-------|--------|---------------|
"""
            
            for p in peaks[:10]:  # Show first 10 peaks
                report += f"| ({p['hkl'][0]}{p['hkl'][1]}{p['hkl'][2]}) | {p['2theta_deg']:.2f} | {p['d_spacing_A']:.4f} |\n"
            
            st.markdown(report)
            
            # Download report as text
            st.download_button(
                label="📄 Download Report (TXT)",
                data=report,
                file_name=f"xrd_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
