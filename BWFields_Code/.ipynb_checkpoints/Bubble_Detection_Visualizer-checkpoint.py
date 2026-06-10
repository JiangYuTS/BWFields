#!/usr/bin/env python3
"""
Bubble Detection Process Visualization Tool - Python Version
============================================================

Interactive tool for visualizing bubble detection in 13CO PPV data
with adjustable parameters and real-time updates.

Author: Assistant
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

class BubbleDetectionVisualizer:
    def __init__(self):
        """Initialize the bubble detection visualizer"""
        self.setup_figure()
        self.initialize_parameters()
        self.generate_synthetic_data()
        self.setup_widgets()
        self.update_visualization()
        
    def setup_figure(self):
        """Setup the main figure and subplots"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('🔍 Bubble Detection in 13CO PPV Data - Interactive Tool', 
                         fontsize=16, fontweight='bold', color='cyan')
        
        # Create grid layout
        gs = GridSpec(4, 4, figure=self.fig, hspace=0.4, wspace=0.3)
        
        # Main visualization panels
        self.ax_intensity = self.fig.add_subplot(gs[0, 0])
        self.ax_radial = self.fig.add_subplot(gs[0, 1])
        self.ax_pv = self.fig.add_subplot(gs[0, 2])
        self.ax_spectra = self.fig.add_subplot(gs[0, 3])
        
        # Workflow and statistics
        self.ax_workflow = self.fig.add_subplot(gs[1, :])
        self.ax_stats = self.fig.add_subplot(gs[2, :2])
        self.ax_results = self.fig.add_subplot(gs[2, 2:])
        
        # Parameter controls area
        self.ax_controls = self.fig.add_subplot(gs[3, :])
        
        # Style the axes
        for ax in [self.ax_intensity, self.ax_radial, self.ax_pv, self.ax_spectra]:
            ax.set_facecolor('black')
            ax.grid(True, alpha=0.3, color='gray')
        
        for ax in [self.ax_workflow, self.ax_stats, self.ax_results, self.ax_controls]:
            ax.set_facecolor('black')
            ax.axis('off')
    
    def initialize_parameters(self):
        """Initialize detection parameters"""
        self.params = {
            'noise_threshold': 6.0,
            'smoothing_sigma': 1.5,
            'min_radius': 3,
            'max_radius': 30,
            'ring_width_ratio': 0.3,
            'min_expansion_vel': 0.5,
            'min_contrast_ratio': 1.2,
            'confidence_threshold': 0.5,
            'duplicate_distance': 10
        }
        
        self.detection_results = {
            'candidates': [],
            'confirmed': [],
            'rejected': [],
            'stats': {}
        }
    
    def generate_synthetic_data(self):
        """Generate synthetic bubble data for demonstration"""
        # Create synthetic bubble structures
        self.bubble_data = []
        self.radii = [5, 8, 12, 15, 20]
        self.centers = [(25, 25), (60, 40), (45, 70), (80, 60), (90, 90)]
        
        # Generate integrated intensity map data
        x, y = np.meshgrid(np.linspace(0, 100, 100), np.linspace(0, 100, 100))
        self.intensity_map = np.random.normal(0.5, 0.1, (100, 100))
        
        # Add bubble structures
        for i, (radius, (cx, cy)) in enumerate(zip(self.radii, self.centers)):
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # Create ring structure: strong at shell, weak at center
            shell_mask = (r >= radius * 0.7) & (r <= radius * 1.3)
            center_mask = r <= radius * 0.5
            
            self.intensity_map[shell_mask] += 2.0 + np.random.normal(0, 0.3, np.sum(shell_mask))
            self.intensity_map[center_mask] *= 0.3  # Dim the center
        
        self.intensity_map = np.maximum(self.intensity_map, 0)
        
        # Create initial bubble candidates
        self.create_bubble_candidates()
        
        # Generate velocity axis
        self.velocity_axis = np.linspace(-10, 10, 41)
        
    def create_bubble_candidates(self):
        """Create bubble candidates based on synthetic data"""
        candidates = []
        for i, (radius, (cx, cy)) in enumerate(zip(self.radii, self.centers)):
            candidates.append({
                'id': i + 1,
                'center_x': cx,
                'center_y': cy,
                'radius': radius,
                'contrast_ratio': 1.2 + np.random.random() * 1.5,
                'expansion_velocity': 0.3 + np.random.random() * 2.0,
                'confidence': 0.4 + np.random.random() * 0.5,
                'criteria': []
            })
        
        self.detection_results['candidates'] = candidates
    
    def setup_widgets(self):
        """Setup interactive parameter controls"""
        # Clear the controls area
        self.ax_controls.clear()
        self.ax_controls.axis('off')
        
        # Create sliders
        slider_props = [
            ('noise_threshold', 'Noise Threshold (σ)', 1, 10, 6.0, 0.5),
            ('min_radius', 'Min Radius (px)', 2, 10, 3, 1),
            ('max_radius', 'Max Radius (px)', 15, 50, 30, 1),
            ('min_expansion_vel', 'Min Expansion (km/s)', 0.1, 3.0, 0.5, 0.1),
            ('min_contrast_ratio', 'Min Contrast', 1.0, 3.0, 1.2, 0.1),
            ('confidence_threshold', 'Confidence Threshold', 0.3, 1.0, 0.5, 0.05)
        ]
        
        self.sliders = {}
        for i, (name, label, vmin, vmax, vinit, vstep) in enumerate(slider_props):
            # Create slider axis
            slider_ax = plt.axes([0.15 + (i % 3) * 0.25, 0.12 - (i // 3) * 0.08, 0.2, 0.03])
            slider = Slider(slider_ax, label, vmin, vmax, valinit=vinit, 
                          valstep=vstep, facecolor='cyan', alpha=0.8)
            slider.on_changed(lambda val, param=name: self.update_parameter(param, val))
            self.sliders[name] = slider
        
        # Add control buttons
        btn_detect_ax = plt.axes([0.85, 0.12, 0.1, 0.04])
        self.btn_detect = Button(btn_detect_ax, 'Run Detection', color='green')
        self.btn_detect.on_clicked(self.run_detection)
        
        btn_reset_ax = plt.axes([0.85, 0.06, 0.1, 0.04])
        self.btn_reset = Button(btn_reset_ax, 'Reset', color='orange')
        self.btn_reset.on_clicked(self.reset_detection)
    
    def update_parameter(self, param_name, value):
        """Update parameter and refresh visualization"""
        self.params[param_name] = value
        if len(self.detection_results['candidates']) > 0:
            self.run_detection(None)
    
    def update_intensity_map(self):
        """Update integrated intensity map visualization"""
        self.ax_intensity.clear()
        self.ax_intensity.set_facecolor('black')
        
        # Plot intensity map
        im = self.ax_intensity.imshow(self.intensity_map, origin='lower', 
                                    cmap='viridis', extent=[0, 100, 0, 100])
        
        # Add bubble markers
        colors = ['red', 'yellow', 'cyan', 'magenta', 'white']
        for i, (radius, (cx, cy)) in enumerate(zip(self.radii, self.centers)):
            circle = Circle((cx, cy), radius, fill=False, color=colors[i % len(colors)], 
                          linewidth=2, alpha=0.8)
            self.ax_intensity.add_patch(circle)
            self.ax_intensity.plot(cx, cy, '+', color=colors[i % len(colors)], 
                                 markersize=10, markeredgewidth=2)
        
        self.ax_intensity.set_title('Integrated Intensity Map\n(Bubble Morphology)', 
                                  color='white', fontweight='bold')
        self.ax_intensity.set_xlabel('X (pixels)', color='white')
        self.ax_intensity.set_ylabel('Y (pixels)', color='white')
        self.ax_intensity.grid(True, alpha=0.3)
    
    def update_radial_profile(self):
        """Update radial profile plot"""
        self.ax_radial.clear()
        self.ax_radial.set_facecolor('black')
        
        # Use first bubble for radial profile
        if self.centers:
            center = self.centers[0]
            radius = self.radii[0]
            
            # Calculate radial profile
            radii = np.linspace(0, 25, 50)
            profile = []
            
            for r in radii:
                if r < radius * 0.3:
                    intensity = 0.5 + np.random.normal(0, 0.1)  # Center cavity
                elif radius * 0.7 <= r <= radius * 1.3:
                    intensity = 2.5 * np.exp(-((r - radius) / 2)**2) + np.random.normal(0, 0.1)  # Shell
                else:
                    intensity = 0.8 + np.random.normal(0, 0.1)  # Background
                profile.append(max(0, intensity))
            
            self.ax_radial.plot(radii, profile, 'cyan', linewidth=2, label='Intensity Profile')
            self.ax_radial.axvline(radius, color='red', linestyle='--', alpha=0.7, 
                                 label=f'Bubble Radius ({radius}px)')
            self.ax_radial.fill_between([0, radius*0.5], 0, max(profile), 
                                      alpha=0.3, color='blue', label='Center')
            self.ax_radial.fill_between([radius*0.7, radius*1.3], 0, max(profile), 
                                      alpha=0.3, color='red', label='Shell')
        
        self.ax_radial.set_title('Radial Intensity Profile\n(Shell Structure)', 
                               color='white', fontweight='bold')
        self.ax_radial.set_xlabel('Radius (pixels)', color='white')
        self.ax_radial.set_ylabel('Mean Intensity', color='white')
        self.ax_radial.legend(fontsize=8)
        self.ax_radial.grid(True, alpha=0.3)
    
    def update_pv_diagram(self):
        """Update position-velocity diagram"""
        self.ax_pv.clear()
        self.ax_pv.set_facecolor('black')
        
        # Create expansion signature
        positions = np.linspace(-20, 20, 81)
        velocities = []
        intensities = []
        
        for pos in positions:
            distance = abs(pos)
            if distance > 3:  # Only shell region
                expansion_vel = 0.3 * distance
                # Blue and red-shifted components
                for sign in [-1, 1]:
                    vel = sign * expansion_vel + np.random.normal(0, 0.2)
                    intensity = np.exp(-((distance - 8) / 3)**2) + np.random.normal(0, 0.1)
                    if intensity > 0:
                        velocities.append(vel)
                        intensities.append(intensity)
                        self.ax_pv.scatter(pos, vel, c=intensity, s=20, cmap='plasma', alpha=0.7)
        
        # Add expansion model lines
        expansion_model_pos = np.linspace(-20, 20, 41)
        expansion_model_vel = 0.3 * np.abs(expansion_model_pos)
        self.ax_pv.plot(expansion_model_pos, expansion_model_vel, 'w--', linewidth=2, 
                       alpha=0.8, label='Expansion Model (+)')
        self.ax_pv.plot(expansion_model_pos, -expansion_model_vel, 'w--', linewidth=2, 
                       alpha=0.8, label='Expansion Model (-)')
        
        self.ax_pv.axhline(0, color='red', linestyle=':', alpha=0.7, label='Systemic Velocity')
        self.ax_pv.axvline(0, color='red', linestyle='-', alpha=0.7, label='Bubble Center')
        
        self.ax_pv.set_title('Position-Velocity Diagram\n(Expansion Signature)', 
                           color='white', fontweight='bold')
        self.ax_pv.set_xlabel('Position (pixels)', color='white')
        self.ax_pv.set_ylabel('Velocity (km/s)', color='white')
        self.ax_pv.legend(fontsize=8)
        self.ax_pv.grid(True, alpha=0.3)
    
    def update_velocity_spectra(self):
        """Update velocity spectra comparison"""
        self.ax_spectra.clear()
        self.ax_spectra.set_facecolor('black')
        
        velocities = np.linspace(-10, 10, 41)
        
        # Center spectrum: narrow profile
        center_spectrum = np.exp(-velocities**2 / 8) + np.random.normal(0, 0.02, len(velocities))
        center_spectrum = np.maximum(center_spectrum, 0)
        
        # Shell spectrum: broader with expansion components
        shell_spectrum = (np.exp(-((velocities - 2) / 2)**2) + 
                         np.exp(-((velocities + 2) / 2)**2) + 
                         np.random.normal(0, 0.02, len(velocities)))
        shell_spectrum = np.maximum(shell_spectrum, 0)
        
        self.ax_spectra.plot(velocities, center_spectrum, 'cyan', linewidth=2, 
                           label='Center Region', alpha=0.8)
        self.ax_spectra.plot(velocities, shell_spectrum, 'red', linewidth=2, 
                           label='Shell Region', alpha=0.8)
        self.ax_spectra.fill_between(velocities, center_spectrum, alpha=0.3, color='cyan')
        self.ax_spectra.fill_between(velocities, shell_spectrum, alpha=0.3, color='red')
        
        self.ax_spectra.axvline(0, color='white', linestyle='--', alpha=0.7, 
                              label='Systemic Velocity')
        
        self.ax_spectra.set_title('Velocity Spectra Comparison\n(Expansion Evidence)', 
                                color='white', fontweight='bold')
        self.ax_spectra.set_xlabel('Velocity (km/s)', color='white')
        self.ax_spectra.set_ylabel('Intensity', color='white')
        self.ax_spectra.legend(fontsize=8)
        self.ax_spectra.grid(True, alpha=0.3)
    
    def update_workflow_status(self):
        """Update workflow status visualization"""
        self.ax_workflow.clear()
        self.ax_workflow.axis('off')
        
        steps = [
            'Data Loading\nPPV Cube',
            'Preprocessing\nNoise Cleaning', 
            'Integration\nIntensity Map',
            'Detection\nMorphological',
            'Analysis\nKinematics',
            'Verification\nMulti-criteria'
        ]
        
        step_colors = ['green', 'green', 'green', 'orange', 'orange', 'red']
        
        # Draw workflow steps
        for i, (step, color) in enumerate(zip(steps, step_colors)):
            x = 0.1 + i * 0.13
            rect = patches.Rectangle((x, 0.3), 0.1, 0.4, linewidth=2, 
                                   edgecolor='white', facecolor=color, alpha=0.7)
            self.ax_workflow.add_patch(rect)
            
            self.ax_workflow.text(x + 0.05, 0.5, step, ha='center', va='center', 
                                fontsize=10, fontweight='bold', color='white')
            
            # Add arrows
            if i < len(steps) - 1:
                self.ax_workflow.annotate('', xy=(x + 0.12, 0.5), xytext=(x + 0.1, 0.5),
                                        arrowprops=dict(arrowstyle='->', color='cyan', lw=2))
        
        self.ax_workflow.set_xlim(0, 1)
        self.ax_workflow.set_ylim(0, 1)
        self.ax_workflow.text(0.5, 0.1, 'Bubble Detection Workflow Pipeline', 
                            ha='center', fontsize=14, fontweight='bold', color='cyan')
    
    def update_statistics(self):
        """Update detection statistics"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        stats = self.detection_results.get('stats', {})
        
        # Calculate statistics
        total_candidates = len(self.detection_results['candidates'])
        confirmed_count = len(self.detection_results['confirmed'])
        rejected_count = len(self.detection_results['rejected'])
        detection_rate = (confirmed_count / max(total_candidates, 1)) * 100
        
        if confirmed_count > 0:
            avg_radius = np.mean([b['radius'] for b in self.detection_results['confirmed']])
            avg_expansion = np.mean([b['expansion_velocity'] for b in self.detection_results['confirmed']])
            avg_confidence = np.mean([b.get('final_confidence', 0.5) for b in self.detection_results['confirmed']])
        else:
            avg_radius = 0
            avg_expansion = 0
            avg_confidence = 0
        
        # Display statistics
        stat_text = f"""
        DETECTION STATISTICS
        ══════════════════════
        
        Total Candidates:     {total_candidates}
        Confirmed Bubbles:    {confirmed_count}
        Rejected Candidates:  {rejected_count}
        Detection Rate:       {detection_rate:.1f}%
        
        Average Radius:       {avg_radius:.1f} px ({avg_radius*0.1:.1f} pc)
        Average Expansion:    {avg_expansion:.2f} km/s  
        Average Confidence:   {avg_confidence:.1%}
        
        Current Parameters:
        • Noise Threshold:    {self.params['noise_threshold']:.1f} σ
        • Size Range:         {self.params['min_radius']}-{self.params['max_radius']} px
        • Min Expansion:      {self.params['min_expansion_vel']:.1f} km/s
        • Min Contrast:       {self.params['min_contrast_ratio']:.1f}
        • Confidence Limit:   {self.params['confidence_threshold']:.1%}
        """
        
        self.ax_stats.text(0.05, 0.95, stat_text, transform=self.ax_stats.transAxes,
                         verticalalignment='top', fontsize=10, fontfamily='monospace',
                         color='white', 
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='darkblue', alpha=0.8))
    
    def update_results_display(self):
        """Update detection results display"""
        self.ax_results.clear()
        self.ax_results.axis('off')
        
        if not (self.detection_results['confirmed'] or self.detection_results['rejected']):
            self.ax_results.text(0.5, 0.5, 'No detection results yet.\nClick "Run Detection" to start.', 
                               ha='center', va='center', fontsize=12, color='gray',
                               transform=self.ax_results.transAxes)
            return
        
        results_text = "DETECTION RESULTS\n" + "="*50 + "\n\n"
        
        # Confirmed bubbles
        if self.detection_results['confirmed']:
            results_text += f"✅ CONFIRMED BUBBLES ({len(self.detection_results['confirmed'])}):\n"
            for bubble in self.detection_results['confirmed']:
                conf = bubble.get('final_confidence', 0.5) * 100
                results_text += (f"  Bubble #{bubble['id']}: ({bubble['center_x']:.0f},{bubble['center_y']:.0f}) "
                               f"R={bubble['radius']}px, V={bubble['expansion_velocity']:.1f}km/s, "
                               f"C={conf:.0f}%\n")
            results_text += "\n"
        
        # Rejected candidates
        if self.detection_results['rejected']:
            results_text += f"❌ REJECTED CANDIDATES ({len(self.detection_results['rejected'])}):\n"
            for bubble in self.detection_results['rejected']:
                conf = bubble.get('final_confidence', 0.5) * 100
                results_text += (f"  Cand. #{bubble['id']}: ({bubble['center_x']:.0f},{bubble['center_y']:.0f}) "
                               f"R={bubble['radius']}px, C={conf:.0f}% "
                               f"[{', '.join(bubble.get('failed_criteria', ['Unknown'])[:2])}]\n")
        
        self.ax_results.text(0.05, 0.95, results_text, transform=self.ax_results.transAxes,
                           verticalalignment='top', fontsize=9, fontfamily='monospace',
                           color='white',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='darkgreen', alpha=0.8))
    
    def run_detection(self, event):
        """Run bubble detection with current parameters"""
        print("🔍 Running bubble detection...")
        
        # Reset results
        self.detection_results['confirmed'] = []
        self.detection_results['rejected'] = []
        
        # Apply detection criteria
        for candidate in self.detection_results['candidates']:
            score = 0
            max_score = 6
            passed_criteria = []
            failed_criteria = []
            
            # Criterion 1: Size range
            if self.params['min_radius'] <= candidate['radius'] <= self.params['max_radius']:
                score += 1
                passed_criteria.append("Size OK")
            else:
                failed_criteria.append("Size range")
            
            # Criterion 2: Contrast ratio
            if candidate['contrast_ratio'] >= self.params['min_contrast_ratio']:
                score += 1
                passed_criteria.append("Contrast OK")
            else:
                failed_criteria.append("Low contrast")
            
            # Criterion 3: Expansion velocity
            if candidate['expansion_velocity'] >= self.params['min_expansion_vel']:
                score += 1
                passed_criteria.append("Expansion OK")
            else:
                failed_criteria.append("No expansion")
            
            # Criterion 4: Morphological coherence (simulated)
            if np.random.random() > 0.3:  # 70% pass rate
                score += 1
                passed_criteria.append("Morphology OK")
            else:
                failed_criteria.append("Poor morphology")
            
            # Criterion 5: Kinematic coherence (simulated)
            if np.random.random() > 0.4:  # 60% pass rate
                score += 1
                passed_criteria.append("Kinematics OK")
            else:
                failed_criteria.append("Poor kinematics")
            
            # Criterion 6: Statistical significance (simulated)
            if candidate['confidence'] > 0.3:
                score += 1
                passed_criteria.append("Significant")
            else:
                failed_criteria.append("Not significant")
            
            # Calculate final confidence
            final_confidence = score / max_score
            candidate['final_confidence'] = final_confidence
            candidate['passed_criteria'] = passed_criteria
            candidate['failed_criteria'] = failed_criteria
            
            # Decision
            if final_confidence >= self.params['confidence_threshold']:
                self.detection_results['confirmed'].append(candidate)
            else:
                self.detection_results['rejected'].append(candidate)
        
        print(f"✅ Detection complete: {len(self.detection_results['confirmed'])} confirmed, "
              f"{len(self.detection_results['rejected'])} rejected")
        
        # Update visualization
        self.update_visualization()
    
    def reset_detection(self, event):
        """Reset detection results"""
        print("🔄 Resetting detection...")
        self.detection_results['confirmed'] = []
        self.detection_results['rejected'] = []
        self.update_visualization()
    
    def update_visualization(self):
        """Update all visualization components"""
        self.update_intensity_map()
        self.update_radial_profile()
        self.update_pv_diagram()
        self.update_velocity_spectra()
        self.update_workflow_status()
        self.update_statistics()
        self.update_results_display()
        
        # Refresh the figure
        self.fig.canvas.draw()
    
    def show(self):
        """Display the interactive visualization"""
        plt.show()

def main():
    """Main function to run the bubble detection visualizer"""
    print("🚀 Starting Bubble Detection Visualization Tool...")
    print("📊 Generating synthetic 13CO PPV data...")
    print("🎛️  Setting up interactive controls...")
    print("🔍 Ready for bubble detection!")
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("1. Adjust parameters using the sliders")
    print("2. Click 'Run Detection' to process")
    print("3. Observe results in real-time")
    print("4. Click 'Reset' to clear results")
    print("="*60 + "\n")
    
    # Create and show the visualizer
    visualizer = BubbleDetectionVisualizer()
    visualizer.show()

if __name__ == "__main__":
    main()

# Additional utility functions for extended analysis
def export_detection_results(results, filename="bubble_detection_results.txt"):
    """Export detection results to file"""
    with open(filename, 'w') as f:
        f.write("BUBBLE DETECTION RESULTS\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Total Candidates: {len(results['candidates'])}\n")
        f.write(f"Confirmed Bubbles: {len(results['confirmed'])}\n")
        f.write(f"Rejected Candidates: {len(results['rejected'])}\n")
        f.write(f"Detection Rate: {len(results['confirmed'])/len(results['candidates'])*100:.1f}%\n\n")
        
        if results['confirmed']:
            f.write("CONFIRMED BUBBLES:\n")
            f.write("-" * 30 + "\n")
            for bubble in results['confirmed']:
                f.write(f"Bubble #{bubble['id']}:\n")
                f.write(f"  Position: ({bubble['center_x']:.1f}, {bubble['center_y']:.1f})\n")
                f.write(f"  Radius: {bubble['radius']} px ({bubble['radius']*0.1:.1f} pc)\n")
                f.write(f"  Expansion: {bubble['expansion_velocity']:.2f} km/s\n")
                f.write(f"  Confidence: {bubble['final_confidence']:.1%}\n")
                f.write(f"  Criteria: {', '.join(bubble['passed_criteria'])}\n\n")
    
    print(f"Results exported to {filename}")

def create_parameter_study(param_name, param_range, base_params):
    """
    Perform parameter study for bubble detection
    
    Parameters:
    -----------
    param_name : str
        Name of parameter to vary
    param_range : array-like
        Range of parameter values to test
    base_params : dict
        Base parameter set
    
    Returns:
    --------
    study_results : dict
        Results of parameter study
    """
    study_results = {
        'parameter': param_name,
        'values': param_range,
        'detection_rates': [],
        'confirmed_counts': [],
        'avg_confidences': []
    }
    
    print(f"🔬 Running parameter study for {param_name}...")
    
    # Create base visualizer for testing
    viz = BubbleDetectionVisualizer()
    
    for param_value in param_range:
        # Update parameter
        viz.params[param_name] = param_value
        
        # Run detection
        viz.run_detection(None)
        
        # Collect results
        confirmed_count = len(viz.detection_results['confirmed'])
        total_count = len(viz.detection_results['candidates'])
        detection_rate = confirmed_count / total_count if total_count > 0 else 0
        avg_confidence = np.mean([b['final_confidence'] for b in viz.detection_results['confirmed']]) if confirmed_count > 0 else 0
        
        study_results['detection_rates'].append(detection_rate)
        study_results['confirmed_counts'].append(confirmed_count)
        study_results['avg_confidences'].append(avg_confidence)
        
        print(f"  {param_name} = {param_value}: {confirmed_count} bubbles, {detection_rate:.1%} rate")
    
    return study_results

def plot_parameter_study(study_results):
    """Plot results of parameter study"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    param_values = study_results['values']
    
    # Detection rate
    axes[0].plot(param_values, study_results['detection_rates'], 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel(f"{study_results['parameter']}")
    axes[0].set_ylabel('Detection Rate')
    axes[0].set_title('Detection Rate vs Parameter')
    axes[0].grid(True, alpha=0.3)
    
    # Number of confirmed bubbles
    axes[1].plot(param_values, study_results['confirmed_counts'], 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel(f"{study_results['parameter']}")
    axes[1].set_ylabel('Confirmed Bubbles')
    axes[1].set_title('Confirmed Count vs Parameter')
    axes[1].grid(True, alpha=0.3)
    
    # Average confidence
    axes[2].plot(param_values, study_results['avg_confidences'], 'go-', linewidth=2, markersize=8)
    axes[2].set_xlabel(f"{study_results['parameter']}")
    axes[2].set_ylabel('Average Confidence')
    axes[2].set_title('Avg Confidence vs Parameter')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Parameter Study: {study_results["parameter"]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Example parameter study usage:
"""
# Run parameter study
study = create_parameter_study('confidence_threshold', 
                               np.linspace(0.3, 0.9, 7), 
                               visualizer.params)
plot_parameter_study(study)
"""