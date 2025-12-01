"""
Advanced MatKit patch function tests with quad elements and displacement fields.
"""
import sys
sys.path.insert(0, '/home/mirza/python/MatKit')

import numpy as np
import matplotlib.pyplot as plt
from matkit import patch

print("=" * 60)
print("Advanced Patch Tests - Quad Elements & Displacement Fields")
print("=" * 60)

# Define mesh geometry
P = np.array([[0, 0],
              [3, 0],
              [3, 1.2],
              [0, 1.2],
              [0.7, 0],
              [1.6, 0],
              [2.5, 0],
              [3, 0.5],
              [2.3, 1.2],
              [1.5, 1.2],
              [0.6, 1.2],
              [0, 0.5],
              [0.55, 0.4],
              [1.45, 0.6],
              [2.4, 0.45]])

nodes = np.array([[1, 5, 13, 12],
                  [12, 13, 11, 4],
                  [5, 6, 14, 13],
                  [13, 14, 10, 11],
                  [14, 15, 9, 10],
                  [6, 7, 15, 14],
                  [7, 2, 8, 15],
                  [15, 8, 3, 9]])

# Test 8: Quad element mesh with node and element labels
print("\nTest 8: Quad element mesh with labels")
fig, ax = plt.subplots(figsize=(10, 6))

# Draw mesh with cyan faces and semi-transparent edges
patch('Faces', nodes, 'Vertices', P,
      'FaceColor', 'cyan',
      'EdgeColor', 'black',
      'EdgeAlpha', 0.3,
      'LineWidth', 1.5,
      ax=ax)

# Plot nodes
ax.plot(P[:, 0], P[:, 1], 'ok', markerfacecolor='black', markersize=16)

# Label nodes with white text
for i in range(len(P)):
    ax.text(P[i, 0] - 0, P[i, 1], str(i+1),
            color='w', fontsize=8, ha='center', va='center')

# Label elements at their centroids
for iel in range(len(nodes)):
    # Get nodes for this element (convert to 0-based)
    element_nodes = nodes[iel] - 1
    # Compute centroid
    xm = np.mean(P[element_nodes, 0])
    ym = np.mean(P[element_nodes, 1])
    ax.text(xm, ym, str(iel+1), fontsize=9, ha='center', va='center')

ax.axis('equal')
ax.set_title('Test 8: Quad Element Mesh with Labels')
plt.savefig('/home/mirza/python/test_patch_quad_mesh.png', dpi=150, bbox_inches='tight')
print("✓ Saved: test_patch_quad_mesh.png")
plt.close()

# Test 9: Displacement field visualization
print("\nTest 9: Displacement field (original + deformed)")

# Create synthetic displacement field (simulating FEM results)
# u is the DOF vector: [u1x, u1y, u2x, u2y, ..., u15x, u15y]
# Create a more interesting displacement pattern
np.random.seed(42)
u_flat = np.zeros(2 * len(P))

# Apply some displacement pattern (e.g., bending)
for i in range(len(P)):
    x, y = P[i]
    # Simulate bending: displacement increases with x, varies with y
    u_flat[2*i] = 0.15 * x * (1 + 0.3 * y)  # x-displacement
    u_flat[2*i+1] = -0.05 * x**2 / 9  # y-displacement (downward)

# Convert to nodal displacement array (n_nodes, 2)
U = np.column_stack([u_flat[0::2], u_flat[1::2]])

# Compute displacement magnitude at each node
UR = np.sqrt(np.sum(U**2, axis=1))

# Scale factor for visualization
scale = 1.0

fig, ax = plt.subplots(figsize=(10, 6))

# Draw original mesh (white faces, semi-transparent edges)
# patch('Faces', nodes, 'Vertices', P,
#       'FaceColor', 'white',
#       'EdgeColor', 'gray',
#       'EdgeAlpha', 0.5,
#       'LineWidth', 1.0,
#       ax=ax)

# Draw deformed mesh with interpolated colors based on displacement magnitude
patch('Faces', nodes, 'Vertices', P + U * scale,
      'FaceVertexCData', UR,
      'FaceColor', 'interp',
      'EdgeColor', 'black',
      'EdgeAlpha', 0.5,
      'LineWidth', 1.0,
      'cmap', 'jet',
      ax=ax)

ax.axis('equal')
ax.set_title(f'Test 9: Displacement Field, scale: {scale}')
plt.colorbar(ax.collections[0], ax=ax, label='Displacement Magnitude')
plt.savefig('/home/mirza/python/test_patch_displacement.png', dpi=150, bbox_inches='tight')
print("✓ Saved: .png")
print(f"  Displacement range: {UR.min():.4f} to {UR.max():.4f}")
plt.close()

# Test 10: Per-element colors (stress field)
print("\nTest 10: Element stress field")

# Create synthetic stress data (one value per element)
element_stresses = np.array([150, -80, 200, -120, 180, 90, -150, 110])

fig, ax = plt.subplots(figsize=(10, 6))

# Draw mesh with per-element colors
patch('Faces', nodes, 'Vertices', P,
      'FaceVertexCData', element_stresses,
      'FaceColor', 'flat',
      'EdgeColor', 'black',
      'LineWidth', 2,
      'cmap', 'RdBu_r',  # Red for tension, blue for compression
      ax=ax)

# Plot nodes
ax.plot(P[:, 0], P[:, 1], 'ok', markerfacecolor='yellow',
        markeredgecolor='black', markersize=10)

ax.axis('equal')
ax.set_title('Test 10: Element Stress Field (Flat Colors)')
plt.colorbar(ax.collections[0], ax=ax, label='Stress (MPa)')
plt.savefig('/home/mirza/python/test_patch_stress.png', dpi=150, bbox_inches='tight')
print("✓ Saved: test_patch_stress.png")
print(f"  Stress range: {element_stresses.min():.1f} to {element_stresses.max():.1f} MPa")
plt.close()

# Test 11: Temperature field (per-vertex interpolated)
print("\nTest 11: Temperature field (per-vertex interpolated)")

# Create synthetic temperature data (one value per node)
# Simulate heat source on right side
node_temps = np.zeros(len(P))
for i in range(len(P)):
    x, y = P[i]
    # Temperature decreases with distance from right edge
    node_temps[i] = 100 * (x / 3.0) + 20

fig, ax = plt.subplots(figsize=(10, 6))

# Draw mesh with interpolated nodal temperatures
patch('Faces', nodes, 'Vertices', P,
      'FaceVertexCData', node_temps,
      'FaceColor', 'interp',
      'EdgeColor', 'black',
      'LineWidth', 1.5,
      'cmap', 'hot',
      ax=ax)

# Plot nodes
ax.plot(P[:, 0], P[:, 1], 'ok', markerfacecolor='cyan',
        markeredgecolor='black', markersize=10)

ax.axis('equal')
ax.set_title('Test 11: Temperature Field (Interpolated Colors)')
plt.colorbar(ax.collections[0], ax=ax, label='Temperature (°C)')
plt.savefig('/home/mirza/python/test_patch_temperature.png', dpi=150, bbox_inches='tight')
print("✓ Saved: test_patch_temperature.png")
print(f"  Temperature range: {node_temps.min():.1f} to {node_temps.max():.1f} °C")
plt.close()

# Test 12: Multiple displacement scales (comparison)
print("\nTest 12: Displacement comparison (multiple scales)")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

collections = []
for idx, scale in enumerate([0.5, 1.0, 2.0]):
    ax = axes[idx]

    # Original mesh
    patch('Faces', nodes, 'Vertices', P,
          'FaceColor', 'white',
          'EdgeColor', 'gray',
          'EdgeAlpha', 0.3,
          'LineWidth', 1.0,
          ax=ax)

    # Deformed mesh
    P_def = P + U * scale
    pc = patch('Faces', nodes, 'Vertices', P_def,
               'FaceVertexCData', UR,
               'FaceColor', 'interp',
               'EdgeColor', 'black',
               'EdgeAlpha', 0.2,
               'LineWidth', 1.5,
               'cmap', 'viridis',
               ax=ax)
    collections.append(pc)

    ax.axis('equal')
    ax.set_title(f'Scale: {scale}')

# Add shared colorbar
fig.colorbar(collections[0], ax=axes.ravel().tolist(), label='Displacement Magnitude', pad=0.02)

plt.suptitle('Test 12: Displacement Field at Different Scales', fontsize=14)
plt.tight_layout()
plt.savefig('/home/mirza/python/test_patch_scales.png', dpi=150, bbox_inches='tight')
print("✓ Saved: test_patch_scales.png")
plt.close()

# Test 13: Transparency variations
print("\nTest 13: FaceAlpha variations")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, alpha in enumerate([0.3, 0.6, 1.0]):
    ax = axes[idx]

    patch('Faces', nodes, 'Vertices', P,
          'FaceVertexCData', element_stresses,
          'FaceColor', 'flat',
          'FaceAlpha', alpha,
          'EdgeColor', 'black',
          'LineWidth', 2,
          'cmap', 'plasma',
          ax=ax)

    ax.plot(P[:, 0], P[:, 1], 'ok', markerfacecolor='white',
            markeredgecolor='black', markersize=8)

    ax.axis('equal')
    ax.set_title(f'FaceAlpha: {alpha}')

plt.suptitle('Test 13: Face Transparency Variations', fontsize=14)
plt.tight_layout()
plt.savefig('/home/mirza/python/test_patch_transparency.png', dpi=150, bbox_inches='tight')
print("✓ Saved: test_patch_transparency.png")
plt.close()

print("\n" + "=" * 60)
print("Advanced tests completed successfully!")
print("=" * 60)
print("\nGenerated files:")
print("  - test_patch_quad_mesh.png (mesh with labels)")
print("  - test_patch_displacement.png (displacement field)")
print("  - test_patch_stress.png (per-element stress)")
print("  - test_patch_temperature.png (per-vertex temperature)")
print("  - test_patch_scales.png (displacement comparison)")
print("  - test_patch_transparency.png (alpha variations)")
