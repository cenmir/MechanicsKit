"""
Test MatKit patch function with truss examples.
"""
import sys
sys.path.insert(0, '/home/mirza/python/MatKit')

import numpy as np
import matplotlib.pyplot as plt
from matkit import patch

# Node coordinates (mm)
P = np.array([[0, 0],
              [500, 0],
              [300, 300],
              [600, 300]])

# Element connectivity (1-based node numbers)
edges = np.array([[1, 2],
                  [1, 3],
                  [2, 3],
                  [2, 4],
                  [3, 4]])

print("=" * 60)
print("Testing MatKit patch() function")
print("=" * 60)

# Test 1: Basic truss visualization
print("\nTest 1: Basic truss visualization (uniform color)")
fig, ax = plt.subplots(figsize=(8, 6))
patch('Faces', edges, 'Vertices', P, 'LineWidth', 2)
ax.plot(P[:, 0], P[:, 1], 'o', color='cyan',
        markeredgecolor='black', markersize=14)
ax.axis('equal')
ax.axis('off')
ax.set_title('Test 1: Basic Truss')
plt.savefig('/home/mirza/python/test_patch_basic.png', dpi=150, bbox_inches='tight')
print("✓ Saved: test_patch_basic.png")
plt.close()

# Test 2: Per-element colors (flat mode)
print("\nTest 2: Per-element colors (flat mode)")
forces = np.array([6052.76, -5582.25, -7274.51, 6380.16, -9912.07])

fig, ax = plt.subplots(figsize=(8, 6))
patch('Faces', edges, 'Vertices', P,
      'FaceVertexCData', forces,
      'FaceColor', 'flat',
      'LineWidth', 3,
      'cmap', 'RdBu_r')  # Red for tension, blue for compression
ax.plot(P[:, 0], P[:, 1], 'o', color='cyan',
        markeredgecolor='black', markersize=14)
ax.axis('equal')
ax.axis('off')
ax.set_title('Test 2: Element Forces (Flat Colors)')
plt.colorbar(ax.collections[0], ax=ax, label='Force (N)')
plt.savefig('/home/mirza/python/test_patch_flat.png', dpi=150, bbox_inches='tight')
print("✓ Saved: test_patch_flat.png")
print(f"  Force range: {forces.min():.1f} to {forces.max():.1f} N")
plt.close()

# Test 3: Per-vertex colors (interpolated mode)
print("\nTest 3: Per-vertex colors (interpolated mode)")
node_temps = np.array([20.0, 25.0, 30.0, 22.0])  # Temperature at each node

fig, ax = plt.subplots(figsize=(8, 6))
patch('Faces', edges, 'Vertices', P,
      'FaceVertexCData', node_temps,
      'FaceColor', 'interp',
      'LineWidth', 3,
      'cmap', 'hot')
ax.plot(P[:, 0], P[:, 1], 'o', color='cyan',
        markeredgecolor='black', markersize=14)
ax.axis('equal')
ax.axis('off')
ax.set_title('Test 3: Nodal Temperatures (Interpolated Colors)')
plt.colorbar(ax.collections[0], ax=ax, label='Temperature (°C)')
plt.savefig('/home/mirza/python/test_patch_interp.png', dpi=150, bbox_inches='tight')
print("✓ Saved: test_patch_interp.png")
print(f"  Temperature range: {node_temps.min():.1f} to {node_temps.max():.1f} °C")
plt.close()

# Test 4: Transparency test
print("\nTest 4: Transparency (FaceAlpha)")
fig, ax = plt.subplots(figsize=(8, 6))
patch('Faces', edges, 'Vertices', P,
      'FaceVertexCData', forces,
      'FaceColor', 'flat',
      'LineWidth', 4,
      'FaceAlpha', 0.5,
      'cmap', 'viridis')
ax.plot(P[:, 0], P[:, 1], 'o', color='red',
        markeredgecolor='black', markersize=14)
ax.axis('equal')
ax.axis('off')
ax.set_title('Test 4: Semi-Transparent Elements (alpha=0.5)')
plt.savefig('/home/mirza/python/test_patch_alpha.png', dpi=150, bbox_inches='tight')
print("✓ Saved: test_patch_alpha.png")
plt.close()

# Test 5: 3D line elements
print("\nTest 5: 3D truss")
P_3d = np.array([[0, 0, 0],
                 [500, 0, 0],
                 [300, 300, 0],
                 [600, 300, 0],
                 [300, 150, 400]])

edges_3d = np.array([[1, 2],
                     [1, 3],
                     [2, 3],
                     [2, 4],
                     [3, 4],
                     [1, 5],
                     [2, 5],
                     [3, 5],
                     [4, 5]])

forces_3d = np.array([100, -80, 50, -120, 90, -150, 110, -95, 130])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
patch('Faces', edges_3d, 'Vertices', P_3d,
      'FaceVertexCData', forces_3d,
      'LineWidth', 2,
      'cmap', 'coolwarm',
      ax=ax)
ax.scatter(P_3d[:, 0], P_3d[:, 1], P_3d[:, 2],
          color='yellow', edgecolor='black', s=100)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Test 5: 3D Truss with Element Forces')
plt.savefig('/home/mirza/python/test_patch_3d.png', dpi=150, bbox_inches='tight')
print("✓ Saved: test_patch_3d.png")
plt.close()

# Test 6: 2D Surface elements (triangles)
print("\nTest 6: 2D Surface elements (triangles)")
vertices_2d = np.array([[0, 0],
                        [1, 0],
                        [0.5, 0.866],
                        [1.5, 0.866],
                        [1, 1.732]])

faces_2d = np.array([[1, 2, 3],
                     [2, 4, 3],
                     [3, 4, 5]])

face_colors = np.array([0.2, 0.5, 0.8])

fig, ax = plt.subplots(figsize=(8, 6))
patch('Faces', faces_2d, 'Vertices', vertices_2d,
      'FaceVertexCData', face_colors,
      'FaceColor', 'flat',
      'EdgeColor', 'black',
      'LineWidth', 2,
      'cmap', 'plasma')
ax.plot(vertices_2d[:, 0], vertices_2d[:, 1], 'ko', markersize=8)
ax.set_aspect('equal')
ax.set_title('Test 6: 2D Triangular Elements')
plt.colorbar(ax.collections[0], ax=ax, label='Element Value')
plt.savefig('/home/mirza/python/test_patch_2d_surface.png', dpi=150, bbox_inches='tight')
print("✓ Saved: test_patch_2d_surface.png")
plt.close()

# Test 7: 3D Surface element (single quad)
print("\nTest 7: 3D Surface element (quad with transparency)")
vertices_3d_surf = np.array([[0, 0, 0],
                             [1, 0, 0],
                             [1, 1, 0],
                             [0, 1, 0]])

faces_3d_surf = np.array([[1, 2, 3, 4]])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
patch('Faces', faces_3d_surf, 'Vertices', vertices_3d_surf,
      'FaceColor', 'red',
      'FaceAlpha', 0.6,
      'EdgeColor', 'black',
      'LineWidth', 2,
      ax=ax)
ax.scatter(vertices_3d_surf[:, 0], vertices_3d_surf[:, 1], vertices_3d_surf[:, 2],
          color='blue', s=100)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Test 7: 3D Quad Surface (Transparent)')
plt.savefig('/home/mirza/python/test_patch_3d_surface.png', dpi=150, bbox_inches='tight')
print("✓ Saved: test_patch_3d_surface.png")
plt.close()

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
print("\nGenerated files:")
print("  - test_patch_basic.png")
print("  - test_patch_flat.png")
print("  - test_patch_interp.png")
print("  - test_patch_alpha.png")
print("  - test_patch_3d.png")
print("  - test_patch_2d_surface.png")
print("  - test_patch_3d_surface.png")
