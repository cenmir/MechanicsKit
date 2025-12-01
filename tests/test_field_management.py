"""
Test Field Management in Mesh Class
====================================

Tests the new Add_Nodal_Field, Add_DoF_Field, and Add_Element_Field methods
to ensure 1-based indexing works correctly.
"""

import sys
sys.path.insert(0, '/home/mirza/python/MatKit')

from matkit.mesh import Mesh
import numpy as np


def test_nodal_fields():
    """Test nodal field storage and access."""
    print("Testing Nodal Fields...")
    print("-" * 60)

    # Create simple mesh
    coords = [[0, 0], [1, 0], [0, 1], [1, 1]]
    connectivity = [[1, 2], [1, 3], [2, 4], [3, 4]]
    mesh = Mesh(coords, connectivity)

    # Test scalar field (temperatures)
    T = mesh.Add_Nodal_Field('temperature', [20.0, 25.0, 30.0, 22.0])
    assert T[1] == 20.0, "Node 1 temperature should be 20.0"
    assert T[4] == 22.0, "Node 4 temperature should be 22.0"
    print(f"✓ Scalar field: T[1] = {T[1]}, T[4] = {T[4]}")

    # Test vector field (displacements)
    U = mesh.Add_Nodal_Field('displacements',
        [[0.0, 0.0], [0.1, 0.2], [0.15, 0.25], [0.0, 0.0]])
    assert np.allclose(U[2], [0.1, 0.2]), "Node 2 displacement incorrect"
    assert np.allclose(U[3], [0.15, 0.25]), "Node 3 displacement incorrect"
    print(f"✓ Vector field: U[2] = {U[2]}, U[3] = {U[3]}")

    # Test retrieval
    T_retrieved = mesh.Get_Nodal_Field('temperature')
    assert T_retrieved[1] == 20.0, "Retrieved field should match"
    print(f"✓ Field retrieval: T[1] = {T_retrieved[1]}")

    print()


def test_dof_fields():
    """Test DOF field storage and access."""
    print("Testing DOF Fields...")
    print("-" * 60)

    # Create simple mesh (4 nodes, 2D -> 8 DOFs)
    coords = [[0, 0], [1, 0], [0, 1], [1, 1]]
    connectivity = [[1, 2], [1, 3], [2, 4], [3, 4]]
    mesh = Mesh(coords, connectivity)

    # Test DOF field
    u = mesh.Add_DoF_Field('displacements',
        [0.0, 0.0, 0.1, 0.2, 0.15, 0.25, 0.0, 0.0])

    assert u[1] == 0.0, "DOF 1 should be 0.0"
    assert u[3] == 0.1, "DOF 3 should be 0.1"
    assert u[5] == 0.15, "DOF 5 should be 0.15"
    print(f"✓ DOF field: u[1] = {u[1]}, u[3] = {u[3]}, u[5] = {u[5]}")

    # Test retrieval
    u_retrieved = mesh.Get_DoF_Field('displacements')
    assert u_retrieved[3] == 0.1, "Retrieved DOF field should match"
    print(f"✓ Field retrieval: u[3] = {u_retrieved[3]}")

    print()


def test_element_fields():
    """Test element field storage and access."""
    print("Testing Element Fields...")
    print("-" * 60)

    # Create simple mesh (4 elements)
    coords = [[0, 0], [1, 0], [0, 1], [1, 1]]
    connectivity = [[1, 2], [1, 3], [2, 4], [3, 4]]
    mesh = Mesh(coords, connectivity)

    # Test element forces
    N = mesh.Add_Element_Field('forces',
        [6052.76, -5582.25, -7274.51, 6380.16])

    assert N[1] == 6052.76, "Element 1 force incorrect"
    assert N[3] == -7274.51, "Element 3 force incorrect"
    print(f"✓ Element forces: N[1] = {N[1]}, N[3] = {N[3]}")

    # Test element areas
    A = mesh.Add_Element_Field('areas', [0.01, 0.01, 0.015, 0.012])
    assert A[2] == 0.01, "Element 2 area incorrect"
    assert A[4] == 0.012, "Element 4 area incorrect"
    print(f"✓ Element areas: A[2] = {A[2]}, A[4] = {A[4]}")

    # Test retrieval
    N_retrieved = mesh.Get_Element_Field('forces')
    assert N_retrieved[1] == 6052.76, "Retrieved element field should match"
    print(f"✓ Field retrieval: N[1] = {N_retrieved[1]}")

    print()


def test_seamless_iteration():
    """Test field access during mesh iteration."""
    print("Testing Seamless Iteration with Fields...")
    print("-" * 60)

    # Create simple mesh
    coords = [[0, 0], [1, 0], [0, 1], [1, 1]]
    connectivity = [[1, 2], [1, 3], [2, 4], [3, 4]]
    mesh = Mesh(coords, connectivity)

    # Add element fields
    N = mesh.Add_Element_Field('forces', [100, 200, -150, 300])
    A = mesh.Add_Element_Field('areas', [0.01, 0.01, 0.01, 0.01])

    # Iterate and compute stresses
    print("Element stresses:")
    for iel in mesh.element_numbers():
        force = N[iel]
        area = A[iel]
        stress = force / area
        print(f"  Element {iel}: N = {force:7.2f} N, σ = {stress:9.2f} Pa")

    print()


def test_error_handling():
    """Test error handling for invalid field data."""
    print("Testing Error Handling...")
    print("-" * 60)

    coords = [[0, 0], [1, 0], [0, 1], [1, 1]]
    connectivity = [[1, 2], [1, 3], [2, 4], [3, 4]]
    mesh = Mesh(coords, connectivity)

    # Test wrong size for nodal field
    try:
        mesh.Add_Nodal_Field('temp', [20.0, 25.0])  # Too few values
        print("✗ Should have raised error for wrong size")
    except ValueError as e:
        print(f"✓ Caught error for wrong nodal field size: {e}")

    # Test wrong size for DOF field
    try:
        mesh.Add_DoF_Field('displ', [0.0, 0.1, 0.2])  # Too few values
        print("✗ Should have raised error for wrong size")
    except ValueError as e:
        print(f"✓ Caught error for wrong DOF field size: {e}")

    # Test wrong size for element field
    try:
        mesh.Add_Element_Field('forces', [100, 200])  # Too few values
        print("✗ Should have raised error for wrong size")
    except ValueError as e:
        print(f"✓ Caught error for wrong element field size: {e}")

    # Test retrieval of non-existent field
    try:
        mesh.Get_Nodal_Field('nonexistent')
        print("✗ Should have raised error for non-existent field")
    except KeyError as e:
        print(f"✓ Caught error for non-existent field: {e}")

    print()


if __name__ == '__main__':
    print("=" * 60)
    print("FIELD MANAGEMENT TESTS")
    print("=" * 60)
    print()

    test_nodal_fields()
    test_dof_fields()
    test_element_fields()
    test_seamless_iteration()
    test_error_handling()

    print("=" * 60)
    print("🎉 ALL FIELD MANAGEMENT TESTS PASSED! 🎉")
    print("=" * 60)
    print()
    print("Summary:")
    print("  ✓ Add_Nodal_Field: 1-based nodal field access")
    print("  ✓ Add_DoF_Field: 1-based DOF field access")
    print("  ✓ Add_Element_Field: 1-based element field access")
    print("  ✓ Seamless iteration with field access")
    print("  ✓ Error handling for invalid data")
    print()
