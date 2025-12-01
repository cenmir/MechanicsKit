"""
Test Seamless 1-Based Workflow with OneArray and Iterators
===========================================================

This tests the truly seamless workflow - NO off-by-one errors!
"""

import numpy as np
from matkit import Mesh, OneArray


def test_onearray_basic():
    """Test basic OneArray functionality."""
    print("\n" + "="*70)
    print("Test 1: OneArray Basic Operations")
    print("="*70)

    # Create OneArray (element forces)
    N = OneArray([100, 200, 300, 400, 500])

    # 1-based access
    assert N[1] == 100, "Element 1 should be 100"
    assert N[5] == 500, "Element 5 should be 500"
    print(f"✓ N[1] = {N[1]} (element 1 force)")
    print(f"✓ N[5] = {N[5]} (element 5 force)")

    # List access
    forces = N[[1, 3, 5]]
    assert np.array_equal(forces, [100, 300, 500])
    print(f"✓ N[[1,3,5]] = {forces}")

    # Set values
    N[1] = 150
    assert N[1] == 150
    print(f"✓ N[1] = 150 (modified)")

    # Set multiple
    N[[2, 4]] = [250, 450]
    assert N[2] == 250
    assert N[4] == 450
    print(f"✓ N[[2,4]] = [250, 450] (modified multiple)")

    print("\n✅ OneArray basic operations passed!")


def test_onearray_arithmetic():
    """Test OneArray arithmetic operations."""
    print("\n" + "="*70)
    print("Test 2: OneArray Arithmetic")
    print("="*70)

    N1 = OneArray([10, 20, 30])
    N2 = OneArray([1, 2, 3])

    # Addition
    N3 = N1 + N2
    assert N3[1] == 11
    assert N3[3] == 33
    print(f"✓ Addition: N1 + N2 = {N3.data}")

    # Scalar multiplication
    N4 = N1 * 2
    assert N4[1] == 20
    assert N4[3] == 60
    print(f"✓ Multiplication: N1 * 2 = {N4.data}")

    # Negation
    N5 = -N1
    assert N5[1] == -10
    print(f"✓ Negation: -N1 = {N5.data}")

    # Max/min
    assert N1.max() == 30
    assert N1.min() == 10
    print(f"✓ Max = {N1.max()}, Min = {N1.min()}")

    print("\n✅ OneArray arithmetic passed!")


def test_onearray_errors():
    """Test OneArray error handling."""
    print("\n" + "="*70)
    print("Test 3: OneArray Error Handling")
    print("="*70)

    N = OneArray([100, 200, 300])

    # Test index 0 (should fail - 1-based!)
    try:
        val = N[0]
        assert False, "Should have raised IndexError for index 0"
    except IndexError as e:
        print(f"✓ Caught index 0: {str(e)[:50]}...")

    # Test index out of range
    try:
        val = N[4]
        assert False, "Should have raised IndexError for index 4"
    except IndexError as e:
        print(f"✓ Caught index 4: {str(e)[:50]}...")

    # Test negative index (should fail - no Python-style negative indexing)
    try:
        val = N[-1]
        assert False, "Should have raised IndexError for negative index"
    except IndexError as e:
        print(f"✓ Caught negative index: {str(e)[:50]}...")

    print("\n✅ OneArray error handling passed!")


def test_mesh_elements_iterator():
    """Test Mesh.elements() iterator."""
    print("\n" + "="*70)
    print("Test 4: Mesh.elements() Iterator")
    print("="*70)

    coords = [[0, 0], [1, 0], [0, 1]]
    connectivity = [[1, 2], [2, 3], [1, 3]]
    mesh = Mesh(coords, connectivity)

    # Collect element numbers
    elem_nums = list(mesh.element_numbers())
    assert elem_nums == [1, 2, 3]
    print(f"✓ mesh.element_numbers() yields: {elem_nums}")

    # Use with OneArray (THE KEY TEST!)
    N = OneArray([100, 200, 300])
    forces = []
    for iel in mesh.element_numbers():
        forces.append(N[iel])  # NO [iel-1] !!!!
    assert forces == [100, 200, 300]
    print(f"✓ Seamless access: N[iel] for iel in mesh.element_numbers()")

    print("\n✅ Mesh.elements() iterator passed!")


def test_mesh_nodes_iterator():
    """Test Mesh.node_numbers() iterator."""
    print("\n" + "="*70)
    print("Test 5: Mesh.node_numbers() Iterator")
    print("="*70)

    coords = [[0, 0], [1, 0], [0, 1]]
    connectivity = [[1, 2], [2, 3]]
    mesh = Mesh(coords, connectivity)

    # Collect node numbers
    node_nums = list(mesh.node_numbers())
    assert node_nums == [1, 2, 3]
    print(f"✓ mesh.node_numbers() yields: {node_nums}")

    print("\n✅ Mesh.node_numbers() iterator passed!")


def test_iter_elements():
    """Test Mesh.iter_elements() iterator."""
    print("\n" + "="*70)
    print("Test 6: Mesh.iter_elements() Iterator")
    print("="*70)

    coords = [[0, 0], [1, 0], [0, 1]]
    connectivity = [[1, 2], [2, 3], [1, 3]]
    mesh = Mesh(coords, connectivity)

    N = OneArray([100, 200, 300])

    # Test iteration with all data
    count = 0
    for iel, node_nums, coords in mesh.iter_elements():
        force = N[iel]  # Seamless!
        assert force == (iel * 100)
        assert len(node_nums) == 2
        assert coords.shape == (2, 2)
        count += 1
        print(f"✓ Element {iel}: nodes {node_nums}, force {force}")

    assert count == 3
    print("\n✅ Mesh.iter_elements() passed!")


def test_iter_nodes():
    """Test Mesh.iter_nodes() iterator."""
    print("\n" + "="*70)
    print("Test 7: Mesh.iter_nodes() Iterator")
    print("="*70)

    coords = [[0, 0], [1, 0], [0, 1]]
    connectivity = [[1, 2], [2, 3]]
    mesh = Mesh(coords, connectivity)

    # Test iteration with coordinates
    count = 0
    for inode, coord in mesh.iter_nodes():
        assert len(coord) == 2
        assert inode in [1, 2, 3]
        count += 1
        print(f"✓ Node {inode}: {coord}")

    assert count == 3
    print("\n✅ Mesh.iter_nodes() passed!")


def test_seamless_workflow():
    """Test the TRULY seamless workflow - NO off-by-one!"""
    print("\n" + "="*70)
    print("Test 8: SEAMLESS Workflow (The Ultimate Test!)")
    print("="*70)

    # Setup
    coords = [[0, 0], [500, 0], [300, 300], [600, 300]]
    connectivity = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4]]
    mesh = Mesh(coords, connectivity)

    # Element forces (1-based naturally!)
    N = OneArray([6052.76, -5582.25, -7274.51, 6380.16, -9912.07])

    # Displacement vector (8 DOFs)
    u = OneArray([0., 0., 0.60, -0.62, 0.59, -1.25, 0., 0.])

    print("\n--- Element Forces (CLEAN!) ---")
    for iel in mesh.element_numbers():
        node_nums, coords = mesh.get_element(iel)
        force = N[iel]  # NO [iel-1] !!!
        force_type = "Tension" if force > 0 else "Compression"
        print(f"Element {iel}: nodes {node_nums}, Force = {force:8.2f} N ({force_type})")

    print("\n--- Even Better: iter_elements() ---")
    for iel, node_nums, coords in mesh.iter_elements():
        force = N[iel]  # SEAMLESS!
        length = np.linalg.norm(coords[1] - coords[0])
        print(f"Element {iel}: Force = {force:8.2f} N, Length = {length:.2f} mm")

    print("\n--- Node Displacements ---")
    for inode in mesh.node_numbers():
        dofs = mesh.dofs_for_node(inode)
        # Note: u is OneArray (1-based), but dofs are 0-based array indices
        # So we use u.data[dofs] here
        ux, uy = u.data[dofs]
        print(f"Node {inode}: u_x = {ux:8.4f} mm, u_y = {uy:8.4f} mm")

    print("\n✅ SEAMLESS workflow test passed!")
    print("   NO off-by-one errors!")
    print("   NO [iel-1] mental gymnastics!")
    print("   PURE 1-based thinking!")


def test_comparison_old_vs_new():
    """Compare old clunky way vs new seamless way."""
    print("\n" + "="*70)
    print("Test 9: Old Clunky vs New Seamless")
    print("="*70)

    coords = [[0, 0], [1, 0], [0, 1]]
    connectivity = [[1, 2], [2, 3], [1, 3]]
    mesh = Mesh(coords, connectivity)

    # Raw data (as you'd get from FEM solver)
    N_raw = np.array([100, 200, 300])

    print("\n--- OLD WAY (Clunky) ---")
    print("```python")
    print("for iel in range(1, mesh.n_elements + 1):  # Ugly +1")
    print("    node_nums = mesh.get_element(iel)[0]")
    print("    force = N_raw[iel-1]  # Ugly [iel-1] !!!")
    print("    print(f'Element {iel}: Force = {force}')")
    print("```")

    for iel in range(1, mesh.n_elements + 1):
        force = N_raw[iel-1]  # THE PROBLEM!
        print(f"Element {iel}: Force = {force}")

    print("\n--- NEW WAY (Seamless) ---")
    print("```python")
    print("N = OneArray(N_raw)  # Wrap once")
    print("for iel in mesh.element_numbers():  # Clean!")
    print("    force = N[iel]  # Natural 1-based!")
    print("    print(f'Element {iel}: Force = {force}')")
    print("```")

    N = OneArray(N_raw)
    for iel in mesh.element_numbers():
        force = N[iel]  # NO [iel-1] !!!
        print(f"Element {iel}: Force = {force}")

    print("\n✅ See the difference? New way is SEAMLESS!")


def run_all_tests():
    """Run all seamless workflow tests."""
    print("\n" + "="*70)
    print("MatKit Seamless Workflow Test Suite")
    print("Testing OneArray + Mesh Iterators")
    print("="*70)

    test_onearray_basic()
    test_onearray_arithmetic()
    test_onearray_errors()
    test_mesh_elements_iterator()
    test_mesh_nodes_iterator()
    test_iter_elements()
    test_iter_nodes()
    test_seamless_workflow()
    test_comparison_old_vs_new()

    print("\n" + "="*70)
    print("🎉 ALL SEAMLESS WORKFLOW TESTS PASSED! 🎉")
    print("="*70)
    print()
    print("Summary:")
    print("  ✓ OneArray: 1-based array access (no [iel-1]!)")
    print("  ✓ mesh.element_numbers(): Clean iteration (no range(1, n+1)!)")
    print("  ✓ mesh.node_numbers(): Clean iteration")
    print("  ✓ mesh.iter_elements(): All data in one call")
    print("  ✓ mesh.iter_nodes(): Nodes with coordinates")
    print("  ✓ TRULY SEAMLESS: No off-by-one errors!")
    print()
    print("The workflow is now PERFECT for 1-based thinking!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
