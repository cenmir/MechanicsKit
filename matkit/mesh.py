"""
Mesh Helper for FEM Education
==============================

This module provides a pedagogical interface for finite element analysis
that uses 1-based indexing for mathematical entities (nodes, elements) while
maintaining compatibility with Python's 0-based NumPy arrays.

Philosophy:
-----------
- **Input/Output**: 1-based indexing (matches textbooks and mathematical notation)
- **Internal computation**: 0-based indexing (efficient NumPy operations)
- **Transparency**: Clear helper methods bridge the gap, students understand both systems

This hybrid approach:
1. Reduces cognitive load when translating from textbook notation to code
2. Maintains full compatibility with Python's scientific computing ecosystem
3. Teaches students about abstraction and interface design
4. Prepares them for professional FEM codes (ANSYS, Abaqus) which use 1-based indexing

Why not Julia/MATLAB?
--------------------
While Julia and MATLAB use 1-based indexing natively, Python dominates modern
scientific computing workflows: automation, APIs, data pipelines, web services,
machine learning. Teaching Python + smart abstractions prepares students for
this broader landscape while preserving mathematical naturalness.

Author: Based on the 1-Indexing Manifesto pedagogical framework
License: MIT
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from .one_array import OneArray

# Element type definitions (ANSA-style)
ELEMENT_TYPES = {
    'ROD': {'nodes_per_element': 2, 'description': '1D rod/truss element'},
    'BEAM': {'nodes_per_element': 2, 'description': '1D beam element with rotations'},
    'TRIA3': {'nodes_per_element': 3, 'description': '2D triangular element'},
    'QUAD4': {'nodes_per_element': 4, 'description': '2D quadrilateral element'},
    'TETRA4': {'nodes_per_element': 4, 'description': '3D tetrahedral element'},
    'HEXA8': {'nodes_per_element': 8, 'description': '3D hexahedral element'},
}


class Mesh:
    """
    General FEM mesh with 1-based node/element numbering.

    This class handles the translation between mathematical notation
    (1-based) and Python arrays (0-based) internally, allowing students
    to work with familiar mathematical conventions.

    The Indexing Translation Problem
    ---------------------------------
    In FEM textbooks, nodes are numbered 1, 2, 3, ... and degrees of freedom
    for node i depend on the element type and dimension. This is universal
    mathematical notation dating back decades.

    Python inherited 0-based indexing from C (1972), where it optimized
    pointer arithmetic. This creates a translation burden:
    - Mathematical "node 3" → Python index 2
    - Mathematical DOFs for node 3 → Different Python indices

    This class eliminates that burden by providing 1-based interfaces while
    storing data efficiently as 0-based NumPy arrays internally.

    Element Types
    -------------
    The mesh can represent different element types:
    - ROD: 2-node truss elements (DOFs: 2 or 3 per node based on dimension)
    - BEAM: 2-node beam elements (DOFs: 3 in 2D, 6 in 3D - includes rotations)
    - TRIA3: 3-node triangular elements (2D)
    - QUAD4: 4-node quadrilateral elements (2D)
    - TETRA4: 4-node tetrahedral elements (3D)
    - HEXA8: 8-node hexahedral elements (3D)

    Examples
    --------
    >>> # Natural mathematical notation
    >>> coords = [[0, 0], [1, 0], [0, 1], [1, 1]]
    >>> connectivity = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4]]
    >>> mesh = Mesh(coords, connectivity)  # Auto-detects ROD from 2 nodes
    >>>
    >>> # Or explicit element type
    >>> mesh = Mesh(coords, connectivity, element_type='ROD')
    >>>
    >>> # Access node 3 directly (not index 2!)
    >>> mesh.get_node(3)
    array([0., 1.])
    >>>
    >>> # Get DOF indices for node 3 (returns 0-based for array access)
    >>> dofs = mesh.dofs_for_node(3)  # Returns [4, 5] for 2D ROD
    >>> u = np.zeros(2 * mesh.n_nodes)
    >>> u[dofs] = [0.1, 0.2]  # Set displacement for node 3
    >>>
    >>> # Works with lists too
    >>> coords = mesh.get_node([1, 3])  # Get multiple nodes
    >>> dofs = mesh.dofs_for_node([1, 3])  # Get DOFs for multiple nodes
    """

    def __init__(self, node_coords, element_connectivity,
                 element_type: Optional[str] = None,
                 dofs_per_node: Optional[int] = None):
        """
        Initialize FEM mesh.

        Parameters
        ----------
        node_coords : array-like, shape (n_nodes, n_dim)
            Nodal coordinates. Will be accessible as nodes 1, 2, 3, ...

        element_connectivity : array-like, shape (n_elements, nodes_per_element)
            Element connectivity using 1-BASED node numbers.
            Each row contains node numbers for one element.

        element_type : str, optional
            Element type: 'ROD', 'BEAM', 'TRIA3', 'QUAD4', 'TETRA4', 'HEXA8'
            If None, auto-detects from nodes_per_element:
            - 2 nodes → ROD
            - 3 nodes → TRIA3
            - 4 nodes → QUAD4 (2D) or TETRA4 (3D)
            - 8 nodes → HEXA8

        dofs_per_node : int, optional
            Explicit DOFs per node. If None, determined from element_type:
            - ROD: n_dim (2 for 2D, 3 for 3D)
            - BEAM: 3 (2D) or 6 (3D) - includes rotations
            - TRIA3/QUAD4: 2 (displacement only)
            - TETRA4/HEXA8: 3 (displacement only)

        Example
        -------
        >>> # Define nodes with 1-based thinking
        >>> coords = [[0.0, 0.0],    # This is node 1
        ...           [1.0, 0.0],    # This is node 2
        ...           [0.0, 1.0],    # This is node 3
        ...           [1.0, 1.0]]    # This is node 4
        >>>
        >>> # Define connectivity with 1-based node numbers
        >>> elements = [[1, 2],  # Element 1 connects nodes 1-2
        ...             [1, 3],  # Element 2 connects nodes 1-3
        ...             [2, 3]]  # Element 3 connects nodes 2-3
        >>>
        >>> mesh = Mesh(coords, elements)  # Auto-detects ROD
        """
        # Store coordinates as-is (will use 0-indexed access internally)
        self.nodes = np.array(node_coords, dtype=float)
        self.n_nodes = len(self.nodes)
        self.n_dim = self.nodes.shape[1]

        # Convert element connectivity from 1-based to 0-based for internal use
        # This is THE critical translation: subtract 1 from user's 1-based input
        self._elements_0based = np.array(element_connectivity, dtype=int) - 1

        # Validate connectivity
        if np.any(self._elements_0based < 0):
            raise ValueError(
                "Element connectivity must use 1-based node numbers. "
                "Did you accidentally use 0-based indexing?"
            )
        if np.any(self._elements_0based >= self.n_nodes):
            raise ValueError(f"Element references non-existent node")

        self.n_elements = len(self._elements_0based)
        self.nodes_per_element = self._elements_0based.shape[1]

        # Auto-detect or validate element type
        if element_type is None:
            self.element_type = self._auto_detect_element_type()
        else:
            if element_type not in ELEMENT_TYPES:
                raise ValueError(
                    f"Unknown element type: {element_type}. "
                    f"Valid types: {list(ELEMENT_TYPES.keys())}"
                )
            expected_nodes = ELEMENT_TYPES[element_type]['nodes_per_element']
            if self.nodes_per_element != expected_nodes:
                raise ValueError(
                    f"Element type '{element_type}' expects {expected_nodes} nodes, "
                    f"but connectivity has {self.nodes_per_element} nodes per element"
                )
            self.element_type = element_type

        # Determine DOFs per node
        if dofs_per_node is not None:
            self.dofs_per_node = dofs_per_node
        else:
            self.dofs_per_node = self._determine_dofs_per_node()

        self.n_dofs = self.n_nodes * self.dofs_per_node

        # Field storage (for nodal, DOF, and element fields)
        self._nodal_fields = {}    # Store nodal fields (1-indexed access)
        self._dof_fields = {}      # Store DOF fields (1-indexed access)
        self._element_fields = {}  # Store element fields (1-indexed access)

    def _auto_detect_element_type(self) -> str:
        """Auto-detect element type from connectivity."""
        if self.nodes_per_element == 2:
            return 'ROD'
        elif self.nodes_per_element == 3:
            return 'TRIA3'
        elif self.nodes_per_element == 4:
            # Ambiguous: could be QUAD4 (2D) or TETRA4 (3D)
            if self.n_dim == 2:
                return 'QUAD4'
            elif self.n_dim == 3:
                return 'TETRA4'
            else:
                raise ValueError(f"Cannot auto-detect 4-node element type for {self.n_dim}D mesh")
        elif self.nodes_per_element == 8:
            return 'HEXA8'
        else:
            raise ValueError(
                f"Cannot auto-detect element type for {self.nodes_per_element} nodes per element. "
                "Please specify element_type explicitly."
            )

    def _determine_dofs_per_node(self) -> int:
        """Determine DOFs per node from element type and dimension."""
        if self.element_type == 'ROD':
            return self.n_dim  # 2 for 2D, 3 for 3D
        elif self.element_type == 'BEAM':
            return 3 if self.n_dim == 2 else 6  # Includes rotations
        elif self.element_type in ['TRIA3', 'QUAD4']:
            return 2  # Only displacements in 2D
        elif self.element_type in ['TETRA4', 'HEXA8']:
            return 3  # Only displacements in 3D
        else:
            raise ValueError(f"Unknown element type: {self.element_type}")

    # ========================================
    # 1-Based Interface (for student use)
    # ========================================

    def get_node(self, node_number: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """
        Get coordinates of node(s) using 1-based numbering.

        Parameters
        ----------
        node_number : int or list of int
            Node number(s) (1, 2, 3, ...) as they appear in textbooks
            Can be a single int or a list/array of ints

        Returns
        -------
        coords : ndarray
            Node coordinates. Shape:
            - (n_dim,) if node_number is scalar
            - (n_nodes, n_dim) if node_number is list/array

        Examples
        --------
        >>> mesh.get_node(3)
        array([0., 1.])
        >>> mesh.get_node([1, 3])
        array([[0., 0.],
               [0., 1.]])

        Note
        ----
        This is where the translation happens: user passes node_number (1-based),
        we access self.nodes[node_number - 1] (0-based). Students can read the
        source code to understand this pattern.
        """
        if np.isscalar(node_number):
            self._validate_node_number(node_number)
            return self.nodes[node_number - 1]  # THE TRANSLATION
        else:
            node_numbers = np.atleast_1d(node_number)
            for n in node_numbers:
                self._validate_node_number(n)
            return self.nodes[node_numbers - 1]  # Vectorized translation

    def get_element(self, elem_number: Union[int, List[int], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get element information using 1-based numbering.

        Parameters
        ----------
        elem_number : int or list of int
            Element number(s) (1, 2, 3, ...) as they appear in textbooks

        Returns
        -------
        node_numbers : ndarray
            Node numbers (1-based). Shape:
            - (nodes_per_element,) if elem_number is scalar
            - (n_elements, nodes_per_element) if elem_number is list
        node_coords : ndarray
            Coordinates of the nodes. Shape:
            - (nodes_per_element, n_dim) if elem_number is scalar
            - (n_elements, nodes_per_element, n_dim) if elem_number is list

        Example
        -------
        >>> node_nums, coords = mesh.get_element(1)
        >>> print(f"Element 1 connects nodes {node_nums}")
        Element 1 connects nodes [1 2]
        >>> node_nums, coords = mesh.get_element([1, 2])
        """
        if np.isscalar(elem_number):
            self._validate_element_number(elem_number)
            indices_0based = self._elements_0based[elem_number - 1]
            node_numbers = indices_0based + 1  # Convert back to 1-based for output
            coords = self.nodes[indices_0based]
            return node_numbers, coords
        else:
            elem_numbers = np.atleast_1d(elem_number)
            for e in elem_numbers:
                self._validate_element_number(e)
            indices_0based = self._elements_0based[elem_numbers - 1]
            node_numbers = indices_0based + 1
            coords = self.nodes[indices_0based]
            return node_numbers, coords

    # ========================================
    # DOF Management - The Most Confusing Part
    # ========================================

    def dofs_for_node(self, node_number: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """
        Get DOF indices for node(s) (returns 0-based indices for array access).

        This method encapsulates the DOF mapping that confuses students most:

        **Mathematical notation** (from textbooks):
        - For 2D ROD: Node i has DOFs [2i-1, 2i] for [u_x, u_y]
        - Node 3 has DOFs [5, 6]

        **Python arrays** (0-based indexing):
        - Node 3 data is at index 2
        - Its DOFs are at indices [4, 5] in displacement vector

        This method handles that translation so students can write:
        >>> dofs = mesh.dofs_for_node(3)  # Think: "node 3"
        >>> u[dofs] = [0.1, 0.2]          # Access: indices [4, 5]

        Parameters
        ----------
        node_number : int or list of int
            Node number(s) (1-based)

        Returns
        -------
        dof_indices : ndarray
            DOF indices (0-based for direct use with displacement arrays)
            Shape:
            - (dofs_per_node,) if node_number is scalar
            - (n_nodes, dofs_per_node) if node_number is list

        Examples
        --------
        >>> mesh.dofs_for_node(3)
        array([4, 5])
        >>> mesh.dofs_for_node([1, 3])
        array([[0, 1],
               [4, 5]])
        """
        if np.isscalar(node_number):
            self._validate_node_number(node_number)
            # Formula: node i (1-based) → DOF indices [(i-1)*dofs_per_node, ..., i*dofs_per_node-1]
            dof_start = (node_number - 1) * self.dofs_per_node
            return np.arange(dof_start, dof_start + self.dofs_per_node)
        else:
            node_numbers = np.atleast_1d(node_number)
            for n in node_numbers:
                self._validate_node_number(n)
            # Vectorized: compute all DOF ranges
            dof_starts = (node_numbers - 1) * self.dofs_per_node
            # Create 2D array of DOF indices
            dofs = np.array([np.arange(start, start + self.dofs_per_node)
                           for start in dof_starts])
            return dofs

    def node_for_dof(self, dof_index: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """
        Get node number(s) (1-based) for DOF index/indices (0-based).

        Inverse of dofs_for_node. Useful for debugging and understanding results.

        Parameters
        ----------
        dof_index : int or list of int
            DOF index/indices (0-based)

        Returns
        -------
        node_number : int or ndarray
            Node number(s) (1-based)

        Examples
        --------
        >>> mesh.node_for_dof(4)
        3
        >>> mesh.node_for_dof([0, 1, 4])
        array([1, 1, 3])
        """
        if np.isscalar(dof_index):
            if not (0 <= dof_index < self.n_dofs):
                raise ValueError(f"DOF index must be between 0 and {self.n_dofs-1}")
            return (dof_index // self.dofs_per_node) + 1
        else:
            dof_indices = np.atleast_1d(dof_index)
            for d in dof_indices:
                if not (0 <= d < self.n_dofs):
                    raise ValueError(f"DOF index must be between 0 and {self.n_dofs-1}")
            return (dof_indices // self.dofs_per_node) + 1

    def dofs_for_element(self, elem_number: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """
        Get all DOF indices for element(s).

        Returns 0-based indices that can be used directly for assembly:
        >>> dofs = mesh.dofs_for_element(1)
        >>> K_global[np.ix_(dofs, dofs)] += K_elem

        Parameters
        ----------
        elem_number : int or list of int
            Element number(s) (1-based)

        Returns
        -------
        dof_indices : ndarray
            DOF indices (0-based). Shape:
            - (nodes_per_element * dofs_per_node,) if elem_number is scalar
            - (n_elements, nodes_per_element * dofs_per_node) if elem_number is list
        """
        if np.isscalar(elem_number):
            node_numbers, _ = self.get_element(elem_number)
            dofs = []
            for node_num in node_numbers:
                dofs.extend(self.dofs_for_node(node_num))
            return np.array(dofs)
        else:
            elem_numbers = np.atleast_1d(elem_number)
            all_dofs = []
            for elem in elem_numbers:
                node_numbers, _ = self.get_element(elem)
                dofs = []
                for node_num in node_numbers:
                    dofs.extend(self.dofs_for_node(node_num))
                all_dofs.append(dofs)
            return np.array(all_dofs)

    # ========================================
    # Iteration Helpers (Seamless Workflow)
    # ========================================

    def element_numbers(self):
        """
        Iterate over element numbers (1-based).

        Yields element numbers 1, 2, 3, ..., n_elements.
        Perfect for seamless iteration without range(1, n+1) clutter.

        Yields
        ------
        elem_number : int
            Element number (1-based)

        Examples
        --------
        >>> # Clean iteration (no more range(1, mesh.n_elements + 1)!)
        >>> for iel in mesh.element_numbers():
        ...     node_nums, coords = mesh.get_element(iel)
        ...     print(f"Element {iel}: nodes {node_nums}")
        Element 1: nodes [1 2]
        Element 2: nodes [1 3]
        ...

        >>> # Works seamlessly with OneArray
        >>> from matkit import OneArray
        >>> N = OneArray([100, 200, 300])
        >>> for iel in mesh.element_numbers():
        ...     print(f"Element {iel}: Force = {N[iel]}")
        Element 1: Force = 100
        Element 2: Force = 200
        Element 3: Force = 300
        """
        for i in range(1, self.n_elements + 1):
            yield i

    def node_numbers(self):
        """
        Iterate over node numbers (1-based).

        Yields node numbers 1, 2, 3, ..., n_nodes.
        Perfect for seamless iteration without range(1, n+1) clutter.

        Yields
        ------
        node_number : int
            Node number (1-based)

        Examples
        --------
        >>> # Clean iteration
        >>> for inode in mesh.node_numbers():
        ...     coord = mesh.get_node(inode)
        ...     print(f"Node {inode}: {coord}")
        Node 1: [0. 0.]
        Node 2: [1. 0.]
        ...

        >>> # Works seamlessly with OneArray for nodal results
        >>> u = OneArray([0.1, 0.2, 0.3, 0.4])  # Nodal displacements
        >>> for inode in mesh.node_numbers():
        ...     dofs = mesh.dofs_for_node(inode)
        ...     print(f"Node {inode}: u = {u.data[dofs]}")
        """
        for i in range(1, self.n_nodes + 1):
            yield i

    def iter_elements(self):
        """
        Iterate over elements yielding (number, node_numbers, coords).

        This is the most convenient way to iterate over elements, providing
        all necessary information in one call.

        Yields
        ------
        elem_number : int
            Element number (1-based)
        node_numbers : ndarray
            Node numbers for this element (1-based)
        coords : ndarray, shape (nodes_per_element, n_dim)
            Coordinates of element nodes

        Examples
        --------
        >>> # Ultra-clean iteration with all data
        >>> for iel, node_nums, coords in mesh.iter_elements():
        ...     print(f"Element {iel}: nodes {node_nums}")
        ...     print(f"  Coordinates:\\n{coords}")
        Element 1: nodes [1 2]
          Coordinates:
          [[0. 0.]
           [1. 0.]]
        ...

        >>> # Perfect for assembly loops
        >>> from matkit import OneArray
        >>> N = OneArray([100, 200, 300])
        >>> for iel, node_nums, coords in mesh.iter_elements():
        ...     force = N[iel]  # Seamless 1-based access!
        ...     length = np.linalg.norm(coords[1] - coords[0])
        ...     print(f"Element {iel}: Force = {force}, Length = {length:.2f}")
        """
        for i in range(1, self.n_elements + 1):
            node_numbers, coords = self.get_element(i)
            yield i, node_numbers, coords

    def iter_nodes(self):
        """
        Iterate over nodes yielding (number, coords).

        Convenient way to iterate over nodes with coordinates.

        Yields
        ------
        node_number : int
            Node number (1-based)
        coords : ndarray, shape (n_dim,)
            Coordinates of this node

        Examples
        --------
        >>> # Clean iteration with coordinates
        >>> for inode, coords in mesh.iter_nodes():
        ...     print(f"Node {inode}: {coords}")
        Node 1: [0. 0.]
        Node 2: [1. 0.]
        ...

        >>> # Perfect for applying boundary conditions
        >>> fixed_nodes = []
        >>> for inode, coords in mesh.iter_nodes():
        ...     if coords[1] == 0:  # Nodes on bottom edge
        ...         fixed_nodes.append(inode)
        """
        for i in range(1, self.n_nodes + 1):
            yield i, self.get_node(i)

    # ========================================
    # Field Management (1-Based Access)
    # ========================================

    def Add_Nodal_Field(self, name: str, data: Union[List, np.ndarray]) -> OneArray:
        """
        Add a scalar or vector field to each node with 1-based access.

        This method stores field data (e.g., displacements, nodal forces, temperatures)
        and wraps it in a OneArray for natural 1-based indexing.

        Parameters
        ----------
        name : str
            Field name (e.g., 'displacements', 'forces', 'temperature')
        data : array-like
            Field values. Can be:
            - Scalar field: shape (n_nodes,) - one value per node
            - Vector field: shape (n_nodes, n_components) - vector per node
              (e.g., [ux, uy] displacements)

        Returns
        -------
        field : OneArray
            The stored field with 1-based access

        Examples
        --------
        >>> # Scalar field (e.g., nodal temperatures)
        >>> T = mesh.Add_Nodal_Field('temperature', [20.0, 25.0, 30.0, 22.0])
        >>> print(T[1])  # Temperature at node 1
        20.0

        >>> # Vector field (e.g., nodal displacements in 2D)
        >>> U = mesh.Add_Nodal_Field('displacements',
        ...     [[0.0, 0.0], [0.1, 0.2], [0.15, 0.25], [0.0, 0.0]])
        >>> print(U[2])  # Displacement vector at node 2
        [0.1 0.2]

        >>> # Access stored field later
        >>> T = mesh.Get_Nodal_Field('temperature')
        >>> print(T[3])  # Temperature at node 3
        30.0

        Notes
        -----
        - Field is stored with 1-BASED indexing via OneArray
        - Use field[i] to access data for node i (i = 1, 2, 3, ...)
        - The field is also stored internally and can be retrieved with Get_Nodal_Field()
        """
        data = np.array(data)

        # Validate data shape
        if data.ndim == 1:
            # Scalar field: one value per node
            if len(data) != self.n_nodes:
                raise ValueError(
                    f"Scalar field must have {self.n_nodes} values (one per node), "
                    f"got {len(data)}"
                )
        elif data.ndim == 2:
            # Vector field: one vector per node
            if data.shape[0] != self.n_nodes:
                raise ValueError(
                    f"Vector field must have {self.n_nodes} rows (one per node), "
                    f"got {data.shape[0]}"
                )
        else:
            raise ValueError(
                f"Nodal field must be 1D (scalar) or 2D (vector), got {data.ndim}D"
            )

        # Wrap in OneArray for 1-based access
        field = OneArray(data)
        self._nodal_fields[name] = field
        return field

    def Add_DoF_Field(self, name: str, data: Union[List, np.ndarray]) -> OneArray:
        """
        Add a field to each degree of freedom with 1-based access.

        This method stores DOF-level data (e.g., displacements as a flat vector,
        nodal forces as a flat vector) with 1-based indexing.

        Parameters
        ----------
        name : str
            Field name (e.g., 'displacement_vector', 'force_vector')
        data : array-like, shape (n_dofs,)
            Field values. Must have length n_dofs.
            For 2D problems: [u1x, u1y, u2x, u2y, u3x, u3y, ...]

        Returns
        -------
        field : OneArray
            The stored field with 1-based access

        Examples
        --------
        >>> # Displacement vector (DOF-level)
        >>> u = mesh.Add_DoF_Field('displacements',
        ...     [0.0, 0.0, 0.1, 0.2, 0.15, 0.25, 0.0, 0.0])
        >>> print(u[1])  # First DOF (u1x)
        0.0
        >>> print(u[3])  # Third DOF (u2x)
        0.1

        >>> # Force vector
        >>> f = mesh.Add_DoF_Field('forces', [0, 0, 100, 0, 0, -50, 0, 0])
        >>> print(f[4])  # Fourth DOF (f2y)
        0

        >>> # Access stored field later
        >>> u = mesh.Get_DoF_Field('displacements')

        Notes
        -----
        - Field is stored with 1-BASED indexing via OneArray
        - Use field[i] to access DOF i (i = 1, 2, 3, ..., n_dofs)
        - DOF ordering follows mesh.dofs_for_node() convention
        """
        data = np.array(data)

        # Validate data shape
        if data.ndim != 1:
            raise ValueError(f"DOF field must be 1D, got {data.ndim}D")
        if len(data) != self.n_dofs:
            raise ValueError(
                f"DOF field must have {self.n_dofs} values (one per DOF), "
                f"got {len(data)}"
            )

        # Wrap in OneArray for 1-based access
        field = OneArray(data)
        self._dof_fields[name] = field
        return field

    def Add_Element_Field(self, name: str, data: Union[List, np.ndarray]) -> OneArray:
        """
        Add a field to each element with 1-based access.

        This method stores element-level data (e.g., element forces, areas,
        Young's modulus) with 1-based indexing.

        Parameters
        ----------
        name : str
            Field name (e.g., 'forces', 'areas', 'E_modulus')
        data : array-like, shape (n_elements,)
            Field values. Must have length n_elements.

        Returns
        -------
        field : OneArray
            The stored field with 1-based access

        Examples
        --------
        >>> # Element forces (axial forces in truss elements)
        >>> N = mesh.Add_Element_Field('forces',
        ...     [6052.76, -5582.25, -7274.51, 6380.16, -9912.07])
        >>> print(N[1])  # Force in element 1
        6052.76
        >>> print(N[3])  # Force in element 3
        -7274.51

        >>> # Element areas
        >>> A = mesh.Add_Element_Field('areas', [0.01, 0.01, 0.01, 0.01, 0.01])
        >>> print(A[2])  # Area of element 2
        0.01

        >>> # Young's modulus per element
        >>> E = mesh.Add_Element_Field('E_modulus',
        ...     [200e9, 200e9, 200e9, 200e9, 200e9])
        >>> print(E[1])  # E for element 1
        200000000000.0

        >>> # Seamless iteration
        >>> for iel in mesh.element_numbers():
        ...     force = N[iel]
        ...     area = A[iel]
        ...     stress = force / area
        ...     print(f"Element {iel}: σ = {stress:.2f} Pa")

        >>> # Access stored field later
        >>> N = mesh.Get_Element_Field('forces')

        Notes
        -----
        - Field is stored with 1-BASED indexing via OneArray
        - Use field[i] to access data for element i (i = 1, 2, 3, ...)
        - Works seamlessly with mesh.element_numbers() iteration
        """
        data = np.array(data)

        # Validate data shape
        if data.ndim != 1:
            raise ValueError(f"Element field must be 1D, got {data.ndim}D")
        if len(data) != self.n_elements:
            raise ValueError(
                f"Element field must have {self.n_elements} values (one per element), "
                f"got {len(data)}"
            )

        # Wrap in OneArray for 1-based access
        field = OneArray(data)
        self._element_fields[name] = field
        return field

    def Get_Nodal_Field(self, name: str) -> OneArray:
        """
        Retrieve a previously stored nodal field.

        Parameters
        ----------
        name : str
            Field name

        Returns
        -------
        field : OneArray
            The stored field with 1-based access

        Raises
        ------
        KeyError
            If field name not found
        """
        if name not in self._nodal_fields:
            raise KeyError(
                f"Nodal field '{name}' not found. "
                f"Available fields: {list(self._nodal_fields.keys())}"
            )
        return self._nodal_fields[name]

    def Get_DoF_Field(self, name: str) -> OneArray:
        """
        Retrieve a previously stored DOF field.

        Parameters
        ----------
        name : str
            Field name

        Returns
        -------
        field : OneArray
            The stored field with 1-based access

        Raises
        ------
        KeyError
            If field name not found
        """
        if name not in self._dof_fields:
            raise KeyError(
                f"DOF field '{name}' not found. "
                f"Available fields: {list(self._dof_fields.keys())}"
            )
        return self._dof_fields[name]

    def Get_Element_Field(self, name: str) -> OneArray:
        """
        Retrieve a previously stored element field.

        Parameters
        ----------
        name : str
            Field name

        Returns
        -------
        field : OneArray
            The stored field with 1-based access

        Raises
        ------
        KeyError
            If field name not found
        """
        if name not in self._element_fields:
            raise KeyError(
                f"Element field '{name}' not found. "
                f"Available fields: {list(self._element_fields.keys())}"
            )
        return self._element_fields[name]

    # ========================================
    # Validation Helpers
    # ========================================

    def _validate_node_number(self, node_number: int):
        """Validate node number is in valid 1-based range."""
        if not (1 <= node_number <= self.n_nodes):
            raise ValueError(
                f"Node number must be between 1 and {self.n_nodes}. "
                f"You provided: {node_number}. "
                f"Remember: we use 1-based numbering for nodes!"
            )

    def _validate_element_number(self, elem_number: int):
        """Validate element number is in valid 1-based range."""
        if not (1 <= elem_number <= self.n_elements):
            raise ValueError(
                f"Element number must be between 1 and {self.n_elements}. "
                f"You provided: {elem_number}. "
                f"Remember: we use 1-based numbering for elements!"
            )

    # ========================================
    # String Representation
    # ========================================

    def __repr__(self):
        return (f"Mesh(element_type='{self.element_type}', "
                f"n_nodes={self.n_nodes}, "
                f"n_elements={self.n_elements}, "
                f"n_dim={self.n_dim}D, "
                f"dofs_per_node={self.dofs_per_node})")

    def summary(self):
        """Print detailed mesh information with 1-based numbering."""
        print(f"Mesh Summary")
        print(f"=" * 60)
        print(f"Element type:       {self.element_type} - {ELEMENT_TYPES[self.element_type]['description']}")
        print(f"Number of nodes:    {self.n_nodes}")
        print(f"Number of elements: {self.n_elements}")
        print(f"Dimension:          {self.n_dim}D")
        print(f"DOFs per node:      {self.dofs_per_node}")
        print(f"Total DOFs:         {self.n_dofs}")
        print()
        print("Node Coordinates (1-based numbering):")
        print("-" * 60)
        for i in range(1, min(self.n_nodes + 1, 11)):  # Limit to first 10
            coord = self.get_node(i)
            print(f"  Node {i}: {coord}")
        if self.n_nodes > 10:
            print(f"  ... ({self.n_nodes - 10} more nodes)")
        print()
        print("Element Connectivity (1-based numbering):")
        print("-" * 60)
        for i in range(1, min(self.n_elements + 1, 11)):  # Limit to first 10
            node_nums, _ = self.get_element(i)
            print(f"  Element {i}: Nodes {list(node_nums)}")
        if self.n_elements > 10:
            print(f"  ... ({self.n_elements - 10} more elements)")
