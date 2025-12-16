"""Quick reference guide for MechanicsKit."""

def quick_ref():
    """
    Display quick reference for MechanicsKit functions.

    Usage: mk.quick_ref()
    """
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                    MechanicsKit Quick Reference                       ║
╚═══════════════════════════════════════════════════════════════════════╝

📊 VISUALIZATION
────────────────
  mk.patch(Faces=..., Vertices=..., CData=..., FaceColor='interp', cmap='jet')
      Create colored patches (MATLAB-style visualization)
      Parameters: Faces, Vertices, CData/FaceVertexCData, FaceColor,
                  EdgeColor, LineWidth, cmap, vmin, vmax, FaceAlpha

  mk.fplot(f, interval, **kwargs)
      Plot symbolic or lambda functions (MATLAB-style fplot)

  mk.colorbar(ax=ax, **kwargs)
      Add colorbar (auto-detects patch data)

  mk.cmap(name)
      Get colormap by name

🔢 ARRAYS & INDEXING
─────────────────────
  mk.OneArray(data)
      1-based indexing array wrapper for FEM results
      Access: arr[1], arr[2], ... instead of arr[0], arr[1], ...

  mk.LatexArray(data, name='A')
  mk.la(data, name='A')
      LaTeX-rendered arrays for notebooks

🌐 MESH & FEM
──────────────
  mk.Mesh(nodes, elements, element_type)
      Create FEM mesh with 1-based indexing

  mk.ELEMENT_TYPES
      Dict of available element types: ROD, BEAM, TRIA3, QUAD4, TETRA4, HEXA8

📚 EXAMPLES
────────────
  # Patch with interpolated colors
  import mechanicskit as mk
  P = [[0,0], [1,0], [1,1], [0,1]]
  U = [20, 30, 40, 25]
  mk.patch(Faces=[[1,2,3,4]], Vertices=P, CData=U,
           FaceColor='interp', cmap='jet')
  mk.colorbar()

  # Plot symbolic function
  from sympy import symbols, sin
  x = symbols('x')
  mk.fplot(sin(x), [-3.14, 3.14])

ℹ️  Help: help(mk.patch), help(mk.fplot), etc.
    """)

__all__ = ['quick_ref']
