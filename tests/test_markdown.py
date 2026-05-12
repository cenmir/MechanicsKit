"""
Tests for mechanicskit.markdown
================================

Covers ``mk.md``: caller-frame interpolation, format specs, arithmetic
placeholders, brace escaping, and the type-dispatch table (LatexArray /
LatexExpression / ndarray / DataFrame / list-of-list / list-of-dict).
"""
import numpy as np
import pytest

from mechanicskit import md, Markdown, la, ltx


class TestBasicInterpolation:
    """Caller-frame variable lookup, kwargs, and nameerror behavior."""

    def test_scalar_passthrough(self):
        a = 42
        assert str(md("a = {a}")) == "a = 42"

    def test_format_spec(self):
        a = 3.14159
        assert str(md("a = {a:.2f}")) == "a = 3.14"

    def test_arithmetic_in_placeholder(self):
        F_st = 12345.0
        assert str(md("F = {F_st/1000:.2f} kN")) == "F = 12.35 kN"

    def test_attribute_access(self):
        class Holder:
            value = 12

        h = Holder()
        assert str(md("M{h.value}")) == "M12"

    def test_method_call_in_placeholder(self):
        s = "hello"
        assert str(md("{s.upper()}")) == "HELLO"

    def test_kwargs_override_frame(self):
        x = 1
        assert str(md("{x}", x=99)) == "99"

    def test_kwargs_introduce_new_name(self):
        # Variable not in caller scope, only as kwarg
        out = str(md("F = {F:.2f}", F=2.71828))
        assert out == "F = 2.72"

    def test_missing_variable_raises_nameerror(self):
        with pytest.raises(NameError, match="mk.md: name not found"):
            md("{undefined_xyz}")

    def test_brace_escaping_for_latex(self):
        # Doubled braces -> literal { and } in output, matching str.format
        F1 = 5
        out = str(md(r"$F = \dfrac{{1}}{{2}} {F1}$"))
        assert out == r"$F = \dfrac{1}{2} 5$"


class TestReturnedObject:
    """The Markdown wrapper exposes the right rendering protocols."""

    def test_str_returns_text(self):
        m = md("hello {x}", x=1)
        assert str(m) == "hello 1"

    def test_repr_markdown_for_jupyter_quarto(self):
        m = md("hello {x}", x=1)
        assert m._repr_markdown_() == "hello 1"

    def test_mime_for_marimo(self):
        m = md("hello {x}", x=1)
        mime, body = m._mime_()
        assert mime == "text/markdown"
        assert body == "hello 1"

    def test_format_returns_source_for_nesting(self):
        # Nested mk.md under f-string should preserve markdown source,
        # not flatten to repr — same trick marimo's _md uses.
        inner = md("inner {a:.1f}", a=2.5)
        assert f"[{inner}]" == "[inner 2.5]"

    def test_nested_md_via_kwarg(self):
        inner = md("inner {a:.1f}", a=2.5)
        outer = md("wrap [{inner}]", inner=inner)
        assert str(outer) == "wrap [inner 2.5]"


class TestTypeDispatch:
    """Render dispatch on built-in and MechanicsKit types."""

    def test_ndarray_2d_small_renders_markdown_table(self):
        A = np.array([[1, 2], [3, 4]])
        out = str(md("{A}"))
        assert "|" in out
        assert "---" in out
        assert "1" in out and "4" in out

    def test_ndarray_2d_large_renders_truncated_bmatrix(self):
        # >8x8 falls back to LaTeX bmatrix
        B = np.arange(81).reshape(9, 9).astype(float)
        out = str(md("{B}"))
        assert r"\begin{bmatrix}" in out
        assert r"\cdots" in out  # truncation indicator

    def test_ndarray_1d_inline(self):
        v = np.array([1.0, 2.0, 3.0])
        out = str(md("eigs = {v}"))
        assert out == "eigs = 1, 2, 3"

    def test_ndarray_0d_scalar(self):
        s = np.array(7.5)
        out = str(md("{s}"))
        assert "7.5" in out

    def test_latex_array_uses_repr_latex(self):
        A = np.array([[1, 2], [3, 4]])
        out = str(md("{A_la}", A_la=la(A)))
        assert "$$" in out
        assert r"\begin{bmatrix}" in out

    def test_latex_expression_uses_repr_latex(self):
        A = np.array([[1, 2], [3, 4]])
        expr = ltx(r"K = ", A)
        out = str(md("{expr}", expr=expr))
        assert "$$" in out
        assert "K =" in out

    def test_list_of_lists(self):
        data = [[1, 2, 3], [4, 5, 6]]
        out = str(md("{data}"))
        assert "|" in out
        assert "---" in out
        assert "1" in out and "6" in out

    def test_list_of_dicts(self):
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        out = str(md("{data}"))
        assert "| a" in out or "a" in out  # header row
        assert "| b" in out or "b" in out

    def test_empty_list_passthrough(self):
        out = str(md("{x}", x=[]))
        assert out == "[]"

    def test_unknown_type_falls_back_to_str(self):
        # Lenient: matches f-string / mo.md behavior on unrecognized types
        out = str(md("{x}", x=None))
        assert out == "None"


class TestPandasDispatch:
    """DataFrame dispatch, lazily registered."""

    def test_dataframe_renders_markdown_table(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"node": [1, 2, 3], "u_x": [0.0, 1.2, 2.4]})
        out = str(md("{df}"))
        assert "node" in out
        assert "u_x" in out
        assert "---" in out


class TestRealExample:
    """The bolt-table example from the design doc, end-to-end."""

    def test_parameter_table_renders(self):
        class Std:
            value = 12

        skruvStandard = Std()
        P = 1.75
        F_st = 89000.0
        M_act = 84.5  # M_åt with safe ASCII

        out = str(md(r"""
| Beteckning | Beskrivning                | Värde                  | Enhet      |
|:---        |:---                        |:---                    |:---        |
| $d$        | Skruvgäng metrisk diameter | M{skruvStandard.value} | M standard |
| $P$        | Gängstigning grov          | {P}                    | mm         |
| $F_{{st}}$ | Sträckkraft                | {F_st/1000:0.2f}       | kN         |
| $M_{{act}}$| Åtdragningsmoment          | {M_act:0.2f}           | Nm         |
"""))
        assert "M12" in out
        assert "1.75" in out
        assert "89.00" in out  # 89000 / 1000 -> 89.00
        assert "84.50" in out
        assert "$F_{st}$" in out
        assert "$M_{act}$" in out


class TestMarkdownClass:
    """The Markdown wrapper itself."""

    def test_construct_directly(self):
        m = Markdown("# heading")
        assert str(m) == "# heading"
        assert m._repr_markdown_() == "# heading"
        assert m._mime_() == ("text/markdown", "# heading")
