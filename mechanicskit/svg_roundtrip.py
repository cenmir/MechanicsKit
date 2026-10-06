"""List the hand edits made to a generated figure, in the figure's own data units.

A figure written by sketch_extra.save_svg carries an id on every drawn part and a map
from data to SVG units. Open it in Inkscape, change it, save it as <name>.edited.svg,
and run

    python -m mechanicskit.svg_roundtrip path/to/<name>.svg

(the edited file is found next to it; a second path names it explicitly). The report
lists, per part, how far it moved in data units, what styles changed, what was deleted
and what was drawn new, which is what the figure script needs to reproduce the edit.
"""
import json
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

SVG = '{http://www.w3.org/2000/svg}'
XLINK = '{http://www.w3.org/1999/xlink}'
DRAWN = {'path', 'use', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'image'}
STYLE_KEYS = ('fill', 'stroke', 'stroke-width', 'opacity', 'fill-opacity', 'stroke-opacity',
              'stroke-dasharray')


# --- geometry -------------------------------------------------------------------------

def _affine(transform):
    """The 3x3 matrix of an SVG transform attribute."""
    M = np.eye(3)
    for name, args in re.findall(r'(\w+)\s*\(([^)]*)\)', transform or ''):
        v = [float(a) for a in re.split(r'[\s,]+', args.strip()) if a]
        if name == 'translate':
            T = np.array([[1, 0, v[0]], [0, 1, v[1] if len(v) > 1 else 0], [0, 0, 1]])
        elif name == 'scale':
            T = np.diag([v[0], v[1] if len(v) > 1 else v[0], 1])
        elif name == 'rotate':
            a = np.radians(v[0])
            T = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            if len(v) == 3:
                T = (np.array([[1, 0, v[1]], [0, 1, v[2]], [0, 0, 1]]) @ T
                     @ np.array([[1, 0, -v[1]], [0, 1, -v[2]], [0, 0, 1]]))
        elif name == 'matrix':
            T = np.array([[v[0], v[2], v[4]], [v[1], v[3], v[5]], [0, 0, 1]])
        else:
            continue
        M = M @ T
    return M


def _path_points(d):
    """End and control points of a path, absolute; enough for a bounding box."""
    toks = re.findall(r'[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', d or '')
    pts, cur, start, cmd, i = [], np.zeros(2), np.zeros(2), None, 0
    nargs = {'M': 2, 'L': 2, 'T': 2, 'H': 1, 'V': 1, 'C': 6, 'S': 4, 'Q': 4, 'A': 7, 'Z': 0}
    while i < len(toks):
        if re.match(r'[A-Za-z]', toks[i]):
            cmd, i = toks[i], i + 1
            if cmd in 'Zz':
                cur = start.copy()
                continue
        n = nargs[cmd.upper()]
        v = [float(t) for t in toks[i:i + n]]
        i += n
        rel = cmd.islower()
        base = cur if rel else np.zeros(2)
        up = cmd.upper()
        if up == 'H':
            cur = np.array([v[0] + (cur[0] if rel else 0), cur[1]])
        elif up == 'V':
            cur = np.array([cur[0], v[0] + (cur[1] if rel else 0)])
        elif up == 'A':
            cur = base + np.array(v[5:7])
        else:
            xy = np.array(v).reshape(-1, 2) + base
            pts.extend(xy[:-1])
            cur = xy[-1]
        if up == 'M':
            start = cur.copy()
            cmd = 'l' if rel else 'L'                      # further pairs are line-tos
        pts.append(cur.copy())
    return np.array(pts) if pts else np.zeros((0, 2))


class Doc:
    """An SVG file with its parent links, ids and the round-trip metadata."""

    def __init__(self, path):
        self.root = ET.parse(path).getroot()
        self.parent = {c: p for p in self.root.iter() for c in p}
        self.ids, self.dup = {}, set()
        for el in self.root.iter():
            i = el.get('id')
            if i:
                if i in self.ids:
                    self.dup.add(i)
                self.ids[i] = el
        meta = next((m for m in self.root.iter(SVG + 'metadata') if m.get('id') == 'roundtrip'),
                    None)
        self.meta = json.loads(meta.text) if meta is not None and meta.text else None

    def ctm(self, el):
        """The transform from ``el``'s own coordinates to the document's."""
        chain = []
        while el is not None:
            chain.append(el)
            el = self.parent.get(el)
        M = np.eye(3)
        for e in reversed(chain):
            M = M @ _affine(e.get('transform'))
        return M

    def _local_points(self, el):
        tag = el.tag.replace(SVG, '')
        f = lambda k, d=0.0: float(re.sub(r'[a-z%]+$', '', el.get(k, str(d))) or d)
        if tag == 'path':
            return _path_points(el.get('d'))
        if tag == 'use':
            ref = self.ids.get((el.get(XLINK + 'href') or el.get('href') or '').lstrip('#'))
            if ref is None:
                return np.zeros((0, 2))
            p = self._local_points(ref)
            if not len(p):
                return p
            # the referenced shape's own transform (glyphs are stored scaled), then x, y
            p = (np.c_[p, np.ones(len(p))] @ _affine(ref.get('transform')).T)[:, :2]
            return p + np.array([f('x'), f('y')])
        if tag == 'rect':
            x, y, w, h = f('x'), f('y'), f('width'), f('height')
            return np.array([[x, y], [x + w, y + h]])
        if tag in ('circle', 'ellipse'):
            rx = f('r') if tag == 'circle' else f('rx')
            ry = f('r') if tag == 'circle' else f('ry')
            return np.array([[f('cx') - rx, f('cy') - ry], [f('cx') + rx, f('cy') + ry]])
        if tag == 'line':
            return np.array([[f('x1'), f('y1')], [f('x2'), f('y2')]])
        if tag in ('polyline', 'polygon'):
            v = [float(t) for t in re.split(r'[\s,]+', el.get('points', '').strip()) if t]
            return np.array(v).reshape(-1, 2)
        return np.zeros((0, 2))

    def bbox(self, el):
        """The bounding box of everything drawn under ``el``, in document units."""
        pts = []
        for e in el.iter():
            if e.tag.replace(SVG, '') in DRAWN and not self.hidden(e):
                p = self._local_points(e)
                if len(p):
                    M = self.ctm(e)
                    pts.append((np.c_[p, np.ones(len(p))] @ M.T)[:, :2])
        if not pts:
            return None
        p = np.vstack(pts)
        return np.r_[p.min(axis=0), p.max(axis=0)]

    def hidden(self, el):
        while el is not None:
            st = _style(el)
            if st.get('display') == 'none' or st.get('visibility') == 'hidden':
                return True
            el = self.parent.get(el)
        return False


def _style(el):
    st = dict(kv.split(':', 1) for kv in (el.get('style') or '').split(';') if ':' in kv)
    st = {k.strip(): v.strip() for k, v in st.items()}
    for k in STYLE_KEYS + ('display', 'visibility'):
        if el.get(k) is not None and k not in st:
            st[k] = el.get(k)
    return st


def _styles(doc, el):
    """The drawing styles found under ``el``: {key: set of values}."""
    out = {}
    for e in el.iter():
        for k, v in _style(e).items():
            if k in STYLE_KEYS:
                out.setdefault(k, set()).add(v)
    return out


# --- the report ----------------------------------------------------------------------

def report(gen_path, edit_path, tol=0.25):
    gen, edit = Doc(gen_path), Doc(edit_path)
    if gen.meta is None:
        sys.exit(f'{gen_path} has no round-trip metadata; regenerate it with save_svg')
    ax = gen.meta['axes'][0]
    sx, sy = ax['scale']
    ox, oy = ax['offset']
    to_data = lambda p: ((p[0] - ox)/sx, (p[1] - oy)/sy)
    items = gen.meta['items']
    lines = []

    for gid, info in items.items():
        what = f"{gid}" + (f"  [{info['text']}]" if info.get('text') else '')
        g, e = gen.ids.get(gid), edit.ids.get(gid)
        if g is None:
            continue
        if e is None or edit.hidden(e):
            lines.append(f'deleted   {what}')
            continue
        bg, be = gen.bbox(g), edit.bbox(e)
        if bg is not None and be is not None:
            cg, ce = (bg[:2] + bg[2:])/2, (be[:2] + be[2:])/2
            size_g, size_e = bg[2:] - bg[:2], be[2:] - be[:2]
            moved = np.abs(ce - cg).max() > tol
            resized = np.any(np.abs(size_e - size_g) > max(tol, 0.02*np.abs(size_g).max()))
            if moved:
                dx, dy = (ce[0] - cg[0])/sx, (ce[1] - cg[1])/sy
                lines.append(f'moved     {what}  by ({dx:+.4g}, {dy:+.4g}) data units '
                             f'to centre ({to_data(ce)[0]:.4g}, {to_data(ce)[1]:.4g})')
            if resized:
                f = np.divide(size_e, size_g, out=np.ones(2), where=size_g > 1e-9)
                lines.append(f'resized   {what}  by x{f[0]:.3g} (width), x{f[1]:.3g} (height)')
        sg, se = _styles(gen, g), _styles(edit, e)
        for k in sorted(set(sg) | set(se)):
            if sg.get(k) != se.get(k):
                lines.append(f'restyled  {what}  {k}: {sorted(sg.get(k, []))} -> '
                             f'{sorted(se.get(k, []))}')

    # parts drawn by hand: drawable elements under no known id that match nothing the
    # script drew (the figure background and other unnamed pieces match themselves)
    def signature(doc, el):
        b = doc.bbox(el)
        return None if b is None else (el.tag, tuple(np.round(b, 1)))

    def loose(doc, known):
        for el in doc.root.iter():
            if el.tag.replace(SVG, '') not in DRAWN or doc.hidden(el):
                continue
            a = el
            while a is not None and a not in known and a.tag not in (
                    SVG + 'defs', SVG + 'metadata', SVG + 'clipPath'):
                a = doc.parent.get(a)
            if a is None:
                yield el

    gen_known = {gen.ids[i] for i in items if i in gen.ids}
    edit_known = {edit.ids[i] for i in items if i in edit.ids}
    unnamed = {signature(gen, el) for el in loose(gen, gen_known)}
    for el in loose(edit, edit_known):
        if signature(edit, el) in unnamed:
            continue
        b = edit.bbox(el)
        if b is None:
            continue
        p0, p1 = to_data(b[:2]), to_data(b[2:])
        st = {k: v for k, v in _style(el).items() if k in STYLE_KEYS}
        lines.append(f"new       {el.tag.replace(SVG, '')} {el.get('id', '')}  from "
                     f"({min(p0[0], p1[0]):.4g}, {min(p0[1], p1[1]):.4g}) to "
                     f"({max(p0[0], p1[0]):.4g}, {max(p0[1], p1[1]):.4g})  {st}")
    return lines


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    gen = argv[1]
    edit = argv[2] if len(argv) > 2 else re.sub(r'\.svg$', '.edited.svg', gen)
    lines = report(gen, edit)
    print(f'{edit}\n  compared with {gen}\n')
    print('\n'.join(lines) if lines else 'no changes found')


if __name__ == '__main__':
    main(sys.argv)
