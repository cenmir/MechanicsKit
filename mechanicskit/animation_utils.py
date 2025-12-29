"""
Animation utilities for mechanicskit.

Provides helper functions for creating interactive animations in Jupyter/marimo notebooks.
"""

from IPython.display import HTML


def to_responsive_html(anim, container_id='anim-container', autoplay=True):
    """
    Convert a matplotlib FuncAnimation to responsive HTML with optional autoplay.

    This function wraps the animation HTML with CSS for responsive sizing and
    optionally adds JavaScript to automatically start the animation when loaded.

    Parameters
    ----------
    anim : matplotlib.animation.FuncAnimation
        The animation object to convert
    container_id : str, optional
        HTML ID for the container div (default: 'anim-container')
    autoplay : bool, optional
        Whether to automatically start the animation (default: True)

    Returns
    -------
    IPython.display.HTML
        HTML object that can be displayed in Jupyter/marimo notebooks

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.animation import FuncAnimation
    >>> import mechanicskit as mk
    >>>
    >>> fig, ax = plt.subplots()
    >>> def animate(frame):
    ...     ax.clear()
    ...     ax.plot([0, frame], [0, frame])
    >>>
    >>> anim = FuncAnimation(fig, animate, frames=10)
    >>> plt.close()
    >>> mk.to_responsive_html(anim)  # Returns HTML with autoplay

    Notes
    -----
    - The animation loops by default (uses `default_mode='loop'`)
    - The responsive CSS ensures the animation scales to fit the container width
    - Autoplay uses JavaScript to click the play button after the page loads
    - The container ID must be unique if multiple animations are on the same page
    """
    # Get the HTML with loop mode enabled
    anim_html = anim.to_jshtml(default_mode='loop')

    # Create autoplay script if requested
    autoplay_script = ''
    if autoplay:
        autoplay_script = f"""
    <script>
    (function() {{
        // Function to find the play button and click it
        const startAnimation = () => {{
            // Matplotlib JS player uses titles for buttons.
            // We look for the button with the title "Play" inside our container.
            const container = document.getElementById('{container_id}');
            const playButton = container.querySelector('button[title="Play"]');

            if (playButton) {{
                playButton.click();
                console.log("Animation started automatically (container: {container_id})");
            }} else {{
                // If the player hasn't rendered yet, try again in 200ms
                setTimeout(startAnimation, 200);
            }}
        }};

        // Start checking once the window has loaded
        if (document.readyState === 'complete') {{
            startAnimation();
        }} else {{
            window.addEventListener('load', startAnimation);
        }}
    }})();
    </script>"""

    # Wrap with CSS for responsive sizing and autoplay script
    responsive_html = f"""
<div id="{container_id}" style="width: 100%; max-width: 100%; overflow: hidden;">
    <style>
        .animation-container {{
            width: 100% !important;
            max-width: 100% !important;
        }}
        .animation-container img {{
            width: 100% !important;
            height: auto !important;
        }}
    </style>
    <div class="animation-container">
        {anim_html}
    </div>
    {autoplay_script}
</div>
"""

    return HTML(responsive_html)
